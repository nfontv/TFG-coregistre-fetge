"""
MedSAM2 inference wrapper implementing BaseSegmentationModel.

This module adapts the inference logic from the Jupyter notebook
`segmentation_models/MedSAM2/notebooks/MedSAM2_inference_HCC.ipynb` and
provides a safe fallback so tests can run when external MedSAM2
dependencies are unavailable.

Key behaviours:
- load_model: tries to build the real MedSAM2 predictor, otherwise keeps a stub
- preprocess: windowing and normalization consistent with the notebook
- predict: if a real predictor is present runs the notebook flow; otherwise
  produces a simple thresholded/connected-component mask as a stub
- postprocess: returns largest connected component as uint8 mask
- save_results: writes NIfTI via SimpleITK

This file intentionally keeps heavy imports inside try/except blocks so the
module can be imported in CI/test environments without MedSAM2 installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch

from .base_model import BaseSegmentationModel
from .environment_manager import EnvironmentManager
from ..utils import (extract_seg_from_dicom_seg,get_key_slice_from_seg_mask,
    get_largest_connected_component,
    extract_bbox_from_mask,
    resize_grayscale_to_rgb_and_resize,
    align_segmentation_with_volume,
    extract_bboxes_from_3d_mask,
)


class MedSAM2Model(BaseSegmentationModel):
    """MedSAM2 inference wrapper.

    Notes/assumptions:
    - The real MedSAM2 predictor builder (sam2.build_sam.build_sam2_video_predictor_npz)
      is used if importable. Otherwise a simple heuristic predictor is used.
    - Input to `infer` is expected to be a 3D numpy array (slices, H, W).
    """

    def __init__(self, model_name: str = "MedSAM2", env_name: str = "medsam2"):
        super().__init__(model_name=model_name, env_name=env_name)
        self.checkpoint = None
        self.model_cfg = None
        self.predictor = None
        # Default checkpoint path (relative to repository root). This keeps the
        # reference portable and version-controlled.
        repo_root = Path(__file__).resolve().parents[3]
        # ...
        self.default_checkpoint = repo_root / "segmentation_models" / "MedSAM2" / "checkpoints" / "MedSAM2_latest.pt"

        # The full path, used for validation to ensure the file exists
        self.default_config_path = repo_root / "segmentation_models" / "MedSAM2" / "sam2" / "configs" / "sam2.1_hiera_t512.yaml"

        # The config NAME, which the sam2 builder function expects
        self.default_config_name = "configs/sam2.1_hiera_t512"

        # lazy import target builder
        try:
            from sam2.build_sam import build_sam2_video_predictor_npz  # type: ignore

            self._build_predictor = build_sam2_video_predictor_npz
        except Exception:
            self._build_predictor = None

        # expose env manager
        self.env_manager = EnvironmentManager()

    def load_model(self, model_path: Optional[str] = None, config_path: Optional[str] = None,
                   device: str = "cuda") -> None:
        # Use provided checkpoint or fall back to the repository default
        if model_path is None:
            model_path = str(self.default_checkpoint)

        config_to_pass_to_builder = None
        config_path_to_validate = None

        if config_path is None:
            # DEFAULT case: Use the NAME for the builder
            config_to_pass_to_builder = self.default_config_name
            # Use the PATH for validation
            config_path_to_validate = self.default_config_path
        else:
            # USER-PROVIDED case: Assume they provided a valid name or path
            config_to_pass_to_builder = config_path
            # If it looks like a path, use it for validation
            if "/" in config_path or "\\" in config_path or ".yaml" in config_path:
                config_path_to_validate = Path(config_path)
            else:
                # It's just a name, validate the default path as a fallback check
                config_path_to_validate = self.default_config_path

        self.checkpoint = model_path
        self.model_cfg = config_to_pass_to_builder  # This will be "sam2.1_hiera_t512"

        # Validate checkpoint exists
        if not Path(model_path).exists():
            raise RuntimeError(
                f"Checkpoint not found: {model_path}. Please download the MedSAM2 checkpoint to this path.")

        # Validate config file exists
        if not Path(config_path_to_validate).exists():
            raise RuntimeError(f"Model config not found: {config_path_to_validate}")

        # ... (rest of the function: env activation, _build_predictor check) ...

        # try to construct the predictor and raise on failure
        try:
            # self.model_cfg will now be "sam2.1_hiera_t512" (in the default case)
            self.predictor = self._build_predictor(self.model_cfg, self.checkpoint)
        except Exception as e:
            raise RuntimeError(f"Failed to build MedSAM2 predictor: {e}")

    def extract_bbox_from_roi(self, roi_dcm_path: Union[str, Path], vol_metadata: Optional[dict] = None,
                              target_label: str = "Liver") -> Tuple[int, np.ndarray]:
        """Extract key slice index and bounding box from a ManualROI DICOM file.

        If vol_metadata is provided, it will align the segmentation to the
        volume's coordinate system before extracting the prompt.

        Returns (key_slice_idx, bbox)
        where bbox is [x_min, y_min, x_max, y_max]
        """
        # 1. Extract the raw, unaligned segmentation mask and info
        seg_mask_unaligned, bboxes_per_slice_orig, seg_info = extract_seg_from_dicom_seg(
            roi_dcm_path,
            extract_bboxes=True,
            target_label=target_label
        )
        if seg_mask_unaligned is None or seg_mask_unaligned.size == 0:
            raise RuntimeError(f"Empty segmentation in ROI DICOM: {roi_dcm_path}")

        key_slice_idx = get_key_slice_from_seg_mask(seg_mask_unaligned)
        bboxes_per_slice = extract_bboxes_from_3d_mask(seg_mask_unaligned)

        # NEW: Clip key_slice_idx to target volume's valid range
        if vol_metadata is not None:
            target_num_slices = vol_metadata.get('num_slices')
            if target_num_slices is not None and target_num_slices > 0:
                # Clip the frame index to [0, target_num_slices - 1]
                original_idx = key_slice_idx
                key_slice_idx = min(max(0, key_slice_idx), target_num_slices - 1)

                if original_idx != key_slice_idx:
                    print(
                        f"Clipped prompt frame index from {original_idx} to {key_slice_idx} (target volume has {target_num_slices} slices)")

                # Ensure we use a slice that actually has a bbox within the valid range
                valid_bbox_slices = [s for s in bboxes_per_slice.keys() if s < target_num_slices]

                if key_slice_idx in valid_bbox_slices:
                    return int(key_slice_idx), bboxes_per_slice[key_slice_idx]

                # Find nearest valid slice with bbox
                if valid_bbox_slices:
                    nearest_slice = min(valid_bbox_slices, key=lambda s: abs(s - key_slice_idx))
                    print(f"Using nearest valid slice {nearest_slice} instead of {key_slice_idx}")
                    return int(nearest_slice), bboxes_per_slice[nearest_slice]

            raise RuntimeError(f"No bounding boxes found within target volume range [0, {target_num_slices})")

        if key_slice_idx in bboxes_per_slice:
            return int(key_slice_idx), bboxes_per_slice[key_slice_idx]
        # If no bbox for key slice, pick nearest slice that has bbox
        if bboxes_per_slice:
            nearest_slice = sorted(bboxes_per_slice.keys(), key=lambda s: abs(s - key_slice_idx))[0]
            return int(nearest_slice), bboxes_per_slice[nearest_slice]

        raise RuntimeError("No bounding box found in ROI DICOM file")

    def extract_bbox_from_series(self, series) -> Tuple[int, np.ndarray]:
        """Given a `Series` object from the pipeline data loader, find an associated
        ManualROI DICOM and extract the key bbox. Returns (key_slice_idx, bbox).
        """
        # Prefer segmentation files listed on the series object
        seg_files = getattr(series, "segmentation_files", None) or []
        if seg_files:
            roi_path = seg_files[0]
            return self.extract_bbox_from_roi(roi_path)

        # Otherwise search the study directory for ManualROI files matching the series name
        series_dir = Path(series.series_path)
        study_dir = series_dir.parent
        pattern = f"{series.series_name}_ManualROI_*.dcm"
        matches = list(study_dir.glob(pattern))
        if matches:
            return self.extract_bbox_from_roi(matches[0])

        raise RuntimeError("No ManualROI segmentation file found for series")

    def preprocess(self, input_data: Any, window_type: str = "liver") -> Tuple[np.ndarray, dict]:
        """Apply DICOM-like windowing and convert to uint8 image like the notebook.

        Returns: tuple of (preprocessed_uint8_volume, metadata dict)
        """
        volume = np.asarray(input_data)
        if volume.ndim != 3:
            raise ValueError("input_data must be a 3D numpy array (slices, H, W)")

        # simple window presets
        windows = {
            "abdomen": (40, 400),
            "liver": (60, 150),
            "lung": (-600, 1500),
            "bone": (400, 1800),
            "soft_tissue": (50, 350),
        }

        if window_type in windows:
            level, width = windows[window_type]
            lower = level - width / 2
            upper = level + width / 2
        else:
            mean_val = np.mean(volume)
            std_val = np.std(volume)
            lower = mean_val - 2 * std_val
            upper = mean_val + 2 * std_val

        clipped = np.clip(volume, lower, upper)
        norm = (clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-12)
        img_uint8 = (norm * 255.0).astype(np.uint8)

        metadata = {"window": (lower, upper), "original_shape": volume.shape}
        return img_uint8, metadata

    def predict(self, preprocessed_data: Any, key_slice_idx: Optional[int] = None, bbox: Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """Run prediction. If predictor exists, run the MedSAM2 flow; otherwise
        raise an error stating the predictor is unavailable.
        """
        img_uint8 = preprocessed_data
        if img_uint8.ndim != 3:
            raise ValueError("preprocessed_data must be 3D array (slices, H, W)")
        # Ensure predictor and torch are available
        if self.predictor is None:
            raise RuntimeError(
                "MedSAM2 predictor is not initialized. Call load_model() with a valid checkpoint and ensure the sam2 package is installed."
            )

        if torch is None:
            raise RuntimeError("PyTorch is required for MedSAM2 inference but is not available in this environment.")

        # Run the notebook-like inference flow using the predictor
        img_resized = resize_grayscale_to_rgb_and_resize(img_uint8, 512)
        img_resized = img_resized / 255.0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        img_tensor = torch.from_numpy(img_resized).to(device=device, dtype=torch.float32)

        # normalise
        img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device)[:, None, None]
        img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device)[:, None, None]
        img_tensor = (img_tensor - img_mean) / img_std

        with torch.inference_mode():
            video_h = img_uint8.shape[1]
            video_w = img_uint8.shape[2]
            inference_state = self.predictor.init_state(img_tensor, video_h, video_w)

            if key_slice_idx is None:
                key_slice_idx = img_uint8.shape[0] // 2

            # If bbox is not provided, try to auto-extract it from available kwargs
            if bbox is None:
                # Accept a Series object, a path to roi_dcm, or a series_path
                series_obj = kwargs.get("series") or kwargs.get("series_obj")
                roi_dcm_path = kwargs.get("roi_dcm_path") or kwargs.get("roi_path")
                series_path = kwargs.get("series_path")

                # --- GET THE VOLUME METADATA PASSED FROM THE INFER SCRIPT ---
                vol_metadata = kwargs.get("vol_metadata")
                # --- GET THE TARGET LABEL FOR SEGMENT SELECTION ---
                target_label = kwargs.get("target_label", "Liver")

                if series_obj is not None:
                    try:
                        key_slice_idx, bbox = self.extract_bbox_from_series(series_obj)
                    except Exception as e:
                        raise ValueError(f"Failed to extract bbox from series: {e}")
                elif roi_dcm_path is not None:
                    try:
                        # --- PASS VOL_METADATA AND TARGET_LABEL TO THE PROMPT EXTRACTOR ---
                        key_slice_idx, bbox = self.extract_bbox_from_roi(
                            roi_dcm_path,
                            vol_metadata=vol_metadata,
                            target_label=target_label
                        )
                    except Exception as e:
                        raise ValueError(f"Failed to extract bbox from ROI DICOM: {e}")
                elif series_path is not None:
                    # try to find ManualROI in the parent study folder
                    try:
                        dummy_series = type("S", (),
                                            {"series_path": series_path, "series_name": Path(series_path).name})()
                        key_slice_idx, bbox = self.extract_bbox_from_series(dummy_series)
                    except Exception as e:
                        raise ValueError(f"Failed to extract bbox from series_path: {e}")
                else:
                    # require a bbox or prompt from caller; do not invent one
                    raise ValueError(
                        "bbox must be provided for MedSAM2 predictor-based inference or pass 'series'/'roi_dcm_path' in kwargs for automatic extraction")

            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=key_slice_idx,
                obj_id=1,
                box=bbox,
            )

            segs_3D = np.zeros(img_uint8.shape, dtype=np.uint8)

            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1

            self.predictor.reset_state(inference_state)

            # backward
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=key_slice_idx,
                obj_id=1,
                box=bbox,
            )

            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state,
                                                                                                 reverse=True):
                segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1

            self.predictor.reset_state(inference_state)

        # postprocess: largest CC
        if segs_3D.max() > 0:
            segs_3D = get_largest_connected_component(segs_3D)
            return np.uint8(segs_3D)
        return np.zeros_like(img_uint8, dtype=np.uint8)

    def infer(self, input_data: Any, **kwargs) -> np.ndarray:
        """
        Complete inference pipeline: preprocess -> predict -> postprocess.

        This method OVERRIDES the one in BaseSegmentationModel to correctly
        handle the (data, metadata) tuple returned by this class's
        preprocess method.
        """
        # 1. Preprocess the data
        #    This returns a tuple: (img_uint8, metadata)
        preprocessed_image, metadata = self.preprocess(input_data)

        # 2. Pass ONLY the image to predict.
        #    kwargs (like 'roi_dcm_path') are passed along.
        predictions = self.predict(preprocessed_image, **kwargs)

        # 3. Postprocess the predictions
        segmentation = self.postprocess(predictions)

        return segmentation

    def postprocess(self, predictions: Any) -> np.ndarray:
        arr = np.asarray(predictions)
        # ensure binary uint8
        return (arr > 0).astype(np.uint8)

    def save_results(self, segmentation: np.ndarray, output_path: Union[str, Path], **kwargs) -> None:
        if sitk is None:
            raise RuntimeError("SimpleITK not available; cannot save results")
        seg_img = sitk.GetImageFromArray(segmentation.astype(np.uint8))
        sitk.WriteImage(seg_img, str(output_path))

    # infer is inherited from BaseSegmentationModel (it calls preprocess->predict->postprocess)


__all__ = ["MedSAM2Model"]
