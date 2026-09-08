# Co-registre rígid 3D del fetge en imatges de TC

Codi del Treball de Fi de Grau de Matemàtiques (UIB).

El projecte implementa un pipeline complet per a: (1) lectura de sèries DICOM i conversió de segmentacions RT-Struct a màscares volumètriques 3D, (2) segmentació automàtica del fetge amb MedSAM2, i (3) co-registre longitudinal entre instants temporals d'un mateix pacient mitjançant transformacions rígides optimitzades sobre diverses funcions de pèrdua.

## Estructura

- `src/` — mòduls del pipeline: càrrega de dades, transformacions rígides, mètriques, funcions de pèrdua, optimitzadors, segmentació i anàlisi estadística
- `tests/` — scripts de comprovació de cada component
- `resultats/` — taules d'experiments i històrics de convergència
- `figures/` — figures incloses a la memòria
- `docs/` — material complementari

Scripts principals a l'arrel: `run.py` i `run_mac.py` (co-registre), `segmentacio_pacients.py` (segmentació), `analisis_dades.py` (anàlisi estadística).

## Dades

Les imatges no s'inclouen al repositori. Provenen del dataset públic **HCC-TACE-Seg** (The Cancer Imaging Archive): sèries de TC hepàtica amb segmentacions ground truth en format DICOM RT-Struct.

## Model de segmentació

MedSAM2 no s'inclou en aquest repositori. Per reproduir la segmentació:

```
mkdir -p segmentation_models
cd segmentation_models
git clone https://github.com/bowang-lab/MedSAM2.git
cd MedSAM2
git checkout 332f30d
```

Els pesos es descarreguen segons les instruccions del repositori original i es
col·loquen a `segmentation_models/MedSAM2/checkpoints/`.

Configuració emprada en aquest treball:

- Checkpoint: `MedSAM2_latest.pt`
- Config del model: `sam2.1_hiera_t512` (arquitectura Hiera-Tiny, entrada 512x512)

Aquestes són les rutes per defecte de `src/segmentation_sam2.py`; es poden
sobreescriure passant `model_path` i `config_path` a `load_model()`.

## Entorns d'execució

- Segmentació i anàlisi estadística: MacBook Air M2 (PyTorch amb backend MPS)
- Co-registre (totes les configuracions per pacient): equip de la UIB

Els scripts detecten el dispositiu disponible automàticament; no s'assumeix CUDA.

