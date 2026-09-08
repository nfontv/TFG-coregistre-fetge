import os
from pathlib import Path
from src import dataset, utils, ai


BASE_PATH = Path(__file__).parent
#DATASET_PATH = Path("/Users/noemifontvoorhoeve/Desktop/TFG/manifest-1758635350325/HCC-TACE-Seg")
DATASET_PATH = Path("/Volumes/noemifont/TFG DATA/manifest-1774974966100/HCC-TACE-Seg")
TAULA_COMPLEMENTARIA_PATH = Path('/Users/noemifontvoorhoeve/Desktop/CoregistreRigidFetge-main/41597_2023_1928_MOESM1_ESM.xlsx')
DESCARTS_MANUALS = {'HCC_054',"HCC_089"}

# Carpeta on guardarem TOTES les segmentacions, reutilitzables per sempre
SEGMENTATIONS_PATH = BASE_PATH / "segmentations_MedSam2"
os.makedirs(SEGMENTATIONS_PATH, exist_ok=True)

#NOMES_PACIENTS = ["HCC_024","HCC_029","HCC_043","HCC_056","HCC_057","HCC_060","HCC_062","HCC_084","HCC_075", "HCC_089","HCC_093","HCC_092","HCC_095","HCC_099","HCC_104"]
#NOMES_PACIENTS = ["HCC_029"]
NOMES_PACIENTS = None

if __name__ == "__main__":
    my_dataset = dataset.load_dataset(DATASET_PATH, TAULA_COMPLEMENTARIA_PATH,DESCARTS_MANUALS)
    pacients_ok = [] # pacients amb >=2 fases i pv segmentades
    pacients_descartats = [] #(nom,motiu)
    n_fases_error = 0


    for sample in my_dataset:
        name = sample["name"]
        if NOMES_PACIENTS and name not in NOMES_PACIENTS:
            continue
        all_ct_paths = sample["list_ct_paths"]
        print(f"\n=== Pacient: {name} "
              f"(fases: {[ct['phase'] for ct in all_ct_paths]}) ===")

        try:

            if sample["segmentation_path"] is None:
                print(f"[DESCARTAT] {name}: sense fitxer SEG")
                pacients_descartats.append((name, "sense fitxer SEG"))
                continue

            dicomseg, mask_loaded_raw = utils.load_segmentation(sample["segmentation_path"])

            #orient_seg = utils.get_orientacio_segmentacio(dicomseg)
            #mask_loaded_raw = utils.corregir_orientacio(mask_loaded_raw, orient_seg)

        except Exception as e:
            print(f"  [DESCARTAT] {name}: no s'ha pogut carregar la SEG ({e})")
            pacients_descartats.append((name, f"SEG no carregable: {e}"))
            continue

        # Una subcarpeta per pacient manté tot ordenat
        patient_dir = SEGMENTATIONS_PATH / name
        os.makedirs(patient_dir, exist_ok=True)

        fases_fetes = set()

        for ct_info in all_ct_paths:
            fase = ct_info["phase"]
            out_path = patient_dir / f"{fase}_segmentation.npy"

            if out_path.exists():
                print(f"[salt] {fase}: Ja existeix: {out_path}")
                continue
            try:
                dcm_slices, ct = utils.load_ct(ct_info)
                mask_aligned = utils.resample_mask_to_ct(mask_loaded_raw, dicomseg, dcm_slices)
                # Diagnostic: el resampling no hauria de fer desapareixer talls amb
                # fetge. Comparem quants talls amb fetge tenia la SEG original vs
                # quants en queden despres de resamplejar a la graella del CT.
                n_seg = int(((mask_loaded_raw > 0).reshape(mask_loaded_raw.shape[0], -1).sum(1) > 0).sum())
                n_ali = int(((mask_aligned > 0).reshape(mask_aligned.shape[0], -1).sum(1) > 0).sum())
                ratio = n_ali / n_seg if n_seg > 0 else 0.0
                print(f"  [geom] {name}/{fase}: talls amb fetge {n_seg} SEG -> {n_ali} CT (ratio {ratio:.2f})")

                mask_estimation = ai.liver_segmentation(ct, mask_aligned)
                utils.save_mask_npy(mask_estimation, out_path)
                fases_fetes.add(fase)

                print(f"[ok] {fase}: desada {out_path}  (voxels fetge: {int(mask_estimation.sum())})")

            except Exception as e:
                n_fases_error += 1
                print(f"  [error] {fase}: {e}")
                # Si falla la pv, no te sentit continuar amb aquest pacient:
                # la pv es la referencia fixa del registre.
                if fase == "pv":
                    print(f"  [AVIS] {name}: la fase pv (referencia) ha fallat; "
                          f"s'abandona el pacient.")
                    break
                # Si falla una fase no-pv, simplement la saltem i seguim.
                continue

        #Un pacient només ens serveix per l'estudi si te com a mínim DUES fases segmentades
        # i una d'elles es la pv (referencia del registre).

        if "pv" not in fases_fetes:
            print(f"  [DESCARTAT] {name}: sense pv segmentada -> {sorted(fases_fetes)}")
            pacients_descartats.append((name, f"sense pv; fases={sorted(fases_fetes)}"))
        elif len(fases_fetes) < 2:
            print(f"  [DESCARTAT] {name}: nomes 1 fase segmentada -> {sorted(fases_fetes)}")
            pacients_descartats.append((name, f"<2 fases; fases={sorted(fases_fetes)}"))
        else:
            pacients_ok.append(name)
            print(f"  [PACIENT OK] {name}: fases utils -> {sorted(fases_fetes)}")

    # Resum final
    print("\n" + "=" * 60)
    print(f"Pacients utils (>=2 fases incloent pv): {len(pacients_ok)}")
    print(f"Pacients descartats: {len(pacients_descartats)}")
    print(f"Fases individuals amb error: {n_fases_error}")
    if pacients_descartats:
        print("\nDetall de descarts:")
        for nom, motiu in pacients_descartats:
            print(f"  - {nom}: {motiu}")
    print(f"\nSegmentacions desades a: {SEGMENTATIONS_PATH}")
    print(f"Pacients utils: {sorted(pacients_ok)}")