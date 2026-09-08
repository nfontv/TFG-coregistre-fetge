from pathlib import Path
from .utils import get_acquisition_number
import os
import re
from datetime import datetime
import pydicom
import pandas as pd
import numpy as np

"""
CARREGUEM LES NOSTRES DADES AMB LA SEGÜENT ESTRUCTURA:

    all_samples_exemple = [
        {
            'name': "HCC_012",
            'list_ct_paths': [
                {
                    "path": dataset_path / "HCC_012" / ....,
                    "acquisition_number": 1,
                    "phase": 'pre-contrast',
                    "series_uid": "1.3.6...",
                    "series_description": "PRE LIVER"
                },
                {
                    "path": dataset_path / "HCC_012" / ....,
                    "acquisition_number": 2,
                    "phase": 'arterial',
                    "series_uid": "1.3.6...",
                    "series_description": "Recon 2 LIVER 3 PHASE AP"
                },
                ...
            ],
            'segmentation_path': dataset_path / "HCC_012" / "..." / "300.000000-Segmentation-XXXXX" / "1-1.dcm",
        },
    ]
"""


def _read_series_info(path_carpeta):
    """
    Llegeix el primer .dcm de la carpeta i retorna una tupla
    (SeriesInstanceUID, SeriesDescription).
    """
    for arxiu in os.listdir(path_carpeta):
        if not arxiu.lower().endswith('.dcm'):
            continue
        try:
            dcm = pydicom.dcmread(
                os.path.join(path_carpeta, arxiu),
                force=True,
                stop_before_pixels=True,
            )
            uid = getattr(dcm, 'SeriesInstanceUID', None)
            desc = getattr(dcm, 'SeriesDescription', None)
            return (
                str(uid).strip() if uid is not None else None,
                str(desc).strip() if desc is not None else None,
            )
        except Exception:
            continue
    return None, None


def _find_segmentation_file(session_path):
    """
    Dins la carpeta de sessio, busca la carpeta que comenca per '3'
    i retorna el primer .dcm amb Modality='SEG'.
    """
    if session_path is None or not os.path.isdir(session_path):
        return None

    for carpeta in os.listdir(session_path):
        if not carpeta.startswith('3'):
            continue
        carpeta_seg = os.path.join(session_path, carpeta)
        if not os.path.isdir(carpeta_seg):
            continue
        for arxiu in os.listdir(carpeta_seg):
            path_candidat = os.path.join(carpeta_seg, arxiu)
            try:
                dcm_temp = pydicom.dcmread(
                    path_candidat, force=True, stop_before_pixels=True
                )
                if getattr(dcm_temp, 'Modality', '') == 'SEG':
                    return Path(path_candidat)
            except Exception:
                continue
    return None

def _norm_desc(text):
    """
    Normalitza una SeriesDescription per comparar disc vs taula:
    minuscules, sense puntuacio, espais col·lapsats, tokens ordenats.
    Aixi 'LIVER 3 PHASE (AP)' i 'LIVER 3 PHASE AP' coincideixen, i
    'Recon 2 LIVER 3 PHASE AP' i 'Recon 2: 3 PHASE LIVER (ABD)'... NO.
    """
    if text is None:
        return ''
    t = str(text).lower()
    t = re.sub(r'[^a-z0-9 ]', ' ', t)      # fora parentesis, dos punts, etc.
    t = re.sub(r'\s+', ' ', t).strip()
    return ' '.join(sorted(t.split()))     # ordre de tokens irrellevant

def _count_slices(path_carpeta):
    """Compta quants fitxers .dcm hi ha a la carpeta (= nombre de talls
    de la serie). Serveix per desempatar series amb la mateixa
    SeriesDescription pero fases diferents (p. ex. HCC_054, dues 'C-A-P')."""
    return sum(
        1 for arxiu in os.listdir(path_carpeta)
        if arxiu.lower().endswith('.dcm')
    )

def load_dataset(dataset_path, taula_suplementaria_path,descarts_manuals):
    '''
    Atributs:
    dataset_path: Camí fins la carpeta que conté les carpetes de cada pacient {HCC_002,...,HCC_010}
    taula_suplementaria_path: cami a la taula que ens proporciona la informació de les fases
    descarts_manuals= diccionari de descarts manuals
    '''

    df = pd.read_excel(taula_suplementaria_path, header=3)
    df.columns = ['name', 'study_date', 'series_uid',
                  'series_description', 'phase', 'n_images']
    df['name'] = df['name'].astype(str).str.strip()
    df['phase'] = df['phase'].astype(str).str.strip().str.lower()  # <-- AFEGEIX

    def is_uid(text):
        if not isinstance(text, str):
            return False
        return bool(re.match(r'^[\d.]+$', text)) and '.' in text

    df['series_uid'] = df['series_uid'].astype(object)
    # Si l'UID s'ha colat dins 'series_description', el movem a 'series_uid'
    mask = df['series_description'].apply(is_uid)
    df.loc[mask, 'series_uid'] = df.loc[mask, 'series_description']
    df.loc[mask, 'series_description'] = np.nan


    # Filtrem per data: ens quedem nomes amb el timepoint pre-tractament
    df['study_date'] = pd.to_datetime(df['study_date'])
    df_pre_tt = df[df.groupby('name')['study_date'].transform('min')
                   == df['study_date']]
    # Filtrem per fases d'interes

    fases_separades = ['pre-contrast', 'arterial', 'pv']
    # eliminam posibles espais en blanc i passem tot a minuscules
    df_pre_tt = df_pre_tt[df_pre_tt['phase'].isin(fases_separades)]
    pacients = {}

    for name, group in df_pre_tt.groupby('name'):
        pacients[name] = {
            'name': name,
            'serie': group.drop(columns=['name']).to_dict('records'),
        }


    # Afegim els camins de directori per cada fase
    all_samples = []

    for pid, dades in pacients.items():
        path_pacient = os.path.join(dataset_path, pid)
        if not os.path.isdir(path_pacient):
            #print(f"[AVIS] No s'ha trobat la carpeta del pacient {pid}")
            continue

        # Llistem totes les carpetes de sessio
        sessions = [s for s in os.listdir(path_pacient)
                    if os.path.isdir(os.path.join(path_pacient, s))]
        if not sessions:
            #print(f"[AVIS] {pid}: no te cap carpeta de sessio")
            continue

        # Triam la sessio amb la data mes antiga (pre-tractament),
        # parsejant els 10 primers caracters del nom 'MM-DD-YYYY-...'
        try:
            sessio_antiga = min(
                sessions,
                key=lambda nom: datetime.strptime(nom[:10], "%m-%d-%Y"),
            )
        except Exception as e:
            #print(f"[AVIS] {pid}: no s'ha pogut triar la data minima ({e})")
            continue

        path_sessio = os.path.join(path_pacient, sessio_antiga)

        # Construim dos indexs a partir de la taula per al pacient actual:
        # uid_to_info : match per SeriesInstanceUID
        # desc_to_info : SeriesDescription (quan no te SeriesInstanceUID)
        uid_to_info = {}
        desc_to_info = {}
        desc_n_to_info = {}  # clau composta (desc_norm, n_images) per desempatar
        for serie in dades['serie']:
            uid = serie.get('series_uid')
            desc = serie.get('series_description')
            n_img = serie.get('n_images')
            if uid is not None and not (isinstance(uid, float) and pd.isna(uid)):
                uid_to_info[str(uid).strip()] = serie
            if desc is not None and not (isinstance(desc, float) and pd.isna(desc)):
                desc_to_info[_norm_desc(desc)] = serie
                if n_img is not None and not (isinstance(n_img, float) and pd.isna(n_img)):
                    desc_n_to_info[(_norm_desc(desc), int(n_img))] = serie

        # Recorrem les carpetes de la sessió (excloent la de segmentació '300...')
        list_ct_paths = []
        for carpeta in os.listdir(path_sessio):
            if carpeta.startswith('300'):
                continue
            path_carpeta = os.path.join(path_sessio, carpeta)
            if not os.path.isdir(path_carpeta):
                continue

            # Llegim UID, descripcio i nombre de talls del primer dcm
            uid_carpeta, desc_carpeta = _read_series_info(path_carpeta)
            n_carpeta = _count_slices(path_carpeta)

            # Creuem amb la taula: 1) per UID, 2) per descripcio + talls, 3) per descripcio sola

            serie_info = None
            # 1 per UID (el mes fiable, quan la taula el te)
            if uid_carpeta is not None:
                serie_info = uid_to_info.get(uid_carpeta)
            # 2 per descripcio + nombre de talls (desempata C-A-P d'HCC_054)
            if serie_info is None and desc_carpeta is not None:
                serie_info = desc_n_to_info.get((_norm_desc(desc_carpeta), n_carpeta))
            # 3 per descripcio sola (ultim recurs; pot ser ambigu)
            if serie_info is None and desc_carpeta is not None:
                serie_info = desc_to_info.get(_norm_desc(desc_carpeta))

            if serie_info is None:
                # No identificada com a fase d'interes, la saltem
                continue

            list_ct_paths.append({
                'path': Path(path_carpeta),
                'acquisition_number': get_acquisition_number(Path(path_carpeta)),
                'phase': serie_info.get('phase'),
                'series_uid': uid_carpeta or serie_info.get('series_uid'),
                'series_description': desc_carpeta or serie_info.get('series_description'),
            })

        if not list_ct_paths:
            print(f"[AVIS] {pid}: no s'ha pogut identificar cap fase a "
                  f"{sessio_antiga}")

        # Detectem col·lisions d'etiqueta: si dues carpetes han resolt a la
        # MATEIXA fase, l'assignacio es ambigua (p. ex. HCC_054 amb dues 'pv',
        # o arterial/pv amb descripcio i talls identics). Com que volem que
        # cada fase sigui inequivoca, descartem aquestes fases conflictives.
        fases_vistes = [ct['phase'] for ct in list_ct_paths]
        fases_duplicades = {f for f in fases_vistes if fases_vistes.count(f) > 1}
        if fases_duplicades:
            print(f"[AVIS] {pid}: fases ambigues (duplicades) {fases_duplicades} "
                  f"a {sessio_antiga}. Es descarten aquestes fases.")
            list_ct_paths = [ct for ct in list_ct_paths
                             if ct['phase'] not in fases_duplicades]

        # Cerquem el fitxer de segmentacio dins la mateixa sessio
        path_seg = _find_segmentation_file(path_sessio)

        all_samples.append({
            'name': pid,
            'list_ct_paths': list_ct_paths,
            'segmentation_path': path_seg,
        })

    # 3) Filtre final (FORA del bucle): ens quedem nomes amb els pacients
    #    que tenen com a minim dues fases DIFERENTS i ben etiquetades, i
    #    que entre elles inclouen la pv (referencia fixa del registre).

    samples_validats = []
    for sample in all_samples:
        if sample['name'] in descarts_manuals:
            #print(f"[DESCARTAT MANUAL] {sample['name']}: pv inutilitzable "
                  #f"(inconsistencia de dades a la font)")
            continue
        fases = {ct['phase'] for ct in sample['list_ct_paths']}
        if len(fases) < 2:
            #print(f"[DESCARTAT] {sample['name']}: menys de 2 fases uniques -> {fases}")
            continue
        if 'pv' not in fases:
            #print(f"[DESCARTAT] {sample['name']}: no te fase pv (referencia) -> {fases}")
            continue
        samples_validats.append(sample)

    return samples_validats