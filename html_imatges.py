"""
Visualització en HTML de els talls generats per visualize guardats en una carpeta
Genera:
  - Un fitxer HTML amb galeria de miniatures de totes les imatges
  - Un resum en consola amb estadistiques (quants OK, quants fallats, mascaras buides)
"""

import os
from pathlib import Path
from datetime import datetime
import base64


BASE_PATH = Path(__file__).parent

# Carpeta de visualitzacions que es vol revisar. Canvia-la per la que toqui
# (les genera visualitzacio_carrega_dades.py, visualitzar_segmentacions_guardades.py, etc.).
RESULTS_PATH = BASE_PATH / "comprovacio_dataset_visualitzacions"
OUTPUT_HTML = RESULTS_PATH / "galeria_basedades_inicial.html"




def image_to_base64(image_path):
    """
    Converteix una imatge a base64 per poder-la incrustar dins l'HTML.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analitza_imatge(image_path):
    """
    Fa una analisi basica de la imatge per detectar problemes evidents.
    Retorna un diccionari amb metainfo.

    Necessita Pillow (pip install Pillow) per llegir metadades de la imatge.
    Si no esta instal·lat, retorna info basica del fitxer.
    """
    info = {
        "path": image_path,
        "name": image_path.name,
        "mida_kb": round(image_path.stat().st_size / 1024, 1),
        "warning": None
    }

    # Una imatge molt petita (<50 KB) pot indicar que esta quasi buida o corrupta
    if info["mida_kb"] < 30:
        info["warning"] = f"Imatge molt petita ({info['mida_kb']} KB) - possible mascara buida o error"

    return info

def parse_nom(filename):
    """
    Visualize_HCC_006_ct0.jpg -> ('HCC_006', 'CT0')
    Fallback per si el nom no segueix el patro esperat.
    """
    stem = filename.replace("Visualize_", "").replace(".jpg", "")
    # Cerquem l'ultim "_ct" per separar pacient del index
    idx = stem.rfind("_ct")
    if idx == -1:
        return stem, ""
    pacient = stem[:idx]
    ct_label = "CT" + stem[idx + 3:]
    return pacient, ct_label

def genera_html_galeria(imatges_info):
    cards_html = ""
    n_warnings = 0

    for info in imatges_info:
        if info["warning"]:
            card_color = "#fff3cd"
            warning_html = (
                f'<p style="color:#856404; font-size:11px; margin:4px 0;">'
                f'{info["warning"]}</p>'
            )
            n_warnings += 1
        else:
            card_color = "#ffffff"
            warning_html = ""

        try:
            img_b64 = image_to_base64(info["path"])
            img_html = (
                f'<img src="data:image/jpeg;base64,{img_b64}" '
                f'style="width:100%; border-radius:4px;">'
            )
        except Exception as e:
            img_html = f'<p style="color:red;">Error: {e}</p>'

        pacient, ct_label = parse_nom(info["name"])
        titol = f"{pacient} - {ct_label}" if ct_label else pacient

        cards_html += f"""
        <div style="
            background:{card_color};
            border:1px solid #ddd;
            border-radius:8px;
            padding:8px;
            break-inside: avoid;
        ">
            {img_html}
            <p style="font-weight:bold; margin:6px 0 2px 0; font-size:13px;">{titol}</p>
            <p style="color:#666; font-size:11px; margin:2px 0;">{info['mida_kb']} KB</p>
            {warning_html}
        </div>
        """

    n_total = len(imatges_info)
    n_ok = n_total - n_warnings
    resum_color = "#28a745" if n_warnings == 0 else "#856404"
    resum_text = (
        f"{n_ok}/{n_total} imatges sense warnings | "
        f"{n_warnings} possibles problemes"
    )

    html = f"""<!DOCTYPE html>
<html lang="ca">
<head>
    <meta charset="UTF-8">
    <title>Galeria de segmentacions - TFG</title>
    <style>
        body {{ font-family: Arial, sans-serif; background: #f5f5f5; padding: 20px; }}
        h1 {{ color: #333; }}
        .resum {{
            background: white; border-radius: 8px; padding: 12px 20px;
            margin-bottom: 20px; border-left: 5px solid {resum_color};
            font-size: 14px; color: {resum_color};
        }}
        .grid {{ column-count: 4; column-gap: 12px; }}
        .grid > div {{ margin-bottom: 12px; }}
        @media (max-width: 1200px) {{ .grid {{ column-count: 3; }} }}
        @media (max-width: 800px)  {{ .grid {{ column-count: 2; }} }}
    </style>
</head>
<body>
    <h1>Galeria de segmentacions (cache MedSAM2)</h1>
    <div class="resum">{resum_text}</div>
    <div class="grid">
        {cards_html}
    </div>
</body>
</html>
"""
    return html, n_warnings



def main():
    print(f"Cercant imatges a: {RESULTS_PATH}")

    if not RESULTS_PATH.exists():
        print(f"ERROR: El directori no existeix: {RESULTS_PATH}")
        return

    # Cerquem tots els fitxers JPG de visualitzacio en el directori de resultats
    imatges = sorted(RESULTS_PATH.glob("Visualize_*.jpg"))

    if not imatges:
        print("No s'han trobat imatges Visualize_*.jpg en el directori especificat.")
        return

    print(f"S'han trobat {len(imatges)} imatges. Analitzant...")

    # Analitzem cada imatge
    imatges_info = [analitza_imatge(p) for p in imatges]

    # Mostrem resum a la consola
    print("\n--- RESUM ---")
    warnings_list = [i for i in imatges_info if i["warning"]]
    print(f"Total imatges: {len(imatges_info)}")
    print(f"Sense problemes detectats: {len(imatges_info) - len(warnings_list)}")
    print(f"Amb possibles problemes: {len(warnings_list)}")
    if warnings_list:
        print("\nImatges amb warnings:")
        for w in warnings_list:
            print(f"  - {w['name']}: {w['warning']}")

    # Generem l'HTML
    print("\nGenerant galeria HTML...")
    html_content, n_warnings = genera_html_galeria(imatges_info)

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Galeria guardada a: {OUTPUT_HTML}")
    print("Obre aquest fitxer al navegador per revisar totes les imatges.")


if __name__ == "__main__":
    main()