import matplotlib.pyplot as plt
from analisis_dades import carrega_dades, _parella_config_control, DADES_PATH

df = carrega_dades(DADES_PATH)

for fase in ["arterial", "pre-contrast"]:
    for control in ["dsc_identitat", "dsc_centroides"]:
        a, b = _parella_config_control(df, fase, "dsc_cmaes", control)
        d = a - b
        plt.figure()
        plt.hist(d, bins=15)
        plt.axvline(0, color="k", ls="--")
        plt.title(f"{fase}: dsc_cmaes - {control}")
        plt.xlabel("d_i")
        plt.savefig(f"hist_{fase}_{control}.png", dpi=150)
        plt.close()
        print(f"desat hist_{fase}_{control}.png")