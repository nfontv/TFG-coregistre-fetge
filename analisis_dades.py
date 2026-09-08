import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import friedmanchisquare, f as f_dist, binomtest, bootstrap
from statsmodels.stats.multitest import multipletests
DADES_PATH = Path('/Users/noemifontvoorhoeve/Desktop/CoregistreRigidFetge-main/tots_els_experiments.csv')

#dades prefixades
ALPHA= 0.05
DELTA = 0.02   # llindar de rellevancia practica
SEED = 42      # reproductibilitat del bootstrap
#primer test
METRIC= "dsc_final"

#segon test
#METRIC = "n_feval"
CORRECCIO= "holm"
FASES = ["arterial", "pre-contrast"]


# Contrastos direccionals planificats: (config, columna_control, alternativa)
PLANNED_CONTRASTS = [
    ("dsc_cmaes", "dsc_identitat",  "greater"),
    ("dsc_cmaes", "dsc_centroides", "greater"),
]

METRIC_DIRECTION = {
    "dsc_final": True, "nsd_final": True,
    "assd_final": False, "hd_final": False, "hd95_final": False,
    "loss_final": False, "n_feval": False
}

def carrega_dades(DADES_PATH):
    try:
        return pd.read_csv(DADES_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(DADES_PATH, encoding="latin-1")

#Bloc per mostrar la taula
def taula_descriptiva(df, metric=METRIC, baseline_cols=("dsc_identitat", "dsc_centroides")):
    """
    Estadística descriptiva de `metric` per configuració (loss x optimizer).
    Retorna un DataFrame amb n, mitjana, desviació mostral i mediana per configuració,
    ordenat per pèrdua i, dins de cada pèrdua, de millor a pitjor.
    També imprimeix la línia base (mitjana de les inicialitzacions).
    """
    resum = (
        df.groupby(["loss_name", "optimizer"])[metric]
          .agg(n="count", mitjana="mean", desv="std", mediana="median")  # std pandas = mostral (n-1)
          .round(3)
          .reset_index()
    )
    # ordena de millor a pitjor segons la direcció de la mètrica
    higher = METRIC_DIRECTION[metric]
    resum = resum.sort_values(["loss_name", "mitjana"], ascending=[True, not higher])

    print(f"\n=== Descriptiu de {metric} per configuració ===")
    print(resum.to_string(index=False))

    # línia base: inicialitzacions (una mitjana global sobre totes les files)
    base = df[list(baseline_cols)].agg(["mean", "std", "median"]).round(3)
    print("\n=== Línia base (inicialitzacions) ===")
    print(base.to_string())

    return resum



#Bloc A: Friedman

def check_friedman_assumptions(df, fase, metric=METRIC, k_expected=12):
    """Verifica les condicions per aplicar el test Friedman."""
    sub = df[df.fase_mobil == fase]

    # No hi hagin configuracions duplicades
    dup = sub.groupby(["pacient", "config"]).size().max()
    assert dup == 1, f"{fase}: pacients amb configs repetides"

    M = sub.pivot(index="pacient", columns="config", values=metric)

    #bloc complet, k configs, cap NaN
    assert M.shape[1] == k_expected, f"{fase}: {M.shape[1]} configs != {k_expected}"
    n_incomplets = M.isna().any(axis=1).sum()
    assert n_incomplets == 0, f"{fase}: {n_incomplets} blocs incomplets"

    # informatiu: empats
    nunq = M.nunique(axis=1)
    print(f"{fase}: N={len(M)}, k={M.shape[1]}, "
          f"valors unics/bloc = {nunq.mean():.1f} (min {nunq.min()})")
    return M

def check_wilcoxon_assumptions(a, b):
    """Comprova aparellament, zeros i simetria dels d_i abans del Wilcoxon."""
    from scipy.stats import skew
    assert len(a) == len(b), "vectors no aparellats"
    d = a - b
    n_zero = int((d == 0).sum())
    print(f"  N={len(d)}, zeros={n_zero}, d>0={(d>0).sum()}, d<0={(d<0).sum()}, "
          f"skew(d)={skew(d):.2f}")
    #skew molt lluny de 0 -> plantejar test dels signes com a alternativa
    return d

def _rank_within_patient(M, higher_is_better=True):
    """Dona un rang a cada fila (pacient); rang 1 = millor. Si hi ha empats empram el rang mitjà."""
    return M.rank(axis=1, ascending=not higher_is_better, method="average")

def friedman_iman_davenport(M):
    "M:= matriu N x k, parella d'una fase concreta pels N pacients i les puntuacions dsc per cada configuració."
    N, k = M.shape
    chi2, p_chi2 = friedmanchisquare(*[M[c].values for c in M.columns])

    denom = N * (k - 1) - chi2 #correcio Iman-Davenport
    df1, df2 = k - 1, (k - 1) * (N - 1)
    if denom <= 0:  # separacio (quasi) perfecta
        F_F, p_FF = np.inf, 0.0
    else:
        F_F = (N - 1) * chi2 / denom
        p_FF = f_dist.sf(F_F, df1, df2)  # cua superior

    W = chi2 / (N * (k - 1))  # Kendall's W in [0, 1]
    return dict(N=N, k=k, chi2=chi2, p_chi2=p_chi2,
                F_F=F_F, df1=df1, df2=df2, p_FF=p_FF, W=W)

def mean_ranks(M, higher_is_better=True):
    """Rangs mitjans R_j per configuracio, ordenats de millor a pitjor."""
    ranks = _rank_within_patient(M, higher_is_better)
    return ranks.mean(axis=0).sort_values()

def nemenyi(M):
    """
    Post-hoc de Nemenyi (tots contra tots). Retorna matriu k x k de p-valors.
    Requereix scikit-posthocs.
    """
    import scikit_posthocs as sp
    nem = sp.posthoc_nemenyi_friedman(M.values)
    nem.index = M.columns
    nem.columns = M.columns
    return nem


def cd_diagram(M, title, out_path, higher_is_better=True):
    """Diagrama de diferencia critica (Demsar). Desa un PNG."""
    import scikit_posthocs as sp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nem = nemenyi(M)
    ranks = mean_ranks(M, higher_is_better)
    plt.figure(figsize=(10, 3))
    sp.critical_difference_diagram(ranks, nem)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def run_analysis(df, fase, metric=METRIC, alpha=ALPHA, cd_out=None):
    higher = METRIC_DIRECTION[metric]
    M = check_friedman_assumptions(df, fase, metric)   # només 3 arguments
    res = friedman_iman_davenport(M)
    Rj = mean_ranks(M, higher)

    print(f"\n==== {fase.upper()}  (metrica={metric}, N={res['N']}, k={res['k']}) ====")
    print(f"chi2_F = {res['chi2']:.3f} | F_F = {res['F_F']:.3f} "
          f"~ F({res['df1']},{res['df2']}) | p = {res['p_FF']:.3e} | W = {res['W']:.3f}")
    print("Rangs mitjans (menor = millor):")
    for c, v in Rj.items():
        print(f"   {c:22s} {v:.2f}")

    if res["p_FF"] < alpha and res["k"] > 2:
        if cd_out:
            cd_diagram(M, f"CD - {fase} (N={res['N']})", cd_out, higher)
            print(f"Diagrama CD desat a: {cd_out}")
    else:
        print("Friedman no significatiu (o k<=2): no es fa post-hoc.")
    return res, Rj

#Bloc B

def _parella_config_control(df, fase, config, control_col, metric=METRIC):
    """Vectors aparellats (config, control) per pacient dins d'una fase."""
    sub = df[df.fase_mobil == fase]
    A = sub[sub.config == config].set_index("pacient")[metric]
    B = sub.drop_duplicates("pacient").set_index("pacient")[control_col]
    idx = A.index.intersection(B.index)
    return A.loc[idx].values, B.loc[idx].values

def sign_test_vs_control(df, fase, config, control_col, alternative, seed=SEED, delta=DELTA):
    """Test dels signes 1-cua config vs control + mida d'efecte (mediana, IC bootstrap)."""
    a, b = _parella_config_control(df, fase, config, control_col)
    d = check_wilcoxon_assumptions(a, b)      # diagnostic: zeros, signes, skew
    d_nonzero = d[d != 0]                      # el test dels signes descarta els zeros
    n = len(d_nonzero)
    s_plus = int((d_nonzero > 0).sum())
    p = binomtest(s_plus, n, 0.5, alternative=alternative).pvalue
    ci = bootstrap((d,), np.median, confidence_level=0.95,
                   n_resamples=10000, method="percentile", random_state=np.random.default_rng(seed)).confidence_interval
    if ci.low >= delta:
        rellevancia = "rellevant"       # tot l'IC per sobre del llindar
    elif ci.high < delta:
        rellevancia = "sota llindar"    # tot l'IC per sota
    else:
        rellevancia = "indeterminat"    # l'IC creua delta

    return dict(fase=fase, contrast=f"{config} vs {control_col}", N=len(d),
                zeros=int((d == 0).sum()), s_plus=s_plus, n_efectiu=n,
                med_d=float(np.median(d)), ci_low=ci.low, ci_high=ci.high, p=p, delta=delta, rellevancia=rellevancia)

def executa_contrastos(df, contrasts=PLANNED_CONTRASTS, fases=FASES,
                       correccio=CORRECCIO, alpha=ALPHA, seed = SEED, delta=DELTA):
    """Corre tots els contrastos planificats a totes les fases i corregeix (Holm)."""
    rows = []
    for fase in fases:
        for config, control_col, alt in contrasts:
            print(f"[{fase}] {config} vs {control_col} ({alt}):")
            rows.append(sign_test_vs_control(df, fase, config, control_col, alt, seed, delta))
    res = pd.DataFrame(rows)
    res["p_adj"] = multipletests(res.p.values, alpha=alpha, method=correccio)[1]
    res["signif"] = res["p_adj"] < alpha
    return res

def taxa_no_millora(df, fase, control_col="dsc_centroides", metric=METRIC):
    """
    DESCRIPTIU (sense p-valor). Per cada configuracio, fraccio de casos que
    NO milloren el control respecte de la seva inicialitzacio:
    empitjoren (d<0) o s'estanquen (d==0). Il·lustra el mecanisme de
    (manca de) robustesa; s'ancora al ranking/CD com a evidencia confirmatoria.
    """
    sub = df[df.fase_mobil == fase]
    base = sub.drop_duplicates("pacient").set_index("pacient")[control_col]
    rows = []
    for config, g in sub.groupby("config"):
        a = g.set_index("pacient")[metric]
        idx = a.index.intersection(base.index)
        d = a.loc[idx].values - base.loc[idx].values
        N = len(d)
        rows.append(dict(
            config=config, N=N,
            pct_empitjora=100 * (d < 0).sum() / N,
            pct_estanca=100 * np.isclose(d, 0).sum() / N,
            pct_no_millora=100 * (d <= 0).sum() / N,
        ))
    return pd.DataFrame(rows).sort_values("pct_no_millora", ascending=False)

def mediana_cost(df, configs, fases=FASES, metric="n_feval"):
    for fase in fases:
        sub = df[df.fase_mobil == fase]
        print(f"\n{fase}:")
        for c in configs:
            v = sub[sub.config == c][metric].median()
            print(f"  {c}: mediana {metric} = {v:.0f}")


if __name__ == "__main__":
    df = carrega_dades(DADES_PATH)

    print("\nDESCRIPTIU: panorama general per configuració.")
    taula_descriptiva(df)

   print("\n BLOC A: qualitat entre configuracions.")
   # Anàlisi omnibus: les 12 configuracions, per fase mobil separada.
   for fase in FASES:
       run_analysis(df, fase, metric=METRIC, cd_out=f"{METRIC}_{fase}.png")

   print("\nBLOC B: validació")
   taula = executa_contrastos(df)
   print(taula.to_string(index=False))

   print("\nDESCRIPTIU: % que no millora el centroide ")
   for fase in FASES:
       print(f"\n-- {fase} --")
       print(taxa_no_millora(df, fase).to_string(index=False))

    print("\n Mediana del cost")
    mediana_cost(df, ["dsc_cmaes", "dsc_powell"])
