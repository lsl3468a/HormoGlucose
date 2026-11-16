"""
Analyse des résultats de eval_scenarios.py.

Entrée  : results_eval/eval_*.csv et traces_*.csv (fichiers _quick ignorés)
Sorties : dossier analyse/
  - patient_means.csv      moyennes sur les graines, par patient
  - deltas.csv             écarts par patient (modèle - référence), par contrôleur
  - summary_deltas.csv     médiane [Q1, Q3] des écarts + test de Wilcoxon apparié
                           (tous patients et par groupe d'âge)
  - controller_compare.csv dégradation PID vs dégradation BBC (Wilcoxon apparié)
  - absolute.csv           valeurs absolues médianes par scénario/modèle/contrôleur
  - table_main.tex         effet de chaque perturbation (Holm), + table_main_values.csv
  - table_ctrl.tex         robustesse du PID par rapport au BBC (Holm), + table_ctrl_values.csv
  - fig_S1.pdf, fig_S2.pdf, fig_S3.pdf   profils glycémiques du jour 2

Usage : python analyse_eval.py --indir results_eval --outdir analyse
        python analyse_eval.py --deltas analyse/deltas.csv --outdir analyse   (tableaux seuls)
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ["tir", "tbr70", "tbr54", "tar180", "mean_bg", "cv", "lbgi", "hbgi", "min_bg"]
KEYS = ["patient", "groupe", "scenario", "model", "controller", "window"]

# (scénario, modèle testé, modèle de référence, fenêtres principales)
COMPARISONS = [
    ("S1_aube", "PA", "base", ["jour2"]),                       # équivalence
    ("S1_aube", "H0", "base", ["jour2"]),                       # équivalence
    ("S1_aube", "H", "base", ["nuit_00_06", "aube_03_08", "matin_06_12",
                              "apm_12_18", "soir_18_24", "jour2"]),
    ("S2_hypo", "H", "base", ["post_bolus_15_1930", "jour2"]),
    ("S3_exercice", "PA", "base", ["post_ex_16_24", "jour2"]),
    ("S3_exercice", "H", "base", ["post_ex_16_24", "jour2"]),
    ("S3_exercice", "H", "PA", ["post_ex_16_24", "jour2"]),
]
# lignes du tableau principal de l'article
MAIN_ROWS = [
    ("S1_aube", "H", "base", "matin_06_12", "Dawn, 06:00--12:00"),
    ("S1_aube", "H", "base", "jour2", "Dawn, 24 h"),
    ("S2_hypo", "H", "base", "post_bolus_15_1930", "Hypoglycemia, 15:00--19:30"),
    ("S3_exercice", "PA", "base", "post_ex_16_24", "Exercise only, 16:00--24:00"),
    ("S3_exercice", "H", "base", "post_ex_16_24", "Exercise + hormones, 16:00--24:00"),
    ("S3_exercice", "H", "PA", "post_ex_16_24", "Hormones under exercise, 16:00--24:00"),
]
MAIN_METRICS = [("tir", r"$\Delta$TIR"), ("tbr70", r"$\Delta$TBR$_{<70}$"),
                ("tbr54", r"$\Delta$TBR$_{<54}$"), ("mean_bg", r"$\Delta$mean BG")]


def wilcoxon_p(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 5 or np.allclose(x, 0):
        return np.nan
    try:
        return float(wilcoxon(x).pvalue)
    except ValueError:
        return np.nan


def holm(pvals):
    """Correction de Holm-Bonferroni ; les NaN sont ignorés."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]
    order = idx[np.argsort(p[idx])]
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


ZERO_INFLATED = {"tbr70", "tbr54", "tar250"}   # beaucoup de zéros -> moyenne (ET)


def summarize(x):
    x = pd.Series(x).dropna()
    return pd.Series({"n": len(x), "median": x.median(), "q1": x.quantile(.25),
                      "q3": x.quantile(.75), "mean": x.mean(), "sd": x.std(),
                      "p_wilcoxon": wilcoxon_p(x)})


def load(indir, prefix):
    files = [f for f in glob.glob(os.path.join(indir, f"{prefix}_*.csv"))
             if not f.endswith("_quick.csv")]
    if not files:
        raise SystemExit(f"Aucun fichier {prefix}_*.csv dans {indir}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def write_tables(d, outdir):
    """Tableaux LaTeX : effet des perturbations (table_main.tex) et
    comparaison des contrôleurs (table_ctrl.tex), p-valeurs corrigées par Holm
    sur l'ensemble des cellules de chaque tableau."""
    # ---- effet de chaque perturbation, par contrôleur
    cells = []
    for scen, test, ref, w, label in MAIN_ROWS:
        comp = f"{scen}:{test}-{ref}"
        for ctrl in ["BBC", "PID"]:
            g = d[(d.comparison == comp) & (d.window == w) & (d.controller == ctrl)]
            for met, _ in MAIN_METRICS:
                x = g[f"d_{met}"].dropna()
                cells.append({"label": label, "comp": comp, "window": w, "ctrl": ctrl,
                              "metric": met, "n": len(x), "median": x.median(),
                              "q1": x.quantile(.25), "q3": x.quantile(.75),
                              "mean": x.mean(), "sd": x.std(), "p": wilcoxon_p(x)})
    c = pd.DataFrame(cells)
    c["p_holm"] = holm(c.p)
    c.to_csv(os.path.join(outdir, "table_main_values.csv"), index=False)

    def fmt(r, met):
        star = "*" if r.p_holm < 0.05 else ""
        if met in ZERO_INFLATED:
            return f"{r['mean']:+.1f}{star} ({r.sd:.1f})"
        return f"{r['median']:+.1f}{star} [{r.q1:+.1f}, {r.q3:+.1f}]"

    lines = [r"\begin{table}",
             r"\caption{Change in glycemic outcomes caused by each disturbance, per"
             r" controller (paired difference test model $-$ reference model over the 30"
             r" virtual patients). $\Delta$TIR and $\Delta$mean BG: median [interquartile"
             r" range]; $\Delta$TBR: mean (SD), as most patients have no time below range."
             r" * $p<0.05$, Wilcoxon signed-rank test with Holm correction over all cells"
             r" of the table.}\label{tab:main}",
             r"\centering\scriptsize\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}ll" + "c" * len(MAIN_METRICS) + r"@{}}", r"\toprule",
             "Scenario & Ctrl & " + " & ".join(h for _, h in MAIN_METRICS) + r"\\",
             r"\midrule"]
    for scen, test, ref, w, label in MAIN_ROWS:
        for i, ctrl in enumerate(["BBC", "PID"]):
            sub = c[(c.label == label) & (c.ctrl == ctrl)].set_index("metric")
            lines.append(f"{label if i == 0 else ''} & {ctrl} & "
                         + " & ".join(fmt(sub.loc[m], m) for m, _ in MAIN_METRICS) + r"\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(outdir, "table_main.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ---- PID plus ou moins robuste que BBC : difference des écarts par patient
    cc = []
    for scen, test, ref, w, label in MAIN_ROWS:
        comp = f"{scen}:{test}-{ref}"
        g = d[(d.comparison == comp) & (d.window == w)]
        for met, _ in MAIN_METRICS:
            p = g.pivot_table(index="patient", columns="controller", values=f"d_{met}")
            if not {"PID", "BBC"} <= set(p.columns):
                continue
            x = (p["PID"] - p["BBC"]).dropna()
            cc.append({"label": label, "metric": met, "median": x.median(),
                       "q1": x.quantile(.25), "q3": x.quantile(.75), "mean": x.mean(),
                       "sd": x.std(), "p": wilcoxon_p(x)})
    cc = pd.DataFrame(cc)
    cc["p_holm"] = holm(cc.p)
    cc.to_csv(os.path.join(outdir, "table_ctrl_values.csv"), index=False)
    lines = [r"\begin{table}",
             r"\caption{Robustness of PID relative to BBC: paired difference, over the 30"
             r" patients, between the change observed with PID and the change observed"
             r" with BBC ($\Delta_{PID}-\Delta_{BBC}$). Negative values for TBR and mean BG"
             r" and positive values for TIR favour PID. Same statistics and"
             r" Holm-corrected Wilcoxon test as Table~\ref{tab:main}.}\label{tab:ctrl}",
             r"\centering\scriptsize\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}l" + "c" * len(MAIN_METRICS) + r"@{}}", r"\toprule",
             "Scenario & " + " & ".join(h for _, h in MAIN_METRICS) + r"\\", r"\midrule"]
    for _, _, _, _, label in MAIN_ROWS:
        sub = cc[cc.label == label].set_index("metric")
        lines.append(f"{label} & " + " & ".join(fmt(sub.loc[m], m) for m, _ in MAIN_METRICS) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(outdir, "table_ctrl.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results_eval")
    ap.add_argument("--outdir", default="analyse")
    ap.add_argument("--deltas", help="refaire seulement les tableaux à partir d'un deltas.csv")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if args.deltas:
        write_tables(pd.read_csv(args.deltas), args.outdir)
        print(f"Tableaux écrits dans {args.outdir}/")
        return

    raw = load(args.indir, "eval")
    raw["ok"] = raw["ok"].astype(str).str.lower().isin(["true", "1"])
    bad = raw[~raw.ok]
    print(f"{raw.patient.nunique()} patients ; "
          f"{bad[['patient', 'scenario', 'model', 'controller', 'seed']].drop_duplicates().shape[0]}"
          f" simulation(s) invalide(s) ou tronquée(s)")
    if len(bad):
        print(bad[["patient", "scenario", "model", "controller", "seed", "error"]]
              .drop_duplicates().head(10).to_string())
        # une simulation tronquée (ex. BG < 10 mg/dL) est un ÉCHEC du contrôleur :
        # on la garde, ses métriques portent sur la partie simulée ; on la compte.
    m = (raw[raw.window != ""].groupby(KEYS)[METRICS].mean().reset_index())
    m.to_csv(os.path.join(args.outdir, "patient_means.csv"), index=False)

    # --- écarts par patient -------------------------------------------------
    rows = []
    for scen, test, ref, windows in COMPARISONS:
        for w in windows:
            a = m[(m.scenario == scen) & (m.model == test) & (m.window == w)]
            b = m[(m.scenario == scen) & (m.model == ref) & (m.window == w)]
            j = a.merge(b, on=["patient", "groupe", "controller"], suffixes=("", "_ref"))
            for met in METRICS:
                j[f"d_{met}"] = j[met] - j[f"{met}_ref"]
            j["comparison"] = f"{scen}:{test}-{ref}"
            j["window"] = w
            rows.append(j[["comparison", "window", "patient", "groupe", "controller"]
                          + [f"d_{x}" for x in METRICS]])
    d = pd.concat(rows, ignore_index=True)
    d.to_csv(os.path.join(args.outdir, "deltas.csv"), index=False)

    # --- résumés : tous patients + par groupe d'âge ---------------------------
    summ = []
    for grp_name, sub in [("all", d)] + list(d.groupby("groupe")):
        for (comp, w, ctrl), g in sub.groupby(["comparison", "window", "controller"]):
            for met in METRICS:
                s = summarize(g[f"d_{met}"])
                summ.append({"groupe": grp_name, "comparison": comp, "window": w,
                             "controller": ctrl, "metric": met, **s.to_dict()})
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(args.outdir, "summary_deltas.csv"), index=False)

    # --- PID plus ou moins robuste que BBC ? ----------------------------------
    cc = []
    for (comp, w), g in d.groupby(["comparison", "window"]):
        p = g.pivot_table(index="patient", columns="controller",
                          values=[f"d_{x}" for x in METRICS])
        for met in METRICS:
            col = f"d_{met}"
            if (col, "PID") in p and (col, "BBC") in p:
                diff = p[(col, "PID")] - p[(col, "BBC")]
                cc.append({"comparison": comp, "window": w, "metric": met,
                           "median_dPID_minus_dBBC": diff.median(),
                           "p_wilcoxon": wilcoxon_p(diff)})
    pd.DataFrame(cc).to_csv(os.path.join(args.outdir, "controller_compare.csv"), index=False)

    # --- valeurs absolues -----------------------------------------------------
    absol = (m.groupby(["scenario", "model", "controller", "window"])[METRICS]
             .median().reset_index())
    absol.to_csv(os.path.join(args.outdir, "absolute.csv"), index=False)

    write_tables(d, args.outdir)

    # --- figures : profils du jour 2 --------------------------------------------
    try:
        tr = load(args.indir, "traces")
    except SystemExit:
        tr = None
    if tr is not None:
        tr = tr[(tr.hour >= 24) & (tr.hour <= 48)].copy()
        tr["h"] = (tr.hour - 24).round(2)
        colors = {"base": "#4b5563", "PA": "#2563eb", "H0": "#9ca3af", "H": "#dc2626"}
        events = {"S2_hypo": [(15.0, 15.05, "extra bolus")],
                  "S3_exercice": [(16.0, 16.75, "exercise")]}
        for scen, models in [("S1_aube", ["base", "H"]), ("S2_hypo", ["base", "H"]),
                             ("S3_exercice", ["base", "PA", "H"])]:
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
            for ax, ctrl in zip(axes, ["BBC", "PID"]):
                for mod in models:
                    g = tr[(tr.scenario == scen) & (tr.model == mod) & (tr.controller == ctrl)]
                    if g.empty:
                        continue
                    q = g.groupby("h").BG.quantile([.25, .5, .75]).unstack()
                    ax.plot(q.index, q[0.5], color=colors[mod], lw=1.6, label=mod)
                    ax.fill_between(q.index, q[0.25], q[0.75], color=colors[mod], alpha=.15, lw=0)
                ax.axhspan(70, 180, color="#16a34a", alpha=.06, lw=0)
                ax.axhline(70, color="#16a34a", lw=.6, ls="--")
                ax.axhline(54, color="#b91c1c", lw=.6, ls=":")
                for a0, a1, lab in events.get(scen, []):
                    ax.axvspan(a0, a1 if a1 - a0 > .1 else a0 + .1, color="#f59e0b", alpha=.3, lw=0)
                for meal in (8.0, 12.5, 19.5):
                    ax.axvline(meal, color="#9ca3af", lw=.5, ls=":")
                ax.set_title(ctrl, fontsize=10)
                ax.set_xlim(0, 24)
                ax.set_xticks(range(0, 25, 3))
                ax.set_xlabel("Time of day (h)")
                ax.grid(alpha=.25)
            axes[0].set_ylabel("Glucose (mg/dL), median and IQR")
            axes[1].legend(fontsize=8, frameon=False)
            fig.tight_layout()
            fig.savefig(os.path.join(args.outdir, f"fig_{scen.split('_')[0]}.pdf"))
            fig.savefig(os.path.join(args.outdir, f"fig_{scen.split('_')[0]}.png"), dpi=200)
            plt.close(fig)

    # --- aperçu console -----------------------------------------------------------
    view = summ[(summ.groupe == "all") & summ.metric.isin(["tir", "tbr70", "tbr54"])]
    print(view.pivot_table(index=["comparison", "window", "controller"], columns="metric",
                           values="median").round(2).to_string())
    print(f"\nSorties dans {args.outdir}/")


if __name__ == "__main__":
    main()
