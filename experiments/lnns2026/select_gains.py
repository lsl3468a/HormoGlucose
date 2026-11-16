"""
Choix des gains du PID par patient à partir des CSV produits par grid_pid.py.

Règle (décidée pour l'article) :
  1. contraintes du consensus (Battelino et al., 2019) : TBR<54 < 1 % ET TBR<70 < 4 %
  2. parmi les combinaisons admissibles : TIR maximal
  3. en cas d'égalité : indice de risque (LBGI + HBGI) minimal, qui pénalise
     à la fois les hypo- et les hyperglycémies (Kovatchev 2017)
  Repli si aucune combinaison n'est admissible (priorité à la sécurité) :
     TBR<54 minimal, puis TIR maximal, puis indice de risque minimal.

Agrégation des journées de réglage :
  --agg pooled  : métriques moyennées sur les journées (= journées regroupées,
                  car toutes ont le même nombre de mesures)
  --agg per-day : plus strict, les contraintes TBR doivent être respectées
                  CHAQUE jour (on prend le maximum journalier des TBR)

Usage :
    python select_gains.py --indir results_grid --agg pooled
"""
import argparse
import glob
import os

import pandas as pd

TBR54_MAX, TBR70_MAX = 1.0, 4.0
METRICS = ["tir", "tbr70", "tbr54", "tar180", "tar250",
           "lbgi", "hbgi", "mean_bg", "cv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results_grid")
    ap.add_argument("--agg", choices=["pooled", "per-day"], default="pooled")
    ap.add_argument("--out", default="gains_pid.csv")
    args = ap.parse_args()

    files = [f for f in glob.glob(os.path.join(args.indir, "grid_*.csv"))
             if not f.endswith("_quick.csv")]
    if not files:
        raise SystemExit(f"Aucun fichier grid_*.csv dans {args.indir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # une combinaison n'est retenue que si TOUTES ses journées sont valides
    keys = ["patient", "P", "I", "D"]
    valid = df.groupby(keys)["ok"].transform("all")
    n_excl = df.loc[~valid, keys].drop_duplicates().shape[0]
    if n_excl:
        print(f"Attention : {n_excl} combinaison(s) exclue(s) (simulation invalide).")
    df = df[valid]

    n_days = df.groupby(keys)["seed"].nunique()
    assert n_days.nunique() == 1, "nombre de journées différent selon les combinaisons"

    agg = {m: "mean" for m in METRICS}
    if args.agg == "per-day":
        agg.update({"tbr54": "max", "tbr70": "max"})
    pooled = df.groupby(keys).agg(agg).reset_index()
    pooled["risk_index"] = pooled["lbgi"] + pooled["hbgi"]
    # évite que des écarts d'arrondi flottant cassent les égalités
    pooled[METRICS + ["risk_index"]] = pooled[METRICS + ["risk_index"]].round(6)

    rows = []
    for patient, g in pooled.groupby("patient"):
        adm = g[(g.tbr54 < TBR54_MAX) & (g.tbr70 < TBR70_MAX)]
        if len(adm):
            best = adm.sort_values(["tir", "risk_index"], ascending=[False, True]).iloc[0]
            rule = "contraintes"
        else:
            best = g.sort_values(["tbr54", "tir", "risk_index"],
                                 ascending=[True, False, True]).iloc[0]
            rule = "repli"
        rows.append({**best.to_dict(), "regle": rule, "n_admissibles": len(adm)})

    res = pd.DataFrame(rows).sort_values("patient")
    res.to_csv(args.out, index=False)
    n_repli = int((res.regle == "repli").sum())
    print(res[["patient", "P", "I", "D", "tir", "tbr70", "tbr54", "lbgi", "hbgi",
               "risk_index", "regle", "n_admissibles"]].to_string(index=False))
    print(f"\n{len(res)} patients, {n_repli} en repli. "
          f"Agrégation : {args.agg}. -> {args.out}")


if __name__ == "__main__":
    main()