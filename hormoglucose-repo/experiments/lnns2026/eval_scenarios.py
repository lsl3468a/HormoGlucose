"""
Évaluation des contrôleurs (BBC, PID réglé) sur les scénarios de test.

Protocole : les gains du PID viennent de gains_pid_v2.csv (réglés sur le
simulateur DE BASE). Ils ne sont jamais modifiés ici.

48 h simulées, analyse sur le JOUR 2 (heures 24-48) pour éviter le
transitoire initial des états hormonaux. Repas fixes, identiques pour
tous ; les graines ne font varier que le bruit du capteur.

Scénarios (heures depuis le début, jour 2 = 24-48) :
  S1 "aube"     : pas d'événement ; modèles base, PA, H0, H ; analyse par tranches
                  de 6 h + aube (3-8 h) + journée entière (effet retardé du cortisol)
                  (PA sans exercice et H0 = H avec xi_C = xi_GH = 0
                   servent de TEST D'ÉQUIVALENCE avec la base)
  S2 "hypo"     : bolus d'erreur (EXTRA_BOLUS_CHO g / CR) à 15:00 du jour 2,
                  sans repas ; modèles base, H
  S3 "exercice" : exercice 16:00-16:45 du jour 2 à HR = EX_HR_FACTOR * HR_b ;
                  modèles base, PA, H

Un appel = un patient (pour un tableau de jobs Slurm).
Test rapide :   python eval_scenarios.py --patient-index 0 --quick --n-jobs 2
Complet :       python eval_scenarios.py --patient-index 0 --n-jobs 8
"""
import argparse
import os
import time
import traceback
from datetime import datetime, timedelta
from multiprocessing import Pool

import numpy as np
import pandas as pd

from grid_pid import glycemic_metrics, patient_names, SENSOR, PUMP
from hormoglucose.simulation.env import T1DSimEnv
from hormoglucose.patient.t1dpatient import T1DPatient
from hormoglucose.patient.t1dpatientPA import T1DPatientPA
from hormoglucose.patient.t1dpatientHypophyse import T1DPatientHypophyse
from hormoglucose.sensor.cgm import CGMSensor
from hormoglucose.actuator.pump import InsulinPump
from hormoglucose.simulation.scenario_multiday import MultiDayCustomScenario  # repas, jour par jour
from hormoglucose.simulation.scenario_pa import CustomScenarioPA
from hormoglucose.patient.t1dpatientPA import Action as PAAction
from hormoglucose.simulation.sim_engine import SimObj
from hormoglucose.controller.base import Action
from hormoglucose.controller.pid_ctrller import PIDController, CONTROL_QUEST
from hormoglucose.controller.basal_bolus_ctrller import BBController

# ---------------------------------------------------------------------------
# Protocole
# ---------------------------------------------------------------------------
START = datetime(2026, 1, 1, 0, 0)
SIM_HOURS = 48
DAY2 = (24.0, 48.0)
TEST_SEEDS = [1001, 1002, 1003, 1004, 1005]   # bruit capteur ; différentes du réglage

MEALS_DAY = [(8.0, 45), (12.5, 70), (19.5, 80)]          # (heure, g de glucides)
MEALS = {0: MEALS_DAY, 1: MEALS_DAY}                      # mêmes repas les jours 1 et 2

EXTRA_BOLUS_HOUR = 24 + 15.0     # 15:00 jour 2
EXTRA_BOLUS_CHO = 40             # bolus d'erreur = 40 g / CR du patient

EX_START, EX_DUR_MIN, EX_HR_FACTOR = 24 + 16.0, 45, 1.8   # 16:00-16:45 jour 2

# Paramètres hormonaux DÉFINITIFS (calibrés sur un profil adulte, identiques
# pour les 30 patients) : ce sont ceux du tableau de l'article.
HYPO_PARAMS = {
    # cortisol
    "n_C": 0.009, "C_b": 80, "k_C": 0.015, "xi_C": 0.20,
    "SR_Cb": 0.72, "sigma_C": 0.50, "G_th_C": 70.0, "delta_C": 2.00,
    "A_circ_C": 5.0, "t_rise": 360, "k_rise": 0.02, "lambda_decay": 0.006,
    # GH
    "n_GH": 0.035, "GH_b": 3, "k_GH": 0.015, "xi_GH": 0.50,
    "SR_GHb": 0.105, "sigma_GH": 0.08, "G_th_GH": 70.0,
    "A_circ_GH": 0.55, "t_peak_GH": 180, "sigma_circ_GH": 60,
}
HYPO_OFF = {"xi_C": 0.0, "xi_GH": 0.0}   # modèle H0 (test d'équivalence)

SCENARIOS = {
    "S1_aube":     {"models": ["base", "PA", "H0", "H"], "extra_bolus": False, "exercise": False,
                    "windows": {"nuit_00_06": (24, 30), "aube_03_08": (27, 32),
                                "matin_06_12": (30, 36), "apm_12_18": (36, 42),
                                "soir_18_24": (42, 48), "jour2": DAY2}},
    "S2_hypo":     {"models": ["base", "H"], "extra_bolus": True, "exercise": False,
                    "windows": {"post_bolus_15_1930": (39, 43.5), "jour2": DAY2}},
    "S3_exercice": {"models": ["base", "PA", "H"], "extra_bolus": False, "exercise": True,
                    "windows": {"post_ex_16_24": (40, 48), "jour2": DAY2}},
}
CONTROLLERS = ["BBC", "PID"]
SIGNAL = "BG"


# ---------------------------------------------------------------------------
# Environnement
# ---------------------------------------------------------------------------
class OneShotScenarioPA(CustomScenarioPA):
    """Comme CustomScenarioPA, mais SANS répétition quotidienne : CustomScenarioPA
    calcule t % 24 et rejouerait l'exercice chaque jour. Ici, l'heure de début
    est comptée depuis START (40 h = 16:00 du jour 2), et la séance n'a lieu
    qu'une fois."""

    def get_action(self, t):
        t_hours = (t - self.start_time).total_seconds() / 3600.0
        for start, dur, factor in self.scenario:
            if start <= t_hours < start + dur / 60.0:
                return PAAction(CHO=0, insulin=0, HR=factor)
        return PAAction(CHO=0, insulin=0, HR=1.0)


def exercise_scenario(active):
    # format (heure_debut, duree_min, facteur_HR), cf. scenario_pa.py
    sessions = [(EX_START, EX_DUR_MIN, EX_HR_FACTOR)] if active else []
    return OneShotScenarioPA(start_time=START, scenario=sessions)


def make_patient(model, name):
    if model == "base":
        return T1DPatient.withName(name)
    if model == "PA":
        return T1DPatientPA.withName(name)
    params = dict(HYPO_PARAMS or {})
    if model == "H0":
        params.update(HYPO_OFF)
    return T1DPatientHypophyse.withName(name, circadian_params=params or None)


def make_env(model, name, seed, exercise):
    patient = make_patient(model, name)
    sensor = CGMSensor.withName(SENSOR, seed=seed)
    pump = InsulinPump.withName(PUMP)
    scen_meal = MultiDayCustomScenario(start_time=START, daily_scenarios=MEALS)
    # le modèle de base ignore HR : l'exercice n'a d'effet que sur PA et H
    return T1DSimEnv(patient, sensor, pump, scen_meal, exercise_scenario(exercise))


# ---------------------------------------------------------------------------
# Contrôleurs
# ---------------------------------------------------------------------------
QUEST = pd.read_csv(CONTROL_QUEST).set_index("Name")


class ExtraBolus:
    """Ajoute un bolus d'erreur (sans repas) à une heure donnée."""

    def __init__(self, ctrl, hour, units):
        self.ctrl, self.hour, self.units = ctrl, hour, units
        self.reset()

    def reset(self):
        self.ctrl.reset()
        self.t = 0.0          # minutes écoulées, compté ici (ne dépend pas de info)
        self.given = False

    def policy(self, observation, reward, done, **info):
        dt = info.get("sample_time", 3)
        a = self.ctrl.policy(observation, reward, done, **info)
        if not self.given and self.t / 60.0 >= self.hour:
            self.given = True
            a = Action(basal=a.basal, bolus=a.bolus + self.units / dt)   # U -> U/min
        self.t += dt
        return a


def make_controller(kind, name, gains, extra_bolus):
    if kind == "BBC":
        ctrl = BBController()
    else:
        g = gains.loc[name]
        ctrl = PIDController(P=g.P, I=g.I, D=g.D, correction=True)
    if extra_bolus:
        ctrl = ExtraBolus(ctrl, EXTRA_BOLUS_HOUR, EXTRA_BOLUS_CHO / QUEST.loc[name, "CR"])
    return ctrl


# ---------------------------------------------------------------------------
def window_metrics(df, h0, h1):
    hours = (pd.to_datetime(df.index) - START).total_seconds() / 3600
    bg = df[SIGNAL].to_numpy(dtype=float)[(hours >= h0) & (hours < h1)]
    if len(bg) < 2:
        return {}
    m = glycemic_metrics(bg)
    m.update({"min_bg": float(bg.min()), "max_bg": float(bg.max())})
    return m


def run_one(task):
    name, scen, model, kind, seed, gains_path = task
    spec = SCENARIOS[scen]
    base = {"patient": name, "groupe": name.split("#")[0], "scenario": scen,
            "model": model, "controller": kind, "seed": seed}
    try:
        gains = pd.read_csv(gains_path).set_index("patient")
        env = make_env(model, name, seed, spec["exercise"])
        ctrl = make_controller(kind, name, gains, spec["extra_bolus"])
        sim = SimObj(env, ctrl, timedelta(hours=SIM_HOURS), animate=False, path=None)
        sim.simulate()
        df = sim.results()
        expected = SIM_HOURS * 60 / env.sample_time
        ok = len(df) >= 0.98 * expected
        rows = []
        for w, (h0, h1) in spec["windows"].items():
            r = dict(base, window=w, n_steps=len(df), ok=ok,
                     error="" if ok else f"tronquee: {len(df)}/{expected:.0f}")
            r.update(window_metrics(df, h0, h1))
            rows.append(r)
        trace = None
        if seed == TEST_SEEDS[0]:
            hours = (pd.to_datetime(df.index) - START).total_seconds() / 3600
            trace = pd.DataFrame({"hour": hours, "BG": df["BG"].values,
                                  "CGM": df["CGM"].values, "insulin": df["insulin"].values})
            for k, v in base.items():
                trace[k] = v
        if spec["exercise"] and model != "base" and seed == TEST_SEEDS[0]:
            # contrôle : l'exercice a bien eu lieu (HR enregistrée si disponible)
            for col in ("HR", "hr"):
                if col in df:
                    rows[0]["hr_max_ratio"] = float(df[col].max())
        return rows, trace
    except Exception:
        err = traceback.format_exc(limit=3).replace("\n", " | ")
        return [dict(base, window="", ok=False, error=err)], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient-index", type=int, required=True)
    ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    ap.add_argument("--gains", default="gains_pid_v2.csv")
    ap.add_argument("--outdir", default="results_eval")
    ap.add_argument("--quick", action="store_true", help="1 graine seulement")
    args = ap.parse_args()

    name = patient_names()[args.patient_index]
    seeds = TEST_SEEDS[:1] if args.quick else TEST_SEEDS
    tasks = [(name, s, m, c, seed, args.gains)
             for s, spec in SCENARIOS.items() for m in spec["models"]
             for c in CONTROLLERS for seed in seeds]
    print(f"{name} : {len(tasks)} simulations de {SIM_HOURS} h sur {args.n_jobs} coeurs", flush=True)

    t0 = time.time()
    with Pool(args.n_jobs) as pool:
        out = pool.map(run_one, tasks, chunksize=1)
    rows = [r for rs, _ in out for r in rs]
    traces = [t for _, t in out if t is not None]

    os.makedirs(args.outdir, exist_ok=True)
    tag = name.replace("#", "_") + ("_quick" if args.quick else "")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.outdir, f"eval_{tag}.csv"), index=False)
    if traces:
        pd.concat(traces).to_csv(os.path.join(args.outdir, f"traces_{tag}.csv"), index=False)

    bad = res[~res.ok.astype(bool)]
    print(f"Terminé en {(time.time() - t0) / 60:.1f} min ; "
          f"{bad[['scenario', 'model', 'controller', 'seed']].drop_duplicates().shape[0]} "
          f"simulation(s) invalide(s) ou tronquée(s).", flush=True)
    if len(bad):
        print(bad[["scenario", "model", "controller", "seed", "error"]].drop_duplicates().head().to_string())

    # contrôle d'équivalence immédiat (S1, sans exercice ni hormones)
    s1 = res[(res.scenario == "S1_aube") & (res.window == "jour2") & res.ok.astype(bool)]
    if len(s1):
        piv = s1.pivot_table(index=["controller", "seed"], columns="model", values="mean_bg")
        for m in ("PA", "H0"):
            if m in piv and "base" in piv:
                print(f"Équivalence {m} vs base : écart max de glycémie moyenne = "
                      f"{(piv[m] - piv['base']).abs().max():.6f} mg/dL (attendu ~0)")
        if "H" in piv and "base" in piv:
            print(f"Effet hormonal H vs base (S1) : écart max = "
                  f"{(piv['H'] - piv['base']).abs().max():.3f} mg/dL (attendu > 0)")
    s3 = res[(res.scenario == "S3_exercice") & (res.window == "post_ex_16_24") & res.ok.astype(bool)]
    if len(s3):
        piv = s3.pivot_table(index=["controller", "seed"], columns="model", values="mean_bg")
        if "PA" in piv and "base" in piv:
            print(f"Effet de l'exercice PA vs base (S3, 16-24 h) : écart max = "
                  f"{(piv['PA'] - piv['base']).abs().max():.3f} mg/dL (attendu > 0)")


if __name__ == "__main__":
    main()