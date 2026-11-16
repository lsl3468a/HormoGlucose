"""
Recherche sur grille des gains du PID (+ bolus de repas) sur le simulateur DE BASE.

Un appel = un patient. Pour chaque combinaison (P, I, D) et chaque scénario de
réglage (graine), on simule 24 h et on enregistre les métriques de la journée.
Le choix des gains se fait ensuite avec select_gains.py.

Test local rapide (8 combinaisons, 1 scénario) :
    python grid_pid.py --patient-index 0 --quick --n-jobs 2
Grille complète pour un patient :
    python grid_pid.py --patient-index 0 --n-jobs 8 --outdir results_grid
"""
import argparse
import itertools
import os
import time
import traceback
from datetime import datetime, timedelta
from multiprocessing import Pool

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# À ADAPTER (1/3) : chemins d'import de ton paquet
# ---------------------------------------------------------------------------
from hormoglucose.simulation.env import T1DSimEnv
from hormoglucose.patient.t1dpatient import T1DPatient
from hormoglucose.sensor.cgm import CGMSensor
from hormoglucose.actuator.pump import InsulinPump
from hormoglucose.simulation.scenario_gen import RandomScenario
from hormoglucose.simulation.scenario_pa import CustomScenarioPA
from hormoglucose.simulation.sim_engine import SimObj
from hormoglucose.controller.pid_ctrller import PIDController, PATIENT_PARA_FILE

# ---------------------------------------------------------------------------
# Protocole de réglage
# ---------------------------------------------------------------------------
# Capteur et pompe IDENTIQUES à ceux de l'évaluation du BBC
# (DEFAULT_CGM / DEFAULT_PUMP de rl_files/eval_model.py et evaluate_bygroup.py).
# Dexcom : un pas de simulation = 3 min -> 480 pas par jour.
SENSOR, PUMP = "Dexcom", "Insulet"

# Bolus de correction du PID : doit être IDENTIQUE à l'évaluation finale.
# True = bolus identiques à ceux du BBC ; False = correction laissée au terme P.
CORRECTION = True

START = datetime(2026, 1, 1, 0, 0)   # chaque journée commence à minuit
SIM_DAYS = 1                         # durée d'une simulation
TUNE_SEEDS = [101, 102, 103]         # scénarios de RÉGLAGE
# Les scénarios de TEST devront utiliser d'autres graines (par ex. >= 1000).

# "BG" = glucose sous-cutané SANS bruit capteur (Gsub = x13/Vg, pas le plasma) ;
# "CGM" = mesure bruitée du capteur, celle que voit le contrôleur.
SIGNAL = "BG"

# Grille : ordres de grandeur, ~3 points par décade, échelle logarithmique
P_GRID = np.logspace(-5, -3, 7)                              # 7 valeurs
I_GRID = np.concatenate([[0.0], np.logspace(-7, -5, 5)])     # 6 valeurs
D_GRID = np.concatenate([[0.0], np.logspace(-4, -2, 5)])     # 6 valeurs -> 252


# ---------------------------------------------------------------------------
def patient_names():
    names = pd.read_csv(PATIENT_PARA_FILE).Name
    return sorted(n for n in names
                  if n.split("#")[0] in ("adolescent", "adult", "child"))


def make_env(patient_name, seed):
    """
    Simulateur DE BASE : modèle UVA/Padova de T1DPatient, sans module
    hormonal (T1DPatientHypophyse / envH) ni exercice (T1DPatientPA).

    env.T1DSimEnv exige un scénario d'exercice : CustomScenarioPA avec une
    liste vide renvoie HR = 1.0 * HR_b à chaque minute, et T1DPatient ignore
    de toute façon HR (son step() n'utilise que CHO et insulin).

    Vérifié : BG et CGM identiques au bit près à simglucose 0.2.11 d'origine
    (3 patients x 2 graines x BBC/PID, 24 h ; scratchpad/check_base_sim.py).
    """
    patient = T1DPatient.withName(patient_name)
    sensor = CGMSensor.withName(SENSOR, seed=seed)
    pump = InsulinPump.withName(PUMP)
    scen_meal = RandomScenario(start_time=START, seed=seed)
    scen_exe = CustomScenarioPA(start_time=START, scenario=[])   # aucun exercice
    return T1DSimEnv(patient, sensor, pump, scen_meal, scen_exe)


def glycemic_metrics(bg):
    """Métriques du consensus (Battelino 2019) + LBGI/HBGI (Kovatchev 2017)."""
    bg = np.asarray(bg, dtype=float)
    f = 1.509 * (np.log(np.clip(bg, 1, None)) ** 1.084 - 5.381)
    risk = 10 * f ** 2
    return {
        "tir": 100 * np.mean((bg >= 70) & (bg <= 180)),
        "tbr70": 100 * np.mean(bg < 70),     # cumulatif (inclut < 54)
        "tbr54": 100 * np.mean(bg < 54),
        "tar180": 100 * np.mean(bg > 180),   # cumulatif (inclut > 250)
        "tar250": 100 * np.mean(bg > 250),
        "lbgi": float(np.mean(np.where(f < 0, risk, 0.0))),
        "hbgi": float(np.mean(np.where(f > 0, risk, 0.0))),
        "mean_bg": float(np.mean(bg)),
        "cv": 100 * float(np.std(bg, ddof=1) / np.mean(bg)),
    }


def run_one(task):
    patient_name, P, I, D, seed = task
    row = {"patient": patient_name, "P": P, "I": I, "D": D, "seed": seed}
    try:
        env = make_env(patient_name, seed)
        ctrl = PIDController(P=P, I=I, D=D, correction=CORRECTION)  # instance neuve
        assert type(env.patient) is T1DPatient, "simulateur de base attendu"
        sim_obj = SimObj(env, ctrl, timedelta(days=SIM_DAYS),
                         animate=False, path=None)
        sim_obj.simulate()                      # pas de sim() : évite l'écriture d'un CSV par run
        df = sim_obj.results()
        bg = df[SIGNAL].to_numpy(dtype=float)
        expected = SIM_DAYS * 24 * 60 / env.sample_time
        row.update(glycemic_metrics(bg))
        row["n_steps"] = len(bg)
        # une simulation tronquée (arrêt prématuré) est marquée comme invalide
        row["ok"] = len(bg) >= 0.98 * expected
        row["error"] = "" if row["ok"] else f"tronquee: {len(bg)}/{expected:.0f} pas"
    except Exception:
        row["ok"] = False
        row["error"] = traceback.format_exc(limit=2).replace("\n", " | ")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient-index", type=int, required=True,
                    help="0..29 (ordre alphabétique des noms)")
    ap.add_argument("--n-jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    ap.add_argument("--outdir", default="results_grid")
    ap.add_argument("--quick", action="store_true",
                    help="test : 8 combinaisons, 1 scénario")
    args = ap.parse_args()

    names = patient_names()
    assert len(names) == 30, f"{len(names)} patients trouvés au lieu de 30"
    name = names[args.patient_index]

    if args.quick:
        grid = list(itertools.product(P_GRID[[0, 4]], [0.0, 1e-6], [0.0, 1e-3]))
        seeds = TUNE_SEEDS[:1]
    else:
        grid = list(itertools.product(P_GRID, I_GRID, D_GRID))
        seeds = TUNE_SEEDS
    tasks = [(name, float(P), float(I), float(D), s)
             for (P, I, D) in grid for s in seeds]

    print(f"{name} : {len(grid)} combinaisons x {len(seeds)} scénarios "
          f"= {len(tasks)} simulations sur {args.n_jobs} coeurs", flush=True)
    t0 = time.time()
    with Pool(args.n_jobs) as pool:
        rows = list(pool.imap_unordered(run_one, tasks, chunksize=4))
    elapsed = time.time() - t0

    out = pd.DataFrame(rows).sort_values(["P", "I", "D", "seed"])
    os.makedirs(args.outdir, exist_ok=True)
    suffix = "_quick" if args.quick else ""
    path = os.path.join(args.outdir, f"grid_{name.replace('#', '_')}{suffix}.csv")
    out.to_csv(path, index=False)

    n_bad = int((~out.ok).sum())
    print(f"Terminé en {elapsed / 60:.1f} min "
          f"({elapsed / len(tasks):.2f} s/simulation, parallèle inclus). "
          f"{n_bad} simulation(s) invalide(s). -> {path}", flush=True)
    if n_bad:
        print(out.loc[~out.ok, ["P", "I", "D", "seed", "error"]].head().to_string())


if __name__ == "__main__":
    main()
