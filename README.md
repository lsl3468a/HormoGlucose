# HormoGlucose

**HormoGlucose** is an extension of [SimGlucose](https://github.com/jxx123/simglucose)
(J. Xie, Python implementation of the UVA/Padova type 1 diabetes simulator, 2008
version) with:

- **cortisol and growth hormone (GH) dynamics**: circadian secretion (dawn
  phenomenon) and a counterregulatory response to hypoglycemia, acting on
  endogenous glucose production through delayed actions;
- **a heart-rate-driven exercise model** (model C of Dalla Man et al., 2009),
  acting on insulin-independent and insulin-dependent glucose utilization.

It keeps the SimGlucose interface (patients, sensors, pumps, scenarios,
controllers, `SimObj`), so existing SimGlucose code can be ported by changing
the import paths from `simglucose` to `hormoglucose`.

| Model | Class | States | Content |
|---|---|---|---|
| Baseline | `hormoglucose.patient.t1dpatient.T1DPatient` | 13 | SimGlucose (UVA/Padova S2008) |
| Exercise | `hormoglucose.patient.t1dpatientPA.T1DPatientPA` | 16 | + exercise states Y, Z, W |
| Full | `hormoglucose.patient.t1dpatientHypophyse.T1DPatientHypophyse` | 25 | + cortisol, GH and their actions on EGP |

The nested models reproduce the baseline to within solver precision when the
added mechanisms are disabled (no exercise; `xi_C = xi_GH = 0`), see
`tests/test_equivalence.py`.

Like SimGlucose, the model contains no glucagon. The hormonal parameters were
calibrated on an adult profile and are shared by all 30 virtual patients.
**For research use only; not for clinical decisions.**

## Installation

```bash
git clone https://github.com/lsl3468a/HormoGlucose.git
cd HormoGlucose
pip install -e .
```

Python >= 3.9. `pathos` is optional (parallel simulations).

## Quick start

```bash
python examples/run_basic.py   # baseline vs. full model, adult#001, 24 h
pytest tests/                  # equivalence checks
```

```python
from datetime import datetime, timedelta
from hormoglucose.simulation.env import T1DSimEnv
from hormoglucose.simulation.sim_engine import SimObj
from hormoglucose.simulation.scenario import CustomScenario
from hormoglucose.simulation.scenario_pa import CustomScenarioPA
from hormoglucose.patient.t1dpatientHypophyse import T1DPatientHypophyse
from hormoglucose.sensor.cgm import CGMSensor
from hormoglucose.actuator.pump import InsulinPump
from hormoglucose.controller.basal_bolus_ctrller import BBController

start = datetime(2026, 1, 1)
meals = CustomScenario(start_time=start, scenario=[(8, 45), (12.5, 70), (19.5, 80)])
exercise = CustomScenarioPA(start_time=start, scenario=[(16.0, 45, 1.8)])  # (hour, min, HR/HR_b)
env = T1DSimEnv(T1DPatientHypophyse.withName("adult#001"),
                CGMSensor.withName("Dexcom", seed=1),
                InsulinPump.withName("Insulet"), meals, exercise)
sim = SimObj(env, BBController(), timedelta(hours=24), animate=False, path=None)
sim.simulate()
df = sim.results()
```

Hormonal parameters can be changed with
`T1DPatientHypophyse.withName(name, circadian_params={...})`; the default
values are those reported in the paper.

## Reproducing the LNNS 2026 experiments

`experiments/lnns2026/` contains the scripts of the extended paper
(controllers tuned on the baseline simulator and evaluated on the extended
models). Run them from that folder:

```bash
cd experiments/lnns2026
# 1. PID grid search on the baseline model (704 gain triplets x 3 days, per patient)
python grid_pid.py --patient-index 0 --n-jobs 8 --outdir results_grid
# 2. gain selection (consensus TBR constraints, then TIR, then LBGI+HBGI)
python select_gains.py --indir results_grid --out gains_pid.csv
# 3. evaluation scenarios S1 (dawn), S2 (erroneous bolus), S3 (exercise)
python eval_scenarios.py --patient-index 0 --n-jobs 8 --gains gains_pid_v2.csv
# 4. statistics (paired Wilcoxon, Holm), tables and figures
python analyse_eval.py --indir results_eval --outdir analyse
```

`--patient-index` goes from 0 to 29; `slurm/` contains array-job templates.
`eval_scenarios.py --quick` runs a short check (one seed) and prints the
equivalence tests. The tuned gains used in the paper are provided in
`gains_pid_v2.csv` (also in `hormoglucose/params/pid_gains_lnns2026.csv`).

## Citation

If you use HormoGlucose, please cite:

> Lasserre, L., Demongeot, J., Fougère, C., Doncescu, A.: An Extended SimGlucose Simulator for Type 1 Diabetes: Modeling Hormonal Counterregulation and Physical Activity. In: Proceedings of the 16th International Conference on Simulation and Modeling Methodologies, Technologies and Applications (SIMULTECH 2026), pp. 171--182. SCITEPRESS (2026). \doi{10.5220/0015188600004094}

and the original simulator:

> J. Xie, *Simglucose v0.2.1* (2018), https://github.com/jxx123/simglucose

## License

MIT (see `LICENSE`). Based on SimGlucose, Copyright (c) 2017 Jinyu Xie.
