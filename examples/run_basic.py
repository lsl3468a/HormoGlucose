"""Minimal example: one virtual patient, basal-bolus controller, 24 h,
baseline SimGlucose model vs. full HormoGlucose model (cortisol + GH).

    python examples/run_basic.py
"""
from datetime import datetime, timedelta

from hormoglucose.simulation.env import T1DSimEnv
from hormoglucose.simulation.sim_engine import SimObj
from hormoglucose.simulation.scenario_multiday import MultiDayCustomScenario
from hormoglucose.simulation.scenario_pa import CustomScenarioPA
from hormoglucose.patient.t1dpatient import T1DPatient
from hormoglucose.patient.t1dpatientHypophyse import T1DPatientHypophyse
from hormoglucose.sensor.cgm import CGMSensor
from hormoglucose.actuator.pump import InsulinPump
from hormoglucose.controller.basal_bolus_ctrller import BBController

START = datetime(2026, 1, 1, 0, 0)
MEALS = {0: [(8.0, 45), (12.5, 70), (19.5, 80)]}   # (hour, g of carbohydrates)
NAME = "adult#001"


def simulate(patient):
    env = T1DSimEnv(patient,
                    CGMSensor.withName("Dexcom", seed=1),
                    InsulinPump.withName("Insulet"),
                    MultiDayCustomScenario(start_time=START, daily_scenarios=MEALS),
                    CustomScenarioPA(start_time=START, scenario=[]))  # no exercise
    sim = SimObj(env, BBController(), timedelta(hours=24), animate=False, path=None)
    sim.simulate()
    return sim.results()


if __name__ == "__main__":
    base = simulate(T1DPatient.withName(NAME))
    full = simulate(T1DPatientHypophyse.withName(NAME))  # default = paper parameters
    dawn = slice(START + timedelta(hours=3), START + timedelta(hours=8))
    print(f"{NAME}, BBC, mean BG 03:00-08:00")
    print(f"  baseline (SimGlucose) : {base['BG'][dawn].mean():6.1f} mg/dL")
    print(f"  cortisol + GH         : {full['BG'][dawn].mean():6.1f} mg/dL")
