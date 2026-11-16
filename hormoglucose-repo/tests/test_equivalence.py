"""The extended models must reproduce the baseline SimGlucose model when the
added mechanisms are disabled (no exercise; xi_C = xi_GH = 0).

    pytest tests/
"""
from datetime import datetime, timedelta

import numpy as np
import pytest

from hormoglucose.simulation.env import T1DSimEnv
from hormoglucose.simulation.sim_engine import SimObj
from hormoglucose.simulation.scenario_multiday import MultiDayCustomScenario
from hormoglucose.simulation.scenario_pa import CustomScenarioPA
from hormoglucose.patient.t1dpatient import T1DPatient
from hormoglucose.patient.t1dpatientPA import T1DPatientPA
from hormoglucose.patient.t1dpatientHypophyse import T1DPatientHypophyse
from hormoglucose.sensor.cgm import CGMSensor
from hormoglucose.actuator.pump import InsulinPump
from hormoglucose.controller.basal_bolus_ctrller import BBController

START = datetime(2026, 1, 1, 0, 0)
MEALS = {0: [(8.0, 45), (12.5, 70)]}
TOL = 1e-2  # mg/dL (solver precision)


def bg(patient, hours=14):
    env = T1DSimEnv(patient, CGMSensor.withName("Dexcom", seed=1),
                    InsulinPump.withName("Insulet"),
                    MultiDayCustomScenario(start_time=START, daily_scenarios=MEALS),
                    CustomScenarioPA(start_time=START, scenario=[]))
    sim = SimObj(env, BBController(), timedelta(hours=hours), animate=False, path=None)
    sim.simulate()
    return sim.results()["BG"].to_numpy(dtype=float)


@pytest.mark.parametrize("name", ["adult#001", "child#001"])
def test_reduces_to_baseline(name):
    ref = bg(T1DPatient.withName(name))
    pa = bg(T1DPatientPA.withName(name))
    h0 = bg(T1DPatientHypophyse.withName(
        name, circadian_params={"xi_C": 0.0, "xi_GH": 0.0}))
    n = min(len(ref), len(pa), len(h0))
    assert n > 0.95 * len(ref)
    assert np.max(np.abs(pa[:n] - ref[:n])) < TOL
    assert np.max(np.abs(h0[:n] - ref[:n])) < TOL


def test_hormones_change_glucose():
    ref = bg(T1DPatient.withName("adult#001"), hours=9)
    full = bg(T1DPatientHypophyse.withName("adult#001"), hours=9)
    n = min(len(ref), len(full))
    assert np.max(np.abs(full[:n] - ref[:n])) > 1.0
