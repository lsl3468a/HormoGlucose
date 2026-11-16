from hormoglucose.patient.t1dpatientPA import Action
from hormoglucose.analysis.risk import risk_index
import pandas as pd
from datetime import timedelta
import logging
from collections import namedtuple
from hormoglucose.simulation.rendering import Viewer

try:
    from rllab.envs.base import Step
except ImportError:
    _Step = namedtuple("Step", ["observation", "reward", "done", "info"])

    def Step(observation, reward, done, **kwargs):
        """
        Convenience method creating a namedtuple with the results of the
        environment.step method.
        Put extra diagnostic info in the kwargs
        """
        return _Step(observation, reward, done, kwargs)


#Observation = namedtuple("Observation", ["CGM", "ACTH", "Cortisol", "GH"])
Observation = namedtuple("Observation", ["CGM"])

logger = logging.getLogger(__name__)


def risk_diff(BG_last_hour):
    if len(BG_last_hour) < 2:
        return 0
    else:
        _, _, risk_current = risk_index([BG_last_hour[-1]], 1)
        _, _, risk_prev = risk_index([BG_last_hour[-2]], 1)
        return risk_prev - risk_current


class T1DSimEnv(object):
    def __init__(self, patient, sensor, pump, scen_meal, scen_exe):
        self.patient = patient
        self.sensor = sensor
        self.pump = pump
        self.scenario_meal = scen_meal
        self.scenario_exe = scen_exe
        self._reset()

    @property
    def time(self):
        return self.scenario_meal.start_time + timedelta(minutes=self.patient.t)

    def mini_step(self, action):
        # Meal
        meal_action = self.scenario_meal.get_action(self.time)
        CHO = meal_action.meal

        #Exercise 
        exercise_action = self.scenario_exe.get_action(self.time)
        HR_factor = exercise_action.HR
        HRb = self.patient._params.HR_b
        HR = HR_factor * HRb 

        basal = self.pump.basal(action.basal)
        bolus = self.pump.bolus(action.bolus)
        insulin = basal + bolus

        patient_mdl_act = Action(insulin=insulin, CHO=CHO, HR=HR)

        # State update
        self.patient.step(patient_mdl_act)

        # next observation
        BG = self.patient.observation.Gsub
        CGM = self.sensor.measure(self.patient)
        # if self.patient.observation.GH :
        #     GH = self.patient.observation.GH
        #     ACTH = self.patient.observation.ACTH
        #     cortisol = self.patient.observation.Cortisol
        # else : 
        #     GH = None
        #     ACTH = None
        #     cortisol = None

        #return CHO, insulin, BG, CGM, HR, GH, ACTH, cortisol
        return CHO, insulin, BG, CGM, HR


    def step(self, action, reward_fun=risk_diff):
       
        """
        action is a namedtuple with keys: basal, bolus
        """
        CHO = 0.0
        insulin = 0.0
        BG = 0.0
        CGM = 0.0
        HR = 0.0
        #GH = 0.0
        #ACTH = 0.0
        #cortisol = 0.0

        for _ in range(int(self.sample_time)):
            # Compute moving average as the sample measurements
            # tmp_CHO, tmp_insulin, tmp_BG, tmp_CGM, tmp_HR, tmp_GH, tmp_ACTH, tmp_cortisol = self.mini_step(action)
            tmp_CHO, tmp_insulin, tmp_BG, tmp_CGM, tmp_HR = self.mini_step(action)
            CHO += tmp_CHO / self.sample_time
            insulin += tmp_insulin / self.sample_time
            BG += tmp_BG / self.sample_time
            CGM += tmp_CGM / self.sample_time
            #GH += tmp_GH / self.sample_time
            #ACTH += tmp_ACTH / self.sample_time
            #cortisol = tmp_cortisol / self.sample_time

            # #A SUPPRIMER PEUT ÊTRE
            HR += tmp_HR /self.sample_time
            
        
        # Compute risk index
        horizon = 1
        LBGI, HBGI, risk = risk_index([BG], horizon)

        # Record current action
        self.CHO_hist.append(CHO)
        self.insulin_hist.append(insulin)
        self.HR_hist.append(HR)
        
        # Record next observation
        self.time_hist.append(self.time)
        self.BG_hist.append(BG)
        self.CGM_hist.append(CGM)
        self.risk_hist.append(risk)
        self.LBGI_hist.append(LBGI)
        self.HBGI_hist.append(HBGI)
        #self.ACTH_list.append(ACTH)
        #self.cortisol_list.append(cortisol)
        #self.GH_list.append(GH)

        # Compute reward, and decide whether game is over
        window_size = int(60 / self.sample_time)
        BG_last_hour = self.CGM_hist[-window_size:]
        reward = reward_fun(BG_last_hour)
        done = BG < 10 or BG > 600
        #obs = Observation(CGM=CGM, ACTH=ACTH, Cortisol=cortisol, GH=GH)
        obs =  Observation(CGM=CGM)

        return Step(
            observation=obs,
            reward=reward,
            done=done,
            sample_time=self.sample_time,
            patient_name=self.patient.name,
            meal=CHO,
            hr = HR,
            patient_state=self.patient.state,
            time=self.time,
            bg=BG,
            lbgi=LBGI,
            hbgi=HBGI,
            risk=risk,
        )

    def _reset(self):
        self.sample_time = self.sensor.sample_time
        self.viewer = None

        BG = self.patient.observation.Gsub
        #ACTH = self.patient.observation.ACTH
        #GH = self.patient.observation.GH
        #cortisol = self.patient.observation.Cortisol
        horizon = 1
        LBGI, HBGI, risk = risk_index([BG], horizon)
        CGM = self.sensor.measure(self.patient)
        self.time_hist = [self.scenario_meal.start_time]
        self.BG_hist = [BG]
        self.CGM_hist = [CGM]
        self.risk_hist = [risk]
        self.LBGI_hist = [LBGI]
        self.HBGI_hist = [HBGI]
        self.CHO_hist = []
        self.insulin_hist = []
        self.HR_hist = []
        #self.ACTH_list = [ACTH]
        #self.GH_list = [GH]
        #self.cortisol_list = [cortisol]


    def reset(self):
        self.patient.reset()
        self.sensor.reset()
        self.pump.reset()
        self.scenario_meal.reset()
        self.scenario_exe.reset()
        self._reset()
        CGM = self.sensor.measure(self.patient)
        #ACTH = 0.0
        #cortisol = 0.0
        #GH = 0.0
        obs = Observation(CGM=CGM)
        #obs = Observation(CGM=CGM, ACTH=ACTH, Cortisol=cortisol, GH=GH)
        return Step(
            observation=obs,
            reward=0,
            done=False,
            sample_time=self.sample_time,
            patient_name=self.patient.name,
            meal=0,
            hr = 1.0,
            patient_state=self.patient.state,
            time=self.time,
            bg=self.BG_hist[0],
            lbgi=self.LBGI_hist[0],
            hbgi=self.HBGI_hist[0],
            risk=self.risk_hist[0],
        )

    def render(self, close=False):
        if close:
            self._close_viewer()
            return

        if self.viewer is None:
            self.viewer = Viewer(self.scenario_meal.start_time, self.patient.name)

        self.viewer.render(self.show_history())


    def _close_viewer(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        

    def show_history(self):
        df = pd.DataFrame()
        df["Time"] = pd.Series(self.time_hist)
        df["BG"] = pd.Series(self.BG_hist)
        df["CGM"] = pd.Series(self.CGM_hist)
        df["CHO"] = pd.Series(self.CHO_hist)
        df["insulin"] = pd.Series(self.insulin_hist)
        df["LBGI"] = pd.Series(self.LBGI_hist)
        df["HBGI"] = pd.Series(self.HBGI_hist)
        df["Risk"] = pd.Series(self.risk_hist)
        df["HR"] = pd.Series(self.HR_hist)
        #df["ACTH"] = pd.Series(self.ACTH_list)
        #df["GH"]=pd.Series(self.GH_list)
        #df["cortisol"] = pd.Series(self.cortisol_list)
        df = df.set_index("Time")
        return df
