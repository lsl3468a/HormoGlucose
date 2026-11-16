from .base import Patient
import numpy as np
from scipy.integrate import ode
import pandas as pd
from collections import namedtuple
import logging
import pkg_resources

logger = logging.getLogger(__name__)

Action = namedtuple("patient_action", ["CHO", "insulin", "HR"])
Observation = namedtuple("observation", ["Gsub"])

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    "hormoglucose", "params/vpatient_params.csv"
)


class T1DPatientPA(Patient):
    SAMPLE_TIME = 1  # min
    EAT_RATE = 5  # g/min CHO
    T_Z = 3 * 600  # min, t_z = 3*Tex (Dalla Man 2009, eq. 12) : remise a 0 de W

    def __init__(self, params, init_state=None, random_init_bg=False, seed=None, t0=0):
        """
        T1DPatientPA constructor.
        Inputs:
            - params: a pandas sequence
            - init_state: customized initial state.
              If not specified, load the default initial state in
              params.iloc[2:15]
            - t0: simulation start time, it is 0 by default
        """
        self._params = params
        self._init_state = init_state
        self.random_init_bg = random_init_bg
        self._seed = seed
        self.t0 = t0
        self.planned_exercises = []  # Liste de sessions {'start', 'durée', 'factor'}
        self.reset()

    @classmethod
    def withID(cls, patient_id, **kwargs):
        """
        Construct patient by patient_id
        id are integers from 1 to 30.
        1  - 10: adolescent#001 - adolescent#010
        11 - 20: adult#001 - adult#001
        21 - 30: child#001 - child#010
        """
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        params = patient_params.iloc[patient_id - 1, :]
        return cls(params, **kwargs)

    @classmethod
    def withName(cls, name, **kwargs):
        """
        Construct patient by name.
        Names can be
            adolescent#001 - adolescent#010
            adult#001 - adult#001
            child#001 - child#010
        """
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        params = patient_params.loc[patient_params.Name == name].squeeze()
        return cls(params, **kwargs)

    @property
    def state(self):
        return self._odesolver.y

    @property
    def t(self):
        return self._odesolver.t

    @property
    def sample_time(self):
        return self.SAMPLE_TIME
    

    # Nouvelles fonctions pour l'exercice 
    def schedule_exercise(self, start, dur, factor):
        """Planifier une session d'exercice"""
        self.planned_exercises.append({"start": start, "dur": dur, "factor": factor})

    def clear_exercises(self):
        self.planned_exercises = []

    def _announce_exercise(self):
        """Déterminer HR(t) en fonction du planning"""
        HRb = self._params.HR_b
        hr = HRb
        for sess in self.planned_exercises:
            if sess["start"] <= self.t < sess["start"] + sess["dur"]:
                hr = HRb * sess["factor"]
                break
        return hr


    def step(self, action):
        # 1) Meals

        # Convert announcing meal to the meal amount to eat at the moment
        to_eat = self._announce_meal(action.CHO)
        action = action._replace(CHO=to_eat)

        # Detect eating or not and update last digestion amount
        if action.CHO > 0 and self._last_action.CHO <= 0:
            logger.info("t = {}, patient starts eating ...".format(self.t))
            self._last_Qsto = self.state[0] + self.state[1]  # unit: mg
            self._last_foodtaken = 0  # unit: g
            self.is_eating = True

        if to_eat > 0:
            logger.debug("t = {}, patient eats {} g".format(self.t, action.CHO))

        if self.is_eating:
            self._last_foodtaken += action.CHO  # g

        # Detect eating ended
        if action.CHO <= 0 and self._last_action.CHO > 0:
            logger.info("t = {}, Patient finishes eating!".format(self.t))
            self.is_eating = False


        # 2) Physical activity 
        HR = action.HR
        if HR > self._params.HR_b and self._last_action.HR <= self._params.HR_b:
            logger.info(f"t = {self.t}, patient starts exercise ...")
            self.is_exercising = True
            self._t_end_exercise = None
        if HR <= self._params.HR_b and self._last_action.HR > self._params.HR_b:
            logger.info(f"t = {self.t}, patient stops exercise.")
            self.is_exercising = False
            self._t_end_exercise = self.t

        # Dalla Man 2009, eq. (12) : W = 0 pour t >= t_z, instant ou Z revient
        # a zero, t_z = 3*Tex apres la fin du dernier exercice.
        if (self._t_end_exercise is not None
                and self.t - self._t_end_exercise >= self.T_Z):
            state = np.array(self.state, copy=True)
            state[15] = 0.0
            self._odesolver.set_initial_value(state, self.t)
            self._t_end_exercise = None


        # construire action enrichi avec HR
        action = action._replace(HR=HR)


        # Update last input
        self._last_action = action

        # ODE solver
        self._odesolver.set_f_params(
            action, self._params, self._last_Qsto, self._last_foodtaken
        )

        if self._odesolver.successful():
            self._odesolver.integrate(self._odesolver.t + self.sample_time)
        else:
            logger.error("ODE solver failed!!")
            raise



    @staticmethod
    def model(t, x, action, params, last_Qsto, last_foodtaken):
        alpha_ = 3e-4
        beta_ = 0.01
        gamma_ = 1e-7
        Thr = 5
        Tin = 1
        Tex = 600
        n = 4
        a = 0.1

        dxdt = np.zeros(16)
        d = action.CHO * 1000  # g -> mg
        insulin = action.insulin * 6000 / params.BW  # U/min -> pmol/kg/min
        basal = params.u2ss * params.BW / 6000  # U/min

        # Glucose in the stomach
        qsto = x[0] + x[1]
        Dbar = last_Qsto + last_foodtaken * 1000  # unit: mg

        if Dbar > 0:
            aa = 5 / (2 * Dbar * (1 - params.b))
            cc = 5 / (2 * Dbar * params.d)
            kgut = params.kmin + (params.kmax - params.kmin) / 2 * (
                np.tanh(aa * (qsto - params.b * Dbar))
                - np.tanh(cc * (qsto - params.d * Dbar))
                + 2
            )
        else:
            kgut = params.kmax

        # Stomach solid
        dxdt[0] = -params.kmax * x[0] + d

        # stomach liquid
        dxdt[1] = params.kmax * x[0] - x[1] * kgut

        # intestine
        dxdt[2] = kgut * x[1] - params.kabs * x[2]

        # Rate of appearance
        Rat = params.f * params.kabs * x[2] / params.BW
        # Glucose Production
        EGPt = params.kp1 - params.kp2 * x[3] - params.kp3 * x[8]
        # Glucose Utilization
        Uiit = params.Fcns

        # renal excretion
        if x[3] > params.ke2:
            Et = params.ke1 * (x[3] - params.ke2)
        else:
            Et = 0

        # glucose kinetics
        # plus dextrose IV injection input u[2] if needed
        dxdt[3] = max(EGPt, 0) + Rat - Uiit - Et - params.k1 * x[3] + params.k2 * x[4]
        dxdt[3] = (x[3] >= 0) * dxdt[3]


        # --- ACTIVITE PHYSIQUE ---

        HRb = params.HR_b
        HR = action.HR 
        HR_excess = max(HR - HRb, 0.0)
        Y, Z, W = x[13], x[14], x[15]

        Yt = -1/Thr * (Y - (HR - HRb))

        term = Y / (a * HRb) if (a * HRb) > 0 else 0.0
        fY = (term**n) / (1 + term**n)

        Zt = - (fY/Tin + 1/Tex) * Z + fY 

        # Dalla Man et al. 2009, modele C, eq. (12) : W = integrale de (HR - HRb)
        # en bpm.min, sans echelle ni fuite. La remise a 0 a t_z est faite dans
        # step() (discontinuite de l'etat, non representable dans l'EDO).
        Wt = HR - HRb

        numerateur = (params.Vm0 * ( 1 + beta_ * Y) + params.Vmx * ( 1 + alpha_ * Z * W) * (x[6] + params.Ib ) - params.Vmx * params.Ib )
        denominateur = params.Km0 * (1 - gamma_ * Z * W * (x[6] + params.Ib)) + x[4]

        Uidt = (numerateur / denominateur)* x[4]
        

        dxdt[13] = Yt
        dxdt[14] = Zt
        dxdt[15] = Wt

        dxdt[4] = -Uidt + params.k1 * x[3] - params.k2 * x[4]
        dxdt[4] = (x[4] >= 0) * dxdt[4]

        # insulin kinetics
        dxdt[5] = (
            -(params.m2 + params.m4) * x[5]
            + params.m1 * x[9]
            + params.ka1 * x[10]
            + params.ka2 * x[11]
        )  # plus insulin IV injection u[3] if needed
        It = x[5] / params.Vi
        dxdt[5] = (x[5] >= 0) * dxdt[5]

        # insulin action on glucose utilization
        dxdt[6] = -params.p2u * x[6] + params.p2u * (It - params.Ib)

        # insulin action on production
        dxdt[7] = -params.ki * (x[7] - It)

        dxdt[8] = -params.ki * (x[8] - x[7])

        # insulin in the liver (pmol/kg)
        dxdt[9] = -(params.m1 + params.m30) * x[9] + params.m2 * x[5]
        dxdt[9] = (x[9] >= 0) * dxdt[9]

        # subcutaneous insulin kinetics
        dxdt[10] = insulin - (params.ka1 + params.kd) * x[10]
        dxdt[10] = (x[10] >= 0) * dxdt[10]

        dxdt[11] = params.kd * x[10] - params.ka2 * x[11]
        dxdt[11] = (x[11] >= 0) * dxdt[11]

        # subcutaneous glucose
        dxdt[12] = -params.ksc * x[12] + params.ksc * x[3]
        dxdt[12] = (x[12] >= 0) * dxdt[12]

        if action.insulin > basal:
            logger.debug("t = {}, injecting insulin: {}".format(t, action.insulin))

        return dxdt

    @property
    def observation(self):
        """
        return the observation from patient
        for now, only the subcutaneous glucose level is returned
        TODO: add heart rate as an observation
        """
        GM = self.state[12]  # subcutaneous glucose (mg/kg)
        Gsub = GM / self._params.Vg
        observation = Observation(Gsub=Gsub)
        return observation

    def _announce_meal(self, meal):
        """
        patient announces meal.
        The announced meal will be added to self.planned_meal
        The meal is consumed in self.EAT_RATE
        The function will return the amount to eat at current time
        """
        self.planned_meal += meal
        if self.planned_meal > 0:
            to_eat = min(self.EAT_RATE, self.planned_meal)
            self.planned_meal -= to_eat
            self.planned_meal = max(0, self.planned_meal)
        else:
            to_eat = 0
        return to_eat

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.reset()

    def reset(self):
        """
        Reset the patient state to default intial state
        """
        if self._init_state is None:
            self.init_state = np.copy(self._params.iloc[2:15].values)
            self.init_state = np.append(self.init_state, [0.0, 0.0, 0.0])
        else:
            self.init_state = self._init_state

        self.random_state = np.random.RandomState(self.seed)
        if self.random_init_bg:
            # Only randomize glucose related states, x4, x5, and x13
            mean = [
                1.0 * self.init_state[3],
                1.0 * self.init_state[4],
                1.0 * self.init_state[12],
            ]
            cov = np.diag(
                [
                    0.1 * self.init_state[3],
                    0.1 * self.init_state[4],
                    0.1 * self.init_state[12],
                ]
            )
            bg_init = self.random_state.multivariate_normal(mean, cov)
            self.init_state[3] = 1.0 * bg_init[0]
            self.init_state[4] = 1.0 * bg_init[1]
            self.init_state[12] = 1.0 * bg_init[2]

        self._last_Qsto = self.init_state[0] + self.init_state[1]
        self._last_foodtaken = 0
        self.name = self._params.Name

        self._odesolver = ode(self.model).set_integrator("dopri5")
        self._odesolver.set_initial_value(self.init_state, self.t0)

        self._last_action = Action(CHO=0, insulin=0, HR=self._params.HR_b)
        self.is_eating = False
        self.is_exercising = False
        self._t_end_exercise = None
        self.planned_meal = 0


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)
    ch.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    p = T1DPatientPA.withName("adolescent#001")
    basal = p._params.u2ss * p._params.BW / 6000  # U/min
    t = []
    CHO = []
    insulin = []
    BG = []
    while p.t < 1000:
        ins = basal
        carb = 0
        if p.t == 100:
            carb = 80
            ins = 80.0 / 6.0 + basal
        # if p.t == 150:
        #     ins = 80.0 / 12.0 + basal
        act = Action(insulin=ins, CHO=carb, HR=p._params.HR_b)
        t.append(p.t)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)
        p.step(act)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(t, BG)
    ax[1].plot(t, CHO)
    ax[2].plot(t, insulin)
    plt.show()