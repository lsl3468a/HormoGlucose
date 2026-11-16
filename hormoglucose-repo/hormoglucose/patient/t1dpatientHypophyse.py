from .base import Patient
import numpy as np
from scipy.integrate import ode, solve_ivp
import pandas as pd
from collections import namedtuple
import logging
import pkg_resources
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# ── Action enrichie avec HR (comme T1DPatientPA) ──────────────────────────
Action = namedtuple("patient_action", ["CHO", "insulin", "HR"])
Observation = namedtuple("observation", ["Gsub", "Cortisol", "GH"])

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    "hormoglucose", "params/vpatient_params.csv"
)


class T1DPatientHypophyse(Patient):
    SAMPLE_TIME = 1  # min
    EAT_RATE = 5     # g/min CHO
    T_Z = 3 * 600    # min, t_z = 3*Tex (Dalla Man 2009, eq. 12) : remise a 0 de W

    def __init__(self, params, init_state=None, random_init_bg=False,
                 seed=None, t0=0, circadian_params=None):
        self._params = params
        self._init_state = init_state
        self.random_init_bg = random_init_bg
        self._seed = seed
        self.t0 = t0

        # ── Paramètres hypophysaires (cortisol + GH) ──────────────────────
        self.hypophyse_params = {
        'n_C':          0.009,
        'C_b':          80,
        'k_C':          0.015,
        'xi_C':         0.20,

        'SR_Cb':        0.72,
        'sigma_C':      0.50,   
        'G_th_C':       70.0,  
        'delta_C':      2.00, 

        
        'A_circ_C':     5.0,
        't_rise':       360,
        'k_rise':       0.02,
        'lambda_decay': 0.006,


        # ── GH ────────────────────────────────────────────────────────────────
        'n_GH':         0.035,
        'GH_b':         3,
        'k_GH':         0.015,
        'xi_GH':        0.50,

        'SR_GHb':       0.105,
        'sigma_GH':     0.08,   # réponse statique à l'hypo
        'G_th_GH':      70.0,   # CORRIGÉ : aligné sur G_th_C (était 80)
        'delta_GH':     0.50,   # réponse dynamique à −dG/dt

        'A_circ_GH':    0.55,   # inchangé (GH circadienne déjà bien calibrée)
        't_peak_GH':    180,
        'sigma_circ_GH':60,


        'k_C_EGP':      0.30,
        'k_GH_EGP':     0.50,
        }

        if circadian_params:
            self.hypophyse_params.update(circadian_params)

        self.reset()

    # ── Constructeurs ─────────────────────────────────────────────────────
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


    # ── Propriétés ────────────────────────────────────────────────────────
    @property
    def state(self):
        return self._odesolver.y

    @property
    def t(self):
        return self._odesolver.t

    @property
    def sample_time(self):
        return self.SAMPLE_TIME
    


    # ── step ──────────────────────────────────────────────────────────────
    def step(self, action):
        # 1) Repas
        to_eat = self._announce_meal(action.CHO)
        action = action._replace(CHO=to_eat)

        if action.CHO > 0 and self._last_action.CHO <= 0:
            logger.info("t = {}, patient starts eating ...".format(self.t))
            self._last_Qsto = self.state[0] + self.state[1]
            self._last_foodtaken = 0
            self.is_eating = True
        if to_eat > 0:
            logger.debug("t = {}, patient eats {} g".format(self.t, action.CHO))
        if self.is_eating:
            self._last_foodtaken += action.CHO
        if action.CHO <= 0 and self._last_action.CHO > 0:
            logger.info("t = {}, Patient finishes eating!".format(self.t))
            self.is_eating = False

        # 2) Exercice — détecter début/fin comme dans T1DPatientPA
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

        self._last_action = action

        self._odesolver.set_f_params(
            action, self._params, self.hypophyse_params,
            self._last_Qsto, self._last_foodtaken
        )
        if self._odesolver.successful():
            self._odesolver.integrate(self._odesolver.t + self.sample_time)
        else:
            logger.error("ODE solver failed!!")
            raise

    # ── Modèle ODE ────────────────────────────────────────────────────────
    @staticmethod
    def model(t, x, action, params, hypo_params, last_Qsto, last_foodtaken):
        """
        États (26 total) :
            x[0]  – estomac solide
            x[1]  – estomac liquide
            x[2]  – intestin
            x[3]  – glucose plasmatique (mg/kg)
            x[4]  – glucose tissu (mg/kg)
            x[5]  – insuline plasma (pmol/kg)
            x[6]  – action insuline sur utilisation
            x[7]  – action insuline sur production (intermédiaire)
            x[8]  – action insuline sur production
            x[9]  – insuline foie
            x[10] – insuline SC compartiment 1
            x[11] – insuline SC compartiment 2
            x[12] – glucose SC
            x[13] – Y  : état exercice (filtre HR)        ← T1DPatientPA
            x[14] – Z  : état exercice (utilisation)      ← T1DPatientPA
            x[15] – W  : état exercice (sensibilité)      ← T1DPatientPA
            x[16] – ACTH (non utilisé, conservé)
            x[17] – X_A (non utilisé, conservé)
            x[18] – Cortisol C
            x[19] – X_C (action retardée cortisol)
            x[20] – GH
            x[21] – X_GH (action retardée GH)
            x[22] – E_hypo (HAAF)
        #   x[23] – u  : oscillateur van der Pol cortisol (état 1)
        #   x[24] – v  : oscillateur van der Pol cortisol (état 2)
        """
        # ── Paramètres exercice (identiques à T1DPatientPA) ───────────────
        alpha_ = 3e-4
        beta_  = 0.01
        gamma_ = 1e-7
        Thr = 5
        Tin = 1
        Tex = 600
        n   = 4
        a   = 0.1

        dxdt = np.zeros(25)

        d      = action.CHO * 1000                  # g → mg
        insulin= action.insulin * 6000 / params.BW  # U/min → pmol/kg/min
        basal  = params.u2ss * params.BW / 6000      # U/min

        # ── Digestion ─────────────────────────────────────────────────────
        qsto = x[0] + x[1]
        Dbar = last_Qsto + last_foodtaken * 1000

        dxdt[0] = -params.kmax * x[0] + d

        if Dbar > 0:
            aa   = 5 / (2 * Dbar * (1 - params.b))
            cc   = 5 / (2 * Dbar * params.d)
            kgut = params.kmin + (params.kmax - params.kmin) / 2 * (
                np.tanh(aa * (qsto - params.b * Dbar))
                - np.tanh(cc * (qsto - params.d * Dbar))
                + 2
            )
        else:
            kgut = params.kmax

        dxdt[1] = params.kmax * x[0] - x[1] * kgut
        dxdt[2] = kgut * x[1] - params.kabs * x[2]

        Rat = params.f * params.kabs * x[2] / params.BW

        # ── EGP (base + cortisol + GH) ────────────────────────────────────
        # Calculee AVANT les secretions : dxdt[3] ne depend que de l'etat x
        # (X_C = x[19], X_GH = x[21]), et SR_C^d en a besoin via dG/dt.
        C_scale      = 100.0
        GH_scale     = 5.0
        EGP_base     = params.kp1 - params.kp2 * x[3] - params.kp3 * x[8]
        EGP_cortisol = hypo_params['xi_C']  * (x[19] / C_scale)
        EGP_GH       = hypo_params['xi_GH'] * (x[21] / GH_scale)
        EGPt         = max(EGP_base + EGP_cortisol + EGP_GH, 0)

        # ── Excrétion rénale ──────────────────────────────────────────────
        Et = params.ke1 * (x[3] - params.ke2) if x[3] > params.ke2 else 0

        # ── Cinétique du glucose plasmatique (eq. A1) ─────────────────────
        Uiit     = params.Fcns
        dxdt[3]  = max(EGPt, 0) + Rat - Uiit - Et - params.k1 * x[3] + params.k2 * x[4]
        dxdt[3]  = (x[3] >= 0) * dxdt[3]

        # ── Glycémie et dérivée (pour cortisol) ───────────────────────────
        G      = x[3] / params.Vg
        dG_dt  = dxdt[3] / params.Vg   # mg/dL/min, derivee instantanee (eq. 13)


        # ── ACTH (non utilisé) ────────────────────────────────────────────
        dxdt[16] = 0.0
        dxdt[17] = 0.0

        # ── Cortisol ──────────────────────────────────────────────────────
        C       = x[18]
        SR_C_static   = (hypo_params['SR_Cb'] + max(0, hypo_params['sigma_C']
                         * (hypo_params['G_th_C'] - G)))
        
        SR_C_dynamic  = hypo_params['delta_C'] * max(0, -dG_dt)

        t_mod = t % (24 * 60)
        t_rel         = t_mod - hypo_params['t_rise']
        logistic      = 1 / (1 + np.exp(-hypo_params['k_rise'] * t_rel))
        decay         = np.exp(-hypo_params['lambda_decay'] * max(0, t_rel))
        SR_C_circadian= hypo_params['A_circ_C'] * logistic * decay


        SR_C          = SR_C_static + SR_C_dynamic + SR_C_circadian
        dxdt[18]      = -hypo_params['n_C'] * C + SR_C

        X_C      = x[19]
        dxdt[19] = (-hypo_params['k_C'] * X_C
                    + hypo_params['k_C'] * max(C - hypo_params['C_b'], 0))

        # ── GH ────────────────────────────────────────────────────────────
        GH             = x[20]
        SR_GH_static   = (hypo_params['SR_GHb']
                          + max(0, hypo_params['sigma_GH']
                                * (hypo_params['G_th_GH'] - G)))
        

        # ── GH circadien — gaussienne périodique ─────────────────────────────
        t_mod         = t % (24 * 60)
        t_rel_GH = (t_mod - hypo_params['t_peak_GH']) % 1440
        if t_rel_GH > 720:
            t_rel_GH -= 1440

        SR_GH_circadian = hypo_params['A_circ_GH'] * np.exp(
            -(t_rel_GH ** 2) / (2 * hypo_params['sigma_circ_GH'] ** 2))

        SR_GH_max = hypo_params['SR_GHb'] + hypo_params['A_circ_GH'] + 0.1
        SR_GH = np.clip(
            SR_GH_static + SR_GH_circadian,
            hypo_params['SR_GHb'],
            SR_GH_max
        )

        dxdt[20] = -hypo_params['n_GH'] * GH + SR_GH

        X_GH     = x[21]
        dxdt[21] = (-hypo_params['k_GH'] * X_GH
                    + hypo_params['k_GH'] * max(GH - hypo_params['GH_b'], 0))

        # ── Utilisation du glucose — AVEC exercice (T1DPatientPA) ─────────
        HRb       = params.HR_b
        HR        = action.HR
        HR_excess = max(HR - HRb, 0.0)

        Y, Z, W = x[13], x[14], x[15]

        # Filtre HR → Y
        dxdt[13] = -1 / Thr * (Y - (HR - HRb))

        # Fonction sigmoïde sur Y
        term = Y / (a * HRb) if (a * HRb) > 0 else 0.0
        fY   = (term ** n) / (1 + term ** n)

        # Dynamique Z (utilisation musculaire)
        dxdt[14] = -(fY / Tin + 1 / Tex) * Z + fY

        # Dynamique W (Dalla Man et al. 2009, modele C, eq. 12) : integrale de
        # (HR - HRb) en bpm.min, sans echelle ni fuite. Remise a 0 a t_z dans
        # step() (discontinuite de l'etat, non representable dans l'EDO).
        dxdt[15] = HR - HRb

        # Utilisation insulino-dépendante MODIFIÉE par l'exercice
        numerateur   = (params.Vm0 * (1 + beta_ * Y)
                        + params.Vmx * (1 + alpha_ * Z * W)
                        * (x[6] + params.Ib)
                        - params.Vmx * params.Ib)
        denominateur = (params.Km0
                        * (1 - gamma_ * Z * W * (x[6] + params.Ib))
                        + x[4])
        Uidt = (numerateur / denominateur) * x[4]

        # ── Cinétique glucose tissulaire (dxdt[3] calcule plus haut) ──────
        dxdt[4]  = -Uidt + params.k1 * x[3] - params.k2 * x[4]
        dxdt[4]  = (x[4] >= 0) * dxdt[4]

        # ── Cinétique insuline ────────────────────────────────────────────
        dxdt[5] = (-(params.m2 + params.m4) * x[5]
                   + params.m1 * x[9]
                   + params.ka1 * x[10]
                   + params.ka2 * x[11])
        It      = x[5] / params.Vi
        dxdt[5] = (x[5] >= 0) * dxdt[5]

        dxdt[6] = -params.p2u * x[6] + params.p2u * (It - params.Ib)
        dxdt[7] = -params.ki * (x[7] - It)
        dxdt[8] = -params.ki * (x[8] - x[7])

        dxdt[9] = -(params.m1 + params.m30) * x[9] + params.m2 * x[5]
        dxdt[9] = (x[9] >= 0) * dxdt[9]

        dxdt[10] = insulin - (params.ka1 + params.kd) * x[10]
        dxdt[10] = (x[10] >= 0) * dxdt[10]

        dxdt[11] = params.kd * x[10] - params.ka2 * x[11]
        dxdt[11] = (x[11] >= 0) * dxdt[11]

        dxdt[12] = -params.ksc * x[12] + params.ksc * x[3]
        dxdt[12] = (x[12] >= 0) * dxdt[12]

        if action.insulin > basal:
            logger.debug("t = {}, injecting insulin: {}".format(t, action.insulin))

        return dxdt

    # ── Observation ───────────────────────────────────────────────────────
    @property
    def observation(self):
        GM       = self.state[12]
        Gsub     = GM / self._params.Vg
        Cortisol = self.state[18]
        GH       = self.state[20]
        return Observation(Gsub=Gsub, Cortisol=Cortisol, GH=GH)

    # ── Repas ─────────────────────────────────────────────────────────────
    def _announce_meal(self, meal):
        self.planned_meal += meal
        if self.planned_meal > 0:
            to_eat = min(self.EAT_RATE, self.planned_meal)
            self.planned_meal -= to_eat
            self.planned_meal = max(0, self.planned_meal)
        else:
            to_eat = 0
        return to_eat

    # ── Seed ──────────────────────────────────────────────────────────────
    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.reset()

    # ── Reset ─────────────────────────────────────────────────────────────
    def reset(self):
        """
        Initialise les 23 états :
            0-12  : états glucidiques/insuliniques standards
            13-15 : états exercice Y, Z, W  (nouveaux)
            16-17 : ACTH/X_A (non utilisés)
            18-19 : Cortisol C, X_C
            20-21 : GH, X_GH
            22    : E_hypo (HAAF)
        """
        if self._init_state is None:
            self.init_state = np.copy(self._params.iloc[2:27].values)
        else:
            self.init_state = self._init_state

        self.random_state = np.random.RandomState(self.seed)
        if self.random_init_bg:
            mean = [1.0 * self.init_state[3],
                    1.0 * self.init_state[4],
                    1.0 * self.init_state[12]]
            cov  = np.diag([0.1 * self.init_state[3],
                            0.1 * self.init_state[4],
                            0.1 * self.init_state[12]])
            bg_init = self.random_state.multivariate_normal(mean, cov)
            self.init_state[3]  = bg_init[0]
            self.init_state[4]  = bg_init[1]
            self.init_state[12] = bg_init[2]

        # ── États exercice : repos → Y=0, Z=0, W=0 ───────────────────────
        self.init_state[13] = 0.0   # Y
        self.init_state[14] = 0.0   # Z
        self.init_state[15] = 0.0   # W

        # ── États hypophysaires ───────────────────────────────────────────
        self.init_state[16] = 0.0                            # ACTH (unused)
        self.init_state[17] = 0.0                            # X_A  (unused)
        self.init_state[18] = self.hypophyse_params['C_b']  # Cortisol
        self.init_state[19] = self.hypophyse_params['C_b']  # X_C
        self.init_state[20] = self.hypophyse_params['GH_b'] # GH
        self.init_state[21] = self.hypophyse_params['GH_b'] # X_GH
        self.init_state[22] = 0.0
        self.init_state[23] = 0  
        self.init_state[24] = 0

        self._last_Qsto     = self.init_state[0] + self.init_state[1]
        self._last_foodtaken= 0
        self.name           = self._params.Name

        self._odesolver = ode(self.model).set_integrator("dopri5")
        self._odesolver.set_initial_value(self.init_state, self.t0)

        # HR_b = fréquence cardiaque au repos (dans vpatient_params.csv)
        self._last_action = Action(CHO=0, insulin=0, HR=self._params.HR_b)
        self.is_eating    = False
        self.is_exercising= False
        self._t_end_exercise = None
        self.planned_meal = 0

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import logging

    logging.basicConfig(level=logging.INFO)

    # ═══════════════════════════════════════════════════════════════════════════
    #  PARAMÈTRES DU SCÉNARIO
    # ═══════════════════════════════════════════════════════════════════════════
    INSULIN_FACTOR  = 11.0    # × basal
    BOLUS_TIME      = 840    # min (14h00)
    BOLUS_DURATION  = 30     # min

    EXO_START       = 960    # min (16h00)
    EXO_DUR         = 45     # min
    HR_FACTOR       = 2.2    # × HR_b

    # ═══════════════════════════════════════════════════════════════════════════
    #  CORRECTIONS PARAMÈTRES HYPOPHYSAIRES
    #
    #  Bug 1 — A_circ_C recalibrée :
    #    Pic physiologique cortisol ≈ 15–20 nmol/L/min
    #    A l'état stationnaire : C = SR_C / n_C
    #    On veut pic circadien ≈ 5 nmol/L/min  →  A_circ_C = 5 (était 35)
    #
    #  Bug 2 — SR_C/GH statique : G_th abaissé à 70 (seuil réel de contre-régulation)
    #    Le seuil de 80 était trop haut → activait la statique même en normoglycémie
    #
    #  Bug 3 — delta_C / delta_GH : réponse dynamique conservée mais cohérente
    #    avec l'unité de dG_dt (mg/dL/min), typiquement −0.5 à −2 mg/dL/min
    #    pendant une chute → delta_C=0.50 donne SR_C_dyn ≈ 0.25–1.0 nmol/L/min ✓
    # ═══════════════════════════════════════════════════════════════════════════
    circadian_hypo = {
        # ── Cortisol ──────────────────────────────────────────────────────────
        'n_C':          0.009,
        'C_b':          80,
        'k_C':          0.015,
        'xi_C':         0.20,

        'SR_Cb':        0.72,
        'sigma_C':      0.50,   
        'G_th_C':       70.0,  
        'delta_C':      2.00, 

        
        'A_circ_C':     5.0,
        't_rise':       360,
        'k_rise':       0.02,
        'lambda_decay': 0.006,


        # ── GH ────────────────────────────────────────────────────────────────
        'n_GH':         0.035,
        'GH_b':         3,
        'k_GH':         0.015,
        'xi_GH':        0.50,

        'SR_GHb':       0.105,
        'sigma_GH':     0.08,   # réponse statique à l'hypo
        'G_th_GH':      70.0,   # CORRIGÉ : aligné sur G_th_C (était 80)
        'delta_GH':     0.50,   # réponse dynamique à −dG/dt

        'A_circ_GH':    0.55,   # inchangé (GH circadienne déjà bien calibrée)
        't_peak_GH':    180,
        'sigma_circ_GH':60,


        'k_C_EGP':      0.30,
        'k_GH_EGP':     0.50,
    }

    # ── Helper composantes — CORRIGÉ : utilise G_plasma = x[3]/Vg ─────────────
    # Bug 2 corrigé : on passe explicitement G_plasma (pas Gsub=x[12]/Vg)
    # Bug 1 corrigé : dG_dt calculé sur x[3] cohérent avec l'ODE
    def _secretion_components(t, G_plasma, dG_dt, hp, u_vdp):
        """
        G_plasma : glucose plasmatique = x[3] / Vg  [mg/dL]
        dG_dt    : dérivée sur x[3]/Vg              [mg/dL/min]
        """
        t_mod = t % (24 * 60)

        # Cortisol
        SR_C_static  = hp['SR_Cb'] + max(0, hp['sigma_C']  * (hp['G_th_C']  - G_plasma))
        SR_C_dynamic = hp['delta_C']  * max(0, -dG_dt)
        t_rel        = t_mod - hp['t_rise']
        logistic     = 1 / (1 + np.exp(-hp['k_rise'] * t_rel))
        decay        = np.exp(-hp['lambda_decay'] * max(0, t_rel))
        SR_C_circ    = hp['A_circ_C'] * logistic * decay

        # p_shape = 3
        # SR_C_circ = hp['A_circ_C'] * max(u_vdp, 0)**p_shape

        # GH
        SR_GH_static  = hp['SR_GHb'] + max(0, hp['sigma_GH'] * (hp['G_th_GH'] - G_plasma))
        
        t_rel_GH = (t_mod - hp['t_peak_GH']) % 1440
        if t_rel_GH > 720:
            t_rel_GH -= 1440
        SR_GH_circ = hp['A_circ_GH'] * np.exp(
            -(t_rel_GH ** 2) / (2 * hp['sigma_circ_GH'] ** 2))

        return SR_C_static, SR_C_dynamic, SR_C_circ, SR_GH_static, SR_GH_circ

    # ── Patient ────────────────────────────────────────────────────────────────
    p     = T1DPatientHypophyse.withName("adult#001", circadian_params=circadian_hypo)
    basal = p._params.u2ss * p._params.BW / 6000
    HRb   = p._params.HR_b
    hp    = p.hypophyse_params
    Vg    = p._params.Vg

    # CORRIGÉ Bug 2 : prev_G sur x[3]/Vg (plasma), pas x[12]/Vg (SC)
    prev_G_plasma = p.state[3] / Vg

    # ── Listes ─────────────────────────────────────────────────────────────────
    t_list         = []
    BG_sc_list     = []    # glucose SC pour affichage (Gsub)
    BG_pl_list     = []    # glucose plasma pour les composantes
    C_list         = []
    GH_list        = []
    C_static_list  = []; C_dynamic_list  = []; C_circ_list  = []
    GH_static_list = [];  GH_circ_list = []

    # ── Boucle 24h ─────────────────────────────────────────────────────────────
    while p.t < 1440:
        t = p.t

        # Insuline : basal + bolus
        ins = basal * INSULIN_FACTOR if BOLUS_TIME <= t < BOLUS_TIME + BOLUS_DURATION else basal

        # Exercice
        hr = HRb * HR_FACTOR if EXO_START <= t < EXO_START + EXO_DUR else HRb

        act = Action(CHO=0, insulin=ins, HR=hr)

        # CORRIGÉ Bug 1+2 : G et dG_dt sur glucose PLASMA x[3]/Vg
        G_plasma = p.state[3] / Vg
        dG_dt    = G_plasma - prev_G_plasma   # mg/dL/min (Δt = 1 min)

        u_vdp = p.state[23]
        sc, dc, cc, sg, cg = _secretion_components(t, G_plasma, dG_dt, hp, u_vdp)

        t_list.append(t)
        BG_sc_list.append(p.observation.Gsub)      # SC — pour le plot BG
        BG_pl_list.append(G_plasma)                 # plasma — pour vérification
        C_list.append(p.state[18])
        GH_list.append(p.state[20])
        C_static_list.append(sc);  C_dynamic_list.append(dc);  C_circ_list.append(cc)
        GH_static_list.append(sg); GH_circ_list.append(cg)

        prev_G_plasma = G_plasma
        p.step(act)

    # ── Figure ─────────────────────────────────────────────────────────────────
    t_arr   = np.array(t_list)
    BG_arr  = np.array(BG_sc_list)
    BGp_arr = np.array(BG_pl_list)

    BLUE   = "#2563EB"; RED    = "#DC2626"; GREEN  = "#16A34A"
    PURPLE = "#7C3AED"; GOLD   = "#D97706"; CYAN   = "#0891B2"
    ORANGE = "#EA580C"; GRAY   = "#6B7280"

    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    fig.suptitle(
        f"adult#001 — Hypoglycémie (bolus ×{INSULIN_FACTOR:.0f} à 14h + exercice à 16h)\n",
        fontsize=11, fontweight="bold"
    )

    def shade(ax):
        ax.axvspan(BOLUS_TIME, BOLUS_TIME + BOLUS_DURATION,
                   alpha=0.12, color=RED,  label=f"Bolus ×{INSULIN_FACTOR:.0f} (14h)")
        ax.axvspan(EXO_START,  EXO_START  + EXO_DUR,
                   alpha=0.15, color=BLUE, label="Exercice (16h)")
        ax.grid(True, alpha=0.3)

    # ── Panel 1 : Glycémie SC + plasma ───────────────────────────────────────
    ax = axes[0]
    ax.plot(t_arr, BG_arr,  color=RED,  lw=2.2, label="BG sous-cutané (mg/dL)")
    ax.plot(t_arr, BGp_arr, color=RED,  lw=1.2, ls="--", alpha=0.5, label="BG plasma (mg/dL)")
    ax.axhspan(70, 180, alpha=0.07, color=GREEN, label="Cible 70–180")
    ax.axhline(70, color=ORANGE, ls="--", lw=1.2, label="Seuil hypo 70")
    ax.axhline(54, color=RED,    ls=":",  lw=1.2, label="Hypo sévère 54")
    shade(ax)
    idx_min = int(np.argmin(BG_arr))
    ax.annotate(f"Nadir {BG_arr[idx_min]:.1f} mg/dL",
                xy=(t_arr[idx_min], BG_arr[idx_min]),
                xytext=(t_arr[idx_min] + 80, BG_arr[idx_min] + 20),
                arrowprops=dict(arrowstyle="->", lw=1.2),
                fontsize=9)
    ax.set_ylabel("Glycémie\n(mg/dL)", fontsize=10)
    ax.legend(fontsize=7, loc="upper right", ncol=3)
    ax.set_ylim(bottom=0)

    # ── Panel 2 : Cortisol & GH (états ODE) ─────────────────────────────────
    ax  = axes[1]
    ax2 = ax.twinx()
    ax.plot(t_arr,  C_list,  color=GREEN,  lw=2.2, label="Cortisol (nmol/L)")
    ax.axhline(hp['C_b'],  color=GREEN,  ls="--", lw=1, alpha=0.6,
               label=f"C_b = {hp['C_b']} nmol/L")
    ax2.plot(t_arr, GH_list, color=PURPLE, lw=2.2, label="GH (ng/mL)")
    ax2.axhline(hp['GH_b'], color=PURPLE, ls="--", lw=1, alpha=0.6,
                label=f"GH_b = {hp['GH_b']} ng/mL")
    shade(ax)
    ax.set_ylabel("Cortisol (nmol/L)", fontsize=10, color=GREEN)
    ax2.set_ylabel("GH (ng/mL)",        fontsize=10, color=PURPLE)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right", ncol=3)

    # ── Panel 3 : Composantes Cortisol ──────────────────────────────────────
    ax = axes[2]
    ax.plot(t_arr, C_static_list,  color=GREEN,  lw=2,    label="SR_C statique  (G_plasma < G_th=70)")
    ax.plot(t_arr, C_dynamic_list, color=ORANGE, lw=2, ls="--", label="SR_C dynamique (−dG_plasma/dt)")
    ax.plot(t_arr, C_circ_list,    color=PURPLE, lw=2, ls="-.", label="SR_C circadienne [A=5, était 35]")
    total_C = np.array(C_static_list) + np.array(C_dynamic_list) + np.array(C_circ_list)
    ax.plot(t_arr, total_C,        color="black", lw=1.5, ls=":", alpha=0.8, label="SR_C totale")
    shade(ax)
    ax.set_ylabel("Sécrétion Cortisol\n(nmol/L/min)", fontsize=10)
    ax.legend(fontsize=7, loc="upper right", ncol=2)

    # ── Panel 4 : Composantes GH ────────────────────────────────────────────
    ax = axes[3]
    ax.plot(t_arr, GH_static_list,  color=GREEN,  lw=2,    label="SR_GH statique  (G_plasma < G_th=70)")
    ax.plot(t_arr, GH_circ_list,    color=CYAN,   lw=2, ls="-.", label="SR_GH circadienne")
    total_GH = np.array(GH_static_list) + np.array(GH_circ_list)
    ax.plot(t_arr, total_GH,        color="black", lw=1.5, ls=":", alpha=0.8, label="SR_GH totale")
    shade(ax)
    ax.set_ylabel("Sécrétion GH\n(ng/mL/min)", fontsize=10)
    ax.legend(fontsize=7, loc="upper right", ncol=2)

    # ── Panel 5 : dG_plasma/dt (diagnostic) ──────────────────────────────────
    dG_arr = np.diff(BGp_arr, prepend=BGp_arr[0])
    ax = axes[4]
    ax.plot(t_arr, dG_arr, color=GRAY, lw=1.5, label="dG_plasma/dt (mg/dL/min)")
    ax.axhline(0, color="black", lw=0.8)
    ax.fill_between(t_arr, dG_arr, 0, where=dG_arr < 0, alpha=0.25, color=RED,
                    label="Chute (active SR_dyn)")
    shade(ax)
    ax.set_ylabel("dG/dt\n(mg/dL/min)", fontsize=10)
    ax.set_xlabel("Temps", fontsize=10)
    ax.legend(fontsize=7, loc="upper right", ncol=2)

    # Axe X
    axes[4].set_xlim(0, 1440)
    tick_pos = list(range(0, 1441, 120))
    axes[4].set_xticks(tick_pos)
    axes[4].set_xticklabels([f"{m//60:02d}h" for m in tick_pos])

    plt.tight_layout()
    plt.savefig("child002_hypoglycemia_corrected.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── Stats ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  child#002 — Hypoglycémie corrigée")
    print(f"{'='*60}")
    print(f"  BG SC  : min={BG_arr.min():.1f}  max={BG_arr.max():.1f}  final={BG_arr[-1]:.1f} mg/dL")
    print(f"  BG PL  : min={BGp_arr.min():.1f}  max={BGp_arr.max():.1f}")
    print(f"  Durée BG_SC < 70 : {(BG_arr  < 70).sum()} min")
    print(f"  Durée BG_SC < 54 : {(BG_arr  < 54).sum()} min")
    print(f"  Cortisol max  : {max(C_list):.1f} nmol/L")
    print(f"  GH max        : {max(GH_list):.2f} ng/mL")
    print(f"  SR_C  : static_max={max(C_static_list):.3f}  dyn_max={max(C_dynamic_list):.4f}  circ_max={max(C_circ_list):.3f}")
    print(f"{'='*60}\n")