from .base import Controller
from .base import Action
import logging

logger = logging.getLogger(__name__)
import pandas as pd
import pkg_resources

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'hormoglucose', 'params/vpatient_params.csv')
CONTROL_QUEST = pkg_resources.resource_filename(
    'hormoglucose', 'params/Quest.csv')

# Valeurs par defaut si le patient est inconnu (memes u2ss/BW que BBController)
DEFAULT_U2SS, DEFAULT_BW = 1.43, 57.0     # pmol/(L*kg), kg
DEFAULT_CR, DEFAULT_CF = 15.0, 50.0       # g/U, mg/dL/U


class PIDController(Controller):
    """
    PID sur le basal + bolus de repas inspire du BBController.

    Basal (U/min) :
        u_basal = basal_patient + P*e + I*int(e dt) + D*de/dt,  e = CGM - target
    Bolus (U, converti en U/min sur un pas) quand un repas est annonce :
        bolus = CHO/CR [+ (CGM > seuil) * (CGM - target)/CF  si correction=True]
    Avec correction=True (defaut), seuil=150 et target=140, le bolus est
    identique a celui du BBController.

    Une seule correction par repas (comme le BBController) : si le repas est
    reparti sur plusieurs pas consecutifs avec meal > 0, la part CHO/CR est
    donnee a chaque pas mais la correction uniquement au premier. Avec
    RandomScenario/CustomScenario, le repas annonce (info['meal']) n'apparait
    de toute facon que sur un seul pas : le patient l'ingere ensuite a
    EAT_RATE = 5 g/min, sans que le controleur le voie.

    Option `hold_after_meal` (min) : gele l'integrale apres un bolus, pour ne
    pas couvrir deux fois la hausse post-prandiale. Desactivee par defaut :
    sur 3 patients x 3 jours (seed 1), un gel de 180 min n'a rien apporte
    (TIR moyen 83.9 % avec gel contre 85.6 % sans).
    """

    def __init__(self, P=1e-4, I=1e-7, D=3e-4, target=140, basal=None,
                 i_max=None, meal_bolus=True, correction=True,
                 correction_threshold=150, hold_after_meal=0):
        self.P, self.I, self.D = P, I, D
        self.target = target
        self.basal_fixed = basal     # U/min ; None -> calcule depuis le patient
        self.i_max = i_max
        self.meal_bolus = meal_bolus
        self.correction = correction
        self.correction_threshold = correction_threshold
        self.hold_after_meal = hold_after_meal    # min
        self.patient_params = pd.read_csv(PATIENT_PARA_FILE)
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.reset()

    # ------------------------------------------------------------------
    def _load_patient(self, name):
        params = self.patient_params[self.patient_params.Name == name]
        quest = self.quest[self.quest.Name == name]
        if len(params) and len(quest):
            u2ss, BW = params.u2ss.item(), params.BW.item()
            self.CR, self.CF = quest.CR.item(), quest.CF.item()
        else:
            logger.warning('Patient %r inconnu : parametres par defaut '
                           '(u2ss=%.2f, BW=%.0f, CR=%.0f, CF=%.0f)', name,
                           DEFAULT_U2SS, DEFAULT_BW, DEFAULT_CR, DEFAULT_CF)
            u2ss, BW = DEFAULT_U2SS, DEFAULT_BW
            self.CR, self.CF = DEFAULT_CR, DEFAULT_CF
        self.basal = (self.basal_fixed if self.basal_fixed is not None
                      else u2ss * BW / 6000)       # U/min

    def _bolus(self, meal, bg, dt):
        """Bolus en U/min (delivre sur un pas de dt minutes)."""
        meal_on = meal is not None and meal > 0
        first_step = meal_on and not self._prev_meal_on   # debut du repas
        self._prev_meal_on = meal_on
        if not self.meal_bolus or not meal_on:
            return 0.0
        cho = meal * dt                            # g ; meal est en g/min
        bolus = cho / self.CR                      # U
        if (self.correction and first_step
                and bg > self.correction_threshold):
            bolus += (bg - self.target) / self.CF  # U
        logger.info('Bolus repas : %.0f g CHO, BG=%.0f -> %.2f U', cho, bg, bolus)
        self.time_since_bolus = 0.0
        return bolus / dt                          # U/min

    # ------------------------------------------------------------------
    def policy(self, observation, reward, done, **kwargs):
        dt = kwargs.get('sample_time', 5)
        bg = observation.CGM
        meal = kwargs.get('meal', 0)               # g/min

        if self.basal is None:
            self._load_patient(kwargs.get('patient_name'))
        if self.prev_state is None:
            self.prev_state = bg

        bolus = self._bolus(meal, bg, dt)

        err = bg - self.target
        deriv = (bg - self.prev_state) / dt
        u_unsat = (self.basal + self.P * err +
                   self.I * self.integrated_state + self.D * deriv)
        u = max(0.0, u_unsat)

        in_hold = self.time_since_bolus < self.hold_after_meal
        # anti-windup conditionnel + gel post-prandial
        if not in_hold and (u_unsat > 0 or err > 0):
            self.integrated_state += err * dt
            if self.i_max is not None:
                self.integrated_state = max(-self.i_max,
                                            min(self.i_max, self.integrated_state))

        self.time_since_bolus += dt
        self.prev_state = bg
        return Action(basal=u, bolus=bolus)

    def reset(self):
        self.integrated_state = 0
        self.prev_state = None
        self.basal = None             # recalcule (avec CR, CF) au premier pas
        self.CR = self.CF = None
        self.time_since_bolus = float('inf')
        self._prev_meal_on = False    # le pas precedent avait-il meal > 0 ?
