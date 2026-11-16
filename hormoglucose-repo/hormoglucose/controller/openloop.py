from .base import Controller
from .base import Action
import numpy as np
import pandas as pd
import pkg_resources
import logging

logger = logging.getLogger(__name__)
CONTROL_QUEST = pkg_resources.resource_filename('hormoglucose',
                                                'params/Quest.csv')
PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'hormoglucose', 'params/vpatient_params.csv')


class OpenLoopController(Controller):
    def __init__(self, basal_rate: float = 0.01):
        self.basal_rate = basal_rate

    def reset(self):
        pass

    def policy(self, observation, reward, done, **kwargs):
        return Action(basal=self.basal_rate, bolus=0.0)