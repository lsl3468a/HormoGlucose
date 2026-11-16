from hormoglucose.simulation.scenario import Scenario
from hormoglucose.patient.t1dpatientPA import Action
import numpy as np
from scipy.stats import truncnorm
from datetime import datetime, timedelta
import logging
import random

logger = logging.getLogger(__name__)


class RandomScenarioPA(Scenario):
    def __init__(self, start_time, seed=None):
        self.start_time = start_time
        self.seed = seed
        self.reset()


    def get_action(self, t):
        # t doit être un datetime.datetime
        delta_t = t - datetime.combine(t.date(), datetime.min.time())
        t_sec = delta_t.total_seconds()

        # Création du scénario en début de journée
        if t_sec < 1:
            logger.info('Creating new exercise scenario for the day...')
            self.scenario = self.create_scenario()

        t_min = int(np.floor(t_sec / 60.0))
        
        # Vérification si on est dans une période d’exercice
        for start, dur, factor in zip(self.scenario['exercise']['time'],
                                      self.scenario['exercise']['duration'],
                                      self.scenario['exercise']['factor']):
            if start <= t_min < start + dur:
                return Action(CHO=0, insulin=0, HR=factor)

        # Sinon pas d’exercice → HR neutre (1.0)
        return Action(CHO=0, insulin=0, HR=1.0)

    
    
    def create_scenario(self):
        scenario = {'exercise': {'time': [], 'duration': [], 'factor': []}}

        n_sessions = self.random_gen.randint(2, 4)  # 2 à 3 séances par jour
        possible_starts = np.linspace(6*60, 21*60, 15)  # entre 6h et 21h

        starts = self.random_gen.choice(possible_starts, n_sessions, replace=False)
        starts.sort()

        for start in starts:
            dur = int(self.random_gen.choice([15, 30, 45, 60]))
            factor = float(self.random_gen.uniform(1.2, 2.5))
            scenario['exercise']['time'].append(start)
            scenario['exercise']['duration'].append(dur)
            scenario['exercise']['factor'].append(factor)

        return scenario

    
    def reset(self):
        self.random_gen = np.random.RandomState(self.seed)
        self.scenario = self.create_scenario()

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.reset()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import copy

    now = datetime.now()
    t0 = datetime.combine(now.date(), datetime.min.time())  # début journée (00h00)
    t = copy.deepcopy(t0)
    sim_time = timedelta(days=1)

    scenario = RandomScenarioPA(start_time=t0, seed=1)
    HR_vals = []
    T = []

    while t < t0 + sim_time:
        action = scenario.get_action(t)
        HR_vals.append(action.HR)
        T.append(t)
        t += timedelta(minutes=1)

    plt.plot(T, HR_vals, label="HR factor")
    ax = plt.gca()
    ax.xaxis.set_minor_locator(mdates.AutoDateLocator())
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))
    plt.ylabel("HR factor")
    plt.title("Random Exercise Scenario")
    plt.legend()
    plt.show()