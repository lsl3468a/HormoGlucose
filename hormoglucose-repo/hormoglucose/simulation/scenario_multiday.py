"""
scenario_multiday.py
--------------------
Scénarios multi-jours pour repas et activité physique.
Les jours non définis → aucun repas / activité (HR factor = 1.0).
"""

from collections import namedtuple

MealAction     = namedtuple("MealAction",     ["meal"])
ExerciseAction = namedtuple("ExerciseAction", ["HR"])


def _hour_to_minutes(hour_float: float) -> int:
    return int(round(hour_float * 60))


class MultiDayCustomScenario:
    """
    Scénario de repas multi-jours.

    Parameters
    ----------
    start_time      : datetime
    daily_scenarios : dict[int, list[tuple(hour_float, grams)]]
        {0: [(7.0, 45), (12.0, 70)], 2: [(8.0, 30)], ...}
        Jours absents → aucun repas.
    """

    def __init__(self, start_time, daily_scenarios: dict):
        self.start_time      = start_time
        self.daily_scenarios = daily_scenarios

    def get_action(self, t):
        elapsed       = t - self.start_time
        total_minutes = int(elapsed.total_seconds() / 60)
        day_idx       = total_minutes // 1440
        minute_in_day = total_minutes % 1440

        cho = 0.0
        for hour_float, grams in self.daily_scenarios.get(day_idx, []):
            if _hour_to_minutes(hour_float) == minute_in_day:
                cho = grams
                break

        return MealAction(meal=cho)

    def reset(self):
        pass


class MultiDayCustomScenarioPA:
    """
    Scénario d'activité physique multi-jours.

    Parameters
    ----------
    start_time      : datetime
    daily_scenarios : dict[int, list[tuple(hour_float, duration_min, hr_factor)]]
        {0: [(8.0, 30, 1.5)], 1: [(17.0, 60, 2.0)], ...}
        Jours absents → HR factor = 1.0 (repos).
    """

    def __init__(self, start_time, daily_scenarios: dict):
        self.start_time      = start_time
        self.daily_scenarios = daily_scenarios

    def get_action(self, t):
        elapsed       = t - self.start_time
        total_minutes = int(elapsed.total_seconds() / 60)
        day_idx       = total_minutes // 1440
        minute_in_day = total_minutes % 1440

        hr_factor = 1.0
        for hour_float, duration_min, factor in self.daily_scenarios.get(day_idx, []):
            start_min = _hour_to_minutes(hour_float)
            if start_min <= minute_in_day < start_min + int(duration_min):
                hr_factor = factor
                break

        return ExerciseAction(HR=hr_factor)

    def reset(self):
        pass