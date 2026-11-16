import logging
from collections import namedtuple
from datetime import datetime
from datetime import timedelta
from hormoglucose.patient.t1dpatientPA import Action

logger = logging.getLogger(__name__)


class Scenario(object):
    def __init__(self, start_time):
        self.start_time = start_time

    def get_action(self, t):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class CustomScenarioPA(Scenario):
    def __init__(self, start_time, scenario):
        '''
        scenario - list of tuples (time, duration, factor)
                   - time: float (hours) or timedelta or datetime
                   - duration: minutes (float or int)
                   - factor: HR multiplicateur (1.0-2.5)
        '''
        Scenario.__init__(self, start_time=start_time)
        self.scenario = scenario

    def get_action(self, t):

        if not self.scenario:
            return Action(CHO=0, insulin=0, HR=1.0)
        
        t_hours = (t - self.start_time).total_seconds() / 3600.0

        #Pour gérer sur plusieurs jours 
        t_day = t_hours % 24

        for time, dur, factor in self.scenario:
            start_hours = parseTimeToHours(time, self.start_time)
            if start_hours <= t_day < start_hours + dur / 60.0:
                return Action(CHO=0, insulin=0, HR=factor)
            
        return Action(CHO=0, insulin=0, HR=1.0)

    def reset(self):
        pass


def parseTimeToHours(time, start_time):
    """Convert input time to hours since start_time."""
    if isinstance(time, (int, float)):
        return float(time)
    elif isinstance(time, timedelta):
        return time.total_seconds() / 3600.0
    elif isinstance(time, datetime):
        return (time - start_time).total_seconds() / 3600.0
    else:
        raise ValueError("Expect time to be int, float, timedelta, or datetime")
