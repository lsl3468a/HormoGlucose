"""HormoGlucose: an extension of SimGlucose with cortisol, growth hormone
and exercise dynamics for type 1 diabetes simulation."""
from gymnasium.envs.registration import register

__version__ = "1.0.0"

register(
    id='hormoglucose-v0',
    entry_point='hormoglucose.envs:T1DSimEnv',
)
