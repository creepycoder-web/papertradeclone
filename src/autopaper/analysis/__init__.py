"""
Analysis package for the V3-C runtime.

V3-C is the only entry strategy. The route selectors that used to be exported
here (Precision, Confirmed Fast Momentum, Early Fast Momentum, Recovery /
Reclaim) have been removed from the runtime and are not importable.
"""

from .decision import DecisionEngine
from .features import FeatureSet, build_features
from .regime import classify, is_continuation, is_fragile
from .routes import (
    RouteChoice,
    build_exit_plan,
    exit_style_for,
    select_route,
)

__all__ = [
    "DecisionEngine",
    "FeatureSet",
    "build_features",
    "classify",
    "is_fragile",
    "is_continuation",
    "RouteChoice",
    "select_route",
    "exit_style_for",
    "build_exit_plan",
]
