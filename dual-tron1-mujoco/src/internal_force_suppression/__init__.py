"""
Internal Force Suppression Module (IFSM)

A modular system for suppressing internal forces in dual-robot cooperative transportation.
Implements "Implicit Force Perception + Residual Admittance Control" strategy.

Architecture:
    - Force Estimator: Sensorless force estimation using momentum observer
    - Internal Force Analyzer: Decompose forces into effective and internal components
    - Admittance Controller: Residual admittance control for compliance

References:
    - De Luca, A., & Mattone, R. (2005). "Sensorless Robot Collision Detection and
      Hybrid Force/Motion Control." ICRA 2005.
    - Erhart, S., & Hirche, S. (2015). "Internal Force Analysis and Load Distribution
      for Cooperative Multi-Robot Manipulation." ICRA 2015.
    - Hogan, N. (1985). "Impedance Control: An Approach to Manipulation."

Author: [Your Name]
Date: 2026-08-03
"""

__version__ = "0.1.0"

from .core.force_estimator import ImplicitForceEstimator
from .core.internal_force_analyzer import InternalForceAnalyzer
from .core.admittance_controller import ResidualAdmittanceController

__all__ = [
    "ImplicitForceEstimator",
    "InternalForceAnalyzer",
    "ResidualAdmittanceController",
]
