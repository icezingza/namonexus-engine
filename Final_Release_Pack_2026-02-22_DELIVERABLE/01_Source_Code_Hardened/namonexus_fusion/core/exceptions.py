"""
Custom exceptions for NamoNexus Fusion Engine.
Patent-Pending Technology | NamoNexus Research Team
"""


class NamoNexusError(Exception):
    """Base exception for all NamoNexus errors."""


class ConfigurationError(NamoNexusError):
    """Raised when engine configuration is invalid."""


class InvalidObservationError(NamoNexusError):
    """Raised when an observation has invalid score or confidence."""


class PriorLearningError(NamoNexusError):
    """Raised when prior learning fails to converge or has insufficient data."""


class SensorBlacklistError(NamoNexusError):
    """Raised when an observation is submitted from a blacklisted sensor."""


class OptimizerError(NamoNexusError):
    """Raised on hyperparameter optimization failure."""
