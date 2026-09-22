"""Framework-specific exceptions with actionable user messages."""


class FineTuneLabError(RuntimeError):
    """Base exception for expected FineTuneLab failures."""


class ConfigurationError(FineTuneLabError):
    """Raised when a configuration requests an unsupported combination."""


class DataValidationError(FineTuneLabError):
    """Raised when a dataset does not match the selected recipe schema."""


class DependencyError(FineTuneLabError):
    """Raised when an optional dependency required by a feature is missing."""
