class MlServiceError(RuntimeError):
    """Raised when a service cannot complete a prediction."""


class MlValidationError(MlServiceError):
    """Raised when a service receives invalid business data."""
