"""Structured logging for Phase 1 pipeline."""
import logging
from pathlib import Path
from typing import Dict, Any


class StructuredLogger:
    """Custom logger with structured logging and decision tracking."""
    
    def __init__(self, name: str, log_file: str = None, log_level: str = "INFO"):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            log_file: Path to log file
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level))
        
        # Remove any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        
        # Formatter
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if log_file specified)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, log_level))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str, **context) -> None:
        """Log info level message with optional context."""
        if context:
            message = f"{message} | {context}"
        self.logger.info(message)
    
    def warning(self, message: str, **context) -> None:
        """Log warning level message with optional context."""
        if context:
            message = f"{message} | {context}"
        self.logger.warning(message)
    
    def error(self, message: str, **context) -> None:
        """Log error level message with optional context."""
        if context:
            message = f"{message} | {context}"
        self.logger.error(message)
    
    def debug(self, message: str, **context) -> None:
        """Log debug level message with optional context."""
        if context:
            message = f"{message} | {context}"
        self.logger.debug(message)
    
    def log_decision(self, decision: str, reason: str, **metadata) -> None:
        """
        Log a pipeline decision with reason and metadata.
        
        Args:
            decision: Description of decision
            reason: Reason for decision
            **metadata: Additional context (counts, values, etc.)
        """
        context_str = " | ".join([f"{k}={v}" for k, v in metadata.items()])
        if context_str:
            self.logger.info(f"DECISION: {decision} | reason={reason} | {context_str}")
        else:
            self.logger.info(f"DECISION: {decision} | reason={reason}")


class Metrics:
    """Track pipeline statistics and metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics: Dict[str, Any] = {}
    
    def increment(self, key: str, value: int = 1) -> None:
        """Increment a metric counter."""
        if key not in self.metrics:
            self.metrics[key] = 0
        self.metrics[key] += value
    
    def set(self, key: str, value: Any) -> None:
        """Set a metric value."""
        self.metrics[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a metric value."""
        return self.metrics.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return self.metrics.copy()
    
    def report(self) -> str:
        """Generate metrics report."""
        lines = ["=== Metrics Summary ==="]
        for key, value in sorted(self.metrics.items()):
            lines.append(f"{key}: {value}")
        return "\n".join(lines)


def setup_logger(config: Dict[str, Any]) -> StructuredLogger:
    """
    Setup logger from configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Configured StructuredLogger instance
    """
    return StructuredLogger(
        name="phase1",
        log_file=config.get("LOG_FILE"),
        log_level=config.get("LOG_LEVEL", "INFO")
    )
