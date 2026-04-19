import logging
import sys
from loguru import logger

class InterceptHandler(logging.Handler):
    """
    Default handler from python logging to loguru.
    See: https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging
    """
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def setup_app_logging():
    """
    Configures loguru to intercept standard logging and use a custom educational format.
    """
    # Remove default handlers for both loguru and standard logging
    logger.remove()
    logging.getLogger().handlers = [InterceptHandler()]
    
    # Silence some noisy loggers
    logging.getLogger("uvicorn.access").handlers = [InterceptHandler()]
    logging.getLogger("uvicorn.error").handlers = [InterceptHandler()]
    logging.getLogger("fastapi").handlers = [InterceptHandler()]

    # Custom format with colored tags for education
    # extra['task'] is used to categorize the process
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[task]: <10}</cyan> | "
        "<level>{message}</level>"
    )

    # Add back the handler with the custom format
    logger.add(
        sys.stdout,
        format=log_format,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
    
    # Pre-populate with a default task value
    logger.configure(extra={"task": "CORE"})
    
    logger.info("Logging system initialized with Loguru.")
