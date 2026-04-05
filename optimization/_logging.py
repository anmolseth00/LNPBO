"""Named logger 'lnpbo' with env-configurable level (LNPBO_LOG_LEVEL).

Usage:
    from LNPBO.optimization._logging import logger
"""

import logging
import os

logger = logging.getLogger("lnpbo")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(_handler)

_level = os.environ.get("LNPBO_LOG_LEVEL", "WARNING").upper()
logger.setLevel(getattr(logging, _level, logging.WARNING))
