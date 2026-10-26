"""Process-level helpers for batch jobs.
"""

import sys


def register_clean_exit_handlers():
    """
    Permite que el proceso salga limpiamente ante SIGTERM/SIGINT.
    """
    import signal
    import sys

    try:
        signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
        signal.signal(signal.SIGINT,  lambda s, f: sys.exit(0))
    except ValueError:
        # Puede pasar si se llama desde un thread que no es el principal.
        pass
