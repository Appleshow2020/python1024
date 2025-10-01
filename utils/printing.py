import time
from enum import Enum

class LT(Enum):
    error = 'error'
    success = 'success'
    warning = 'warning'
    info = 'info'
    debug = 'debug'

class Colors:
    error = '\033[31m'
    success = '\033[32m'
    warning = '\033[33m'
    info = '\033[34m'
    debug = '\033[90m'
    reset = '\033[0m'

def printf(*text: object, **kwargs):
    ptype: LT | None = kwargs.get('ptype', LT.debug)
    end: str | None = kwargs.get('end', '\n')
    sep: str | None = kwargs.get('sep', ' ')
    useReset: bool | None = kwargs.get('useReset', True)

    def now2() -> str:
        return time.strftime("%X")

    color = getattr(Colors, ptype.value, Colors.reset)
    
    prefix = f"{color}[{now2()}] [{ptype.name}]"
    suffix = Colors.reset if useReset else ""

    print(prefix, *text, suffix, end=end, sep=sep)
