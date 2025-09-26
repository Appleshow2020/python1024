import time
from enum import Enum

class LT(Enum):
    error = 'error'
    success = 'success'
    warning = 'warning'
    info = 'info'
    debug = 'debug'



def printf(*text:object,
           ptype:LT|None = LT.debug,
           end:str|None = '\n',
           sep:str|None=' ',
           useReset:bool|None = True):
    """
    type: error, warning, info, debug, success
    """
    
    class Colors:
        error = '\033[31m'
        success = '\033[32m'
        warning = '\033[33m'
        info = '\033[34m'
        debug = '\033[90m'
        reset = '\033[0m'
        
    def now2() -> str:
        return time.strftime("%X")
    
    color = getattr(Colors, ptype.value, Colors.reset)
    if useReset:
        print(f"{color}[{now2()}]", f"[{ptype.name}]", text, Colors.reset, end=end, sep=sep)
    else:
        print(f"{color}[{now2()}]", f"[{ptype.name}]", end=end, sep=sep)