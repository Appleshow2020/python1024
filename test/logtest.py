import time
from enum import Enum

class Colors:
    error = '\033[31m'
    success = '\033[32m'
    warning = '\033[33m'
    info = '\033[34m'
    debug = '\033[90m'
    reset = '\033[0m'

class LT(Enum):
    error = 'error'
    success = 'success'
    warning = 'warning'
    info = 'info'
    debug = 'debug'

def now2() -> str:
    return time.strftime("%X")

def printf(text:str, ptype:LT):
    """
    type: error, warning, info, debug, success
    """
    color = getattr(Colors, ptype.value, Colors.reset)
    print(color, f"[{now2()}]", f"[{ptype.name}]", text, Colors.reset)

printf("testmessage1",LT.error)
printf("testmessage2",LT.warning)
printf("testmessage3",LT.info)
printf("testmessage4",LT.debug)
printf("testmessage5",LT.success)