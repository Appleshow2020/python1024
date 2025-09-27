# main/app.py
import sys
from main.ApplicationController import ApplicationController
from classes.printing import printf, LT

def main():
    """메인 함수 - 애플리케이션 진입점"""
    printf("Starting Ball Tracking Application...", ptype=LT.info)
    
    app = ApplicationController()
    exit_code = 0
    
    try:
        success = app.run()
        exit_code = 0 if success else 1
        
    except Exception as e:
        printf(f"Unexpected error in main: {e}", ptype=LT.error)
        exit_code = 1
        
    finally:
        app.cleanup()
        printf("Application terminated", ptype=LT.info)
    
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)