import sys

from main.ApplicationController import ApplicationController
from utils.printing import printf, LT

def _main_impl() -> int:
    app = ApplicationController()
    exit_code = 0

    printf(
        "Starting Ball Tracking Application...",
        ptype=LT.info
    )

    try:
        result = app.run()

        if not result:
            exit_code = 1

    except Exception as exc:
        printf(
            "Unexpected error in main:",
            exc,
            ptype=LT.error
        )
        exit_code = 1

    finally:
        try:
            app.cleanup()
        except Exception as exc:
            printf(
                "Cleanup failed:",
                exc,
                ptype=LT.error
            )
            exit_code = 1

        printf(
            "Application terminated",
            ptype=LT.info
        )

    return exit_code

def main() -> int:
    return _main_impl()


if __name__ == "__main__":
    sys.exit(main())