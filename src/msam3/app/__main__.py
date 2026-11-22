"""``python -m msam3.app``: the app on its own, or for one session of the Metashape launcher."""

import argparse
from pathlib import Path

from msam3.app.server import serve


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m msam3.app")
    parser.add_argument("--session", type=Path, help="Session folder written by the launcher")
    parser.add_argument("--no-browser", action="store_true", help="Only print and write the URL")
    args = parser.parse_args()
    session = args.session.resolve() if args.session else None
    return serve(session, open_browser=not args.no_browser)


if __name__ == "__main__":
    raise SystemExit(main())
