"""venv/bin/python -m airpoint.lab        practice pages and the pointing test, on this machine only."""
import argparse
import os
import webbrowser

from .server import PORT, make_server

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    ap = argparse.ArgumentParser(description="AirPoint lab: tutorial and pointing test")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-open", action="store_true", help="do not open the browser")
    args = ap.parse_args(argv)
    httpd = make_server(ROOT, args.port)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"[AirPoint lab] {url}   results -> results/trials.jsonl   Ctrl+C stops")
    if not args.no_open:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
