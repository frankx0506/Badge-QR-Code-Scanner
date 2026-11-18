"""Badge Scanner Lab as a desktop app.

Starts the scanner and opens it in its own window (Chromium / Chrome "app
mode": no address bar, no tabs). Closing the window stops the scanner and
releases the camera. Works offline; on a Raspberry Pi it uses Chromium,
which ships with Raspberry Pi OS.

    python app.py                 # live D435i
    python app.py --replay DIR    # replay saved photos
    python app.py --fullscreen    # kiosk / full screen (e.g. on the robot)
"""

import os
import shutil
import subprocess
import sys
import threading
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)                      # so models/ and badges.json resolve when started from the icon
sys.path.insert(0, HERE)

# Started from the icon there is no terminal: keep the app's messages in logs/app.log, and let
# `kill -USR1 <pid>` write every thread's current position there (for diagnosing a freeze).
if not sys.stdout or not sys.stdout.isatty():
    import faulthandler
    import signal
    import time as _time
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    _log = open(os.path.join(HERE, "logs", "app.log"), "a", buffering=1)
    _log.write(f"\n===== app start {_time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    sys.stdout = sys.stderr = _log
    faulthandler.enable(file=_log)
    faulthandler.register(signal.SIGUSR1, file=_log, all_threads=True)

import web_ui  # noqa: E402

BROWSERS = ["chromium", "chromium-browser", "google-chrome", "google-chrome-stable",
            "microsoft-edge", "brave-browser"]


def find_browser():
    for name in BROWSERS:
        path = shutil.which(name)
        if path:
            return path
    return None


def error_window(msg):
    """Show a startup error even when launched from the desktop icon."""
    print(msg, file=sys.stderr)
    for cmd in (["zenity", "--error", "--title=Badge Scanner", f"--text={msg}"],
                ["notify-send", "Badge Scanner", msg]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd)
            return


def main():
    ap = web_ui.build_parser()
    ap.add_argument("--fullscreen", action="store_true", help="kiosk mode (full screen)")
    args = ap.parse_args()

    try:
        srv, src, url = web_ui.start(args)
    except RuntimeError as e:
        hint = ("\n\nThe camera is probably in use by another scanner window - close it and try again."
                if "busy" in str(e).lower() else "\n\nIs the D435i plugged in?")
        error_window(f"Could not open the camera:\n{e}{hint}")
        sys.exit(1)

    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"Badge Scanner running at {url}", flush=True)

    browser = find_browser()
    try:
        if browser:
            profile = os.path.expanduser("~/.cache/badge-scanner-app")
            cmd = [browser, f"--app={url}", f"--user-data-dir={profile}", "--class=BadgeScanner",
                   "--no-first-run", "--no-default-browser-check", "--disable-translate",
                   "--window-size=1600,950", "--password-store=basic"]
            if args.fullscreen:
                cmd.append("--kiosk")
            # A separate profile makes this its own browser process, so this call
            # returns only when the app window is closed.
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print("Chromium/Chrome not found - opening in the default browser. Press Ctrl-C to quit.")
            webbrowser.open(url)
            threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        src.close()
        print("Badge Scanner closed.", flush=True)


if __name__ == "__main__":
    main()
