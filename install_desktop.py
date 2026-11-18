"""Add a "Badge Scanner" icon to the app menu and desktop.

Run it with the Python that has the scanner's packages, e.g.
    ~/miniconda3/envs/badge/bin/python install_desktop.py
(on the Raspberry Pi: python3 install_desktop.py). Re-run after moving the folder.
"""

import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ICON = os.path.join(HERE, "assets", "badge_scanner.png")


def make_icon(path, size=256):
    img = np.zeros((size, size, 4), np.uint8)
    # rounded blue tile
    tile = np.zeros((size, size), np.uint8)
    r = 48
    cv2.rectangle(tile, (r, 0), (size - r, size), 255, -1)
    cv2.rectangle(tile, (0, r), (size, size - r), 255, -1)
    for c in [(r, r), (size - r, r), (r, size - r), (size - r, size - r)]:
        cv2.circle(tile, c, r, 255, -1)
    img[tile > 0] = (255, 154, 76, 255)                      # BGR #4c9aff
    # white badge card with lanyard clip
    cv2.rectangle(img, (68, 52), (188, 214), (255, 255, 255, 255), -1)
    cv2.rectangle(img, (108, 36), (148, 60), (40, 40, 40, 255), -1)
    cv2.rectangle(img, (68, 176), (188, 214), (67, 159, 255, 255), -1)   # orange band
    # QR-like pattern
    rng = np.random.default_rng(7)
    x0, y0, cell = 92, 78, 8
    for i in range(9):
        for j in range(9):
            finder = (i < 3 and j < 3) or (i < 3 and j > 5) or (i > 5 and j < 3)
            on = (finder and not (i % 6 == 1 and j % 6 == 1)) or (not finder and rng.random() < 0.5)
            if on:
                cv2.rectangle(img, (x0 + j * cell, y0 + i * cell), (x0 + j * cell + cell - 1, y0 + i * cell + cell - 1),
                              (30, 30, 30, 255), -1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def main():
    make_icon(ICON)
    entry = f"""[Desktop Entry]
Type=Application
Name=Badge Scanner
Comment=Identify badges and scan their QR codes with the RealSense camera
Exec="{sys.executable}" "{os.path.join(HERE, 'app.py')}"
Path={HERE}
Icon={ICON}
Terminal=false
Categories=Utility;
StartupWMClass=BadgeScanner
"""
    apps = os.path.expanduser("~/.local/share/applications")
    os.makedirs(apps, exist_ok=True)
    menu_file = os.path.join(apps, "badge-scanner.desktop")
    with open(menu_file, "w") as f:
        f.write(entry)
    os.chmod(menu_file, 0o755)
    print("App menu entry:", menu_file)

    desktop_dir = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True).stdout.strip() \
        if shutil.which("xdg-user-dir") else os.path.expanduser("~/Desktop")
    if desktop_dir and os.path.isdir(desktop_dir):
        dfile = os.path.join(desktop_dir, "badge-scanner.desktop")
        shutil.copy(menu_file, dfile)
        os.chmod(dfile, 0o755)
        if shutil.which("gio"):  # lets GNOME launch it without the "untrusted" prompt
            subprocess.run(["gio", "set", dfile, "metadata::trusted", "true"], stderr=subprocess.DEVNULL)
        print("Desktop icon:", dfile)


if __name__ == "__main__":
    main()
