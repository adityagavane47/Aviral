"""
download_rife_weights.py — Project Aviral
==========================================
Downloads the official RIFE HD pre-trained weights from Google Drive
and places them at models/train_log/flownet.pkl

Usage:
    python download_rife_weights.py
"""

import os
import sys
import urllib.request

TRAIN_LOG_DIR = os.path.join("models", "train_log")
WEIGHT_FILE   = os.path.join(TRAIN_LOG_DIR, "flownet.pkl")

# Official RIFE HD weights — direct download from Google Drive
# File ID from: https://drive.google.com/file/d/1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_/
GDRIVE_FILE_ID  = "1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_"
CONFIRM_URL     = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}&confirm=t"

os.makedirs(TRAIN_LOG_DIR, exist_ok=True)

def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = "#" * int(pct / 2)
        sys.stdout.write(f"\r  [{bar:<50}] {pct:.1f}%  ({downloaded//1024//1024}MB / {total_size//1024//1024}MB)")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\r  Downloaded: {downloaded // 1024 // 1024} MB...")
        sys.stdout.flush()

if os.path.exists(WEIGHT_FILE):
    size_mb = os.path.getsize(WEIGHT_FILE) / 1024 / 1024
    print(f"[OK] Weights already exist at {WEIGHT_FILE} ({size_mb:.1f} MB)")
    print("     Nothing to do. You can now set USE_RIFE = True in ml_engine.py")
    sys.exit(0)

print(f"Downloading RIFE HD weights to {WEIGHT_FILE}...")
print(f"Source: Google Drive (File ID: {GDRIVE_FILE_ID})")
print()

try:
    # Google Drive large file requires a special confirm token
    req = urllib.request.Request(
        CONFIRM_URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req) as response:
        data = response.read()

    dest_path = WEIGHT_FILE
    with open(dest_path, "wb") as f:
        f.write(data)

    size_mb = os.path.getsize(dest_path) / 1024 / 1024
    print(f"\n[OK] Download complete! {dest_path} ({size_mb:.1f} MB)")

except Exception as e:
    print(f"\n[ERROR] Download failed: {e}")
    print()
    print("  Manual download instructions:")
    print("  1. Open this URL in your browser:")
    print(f"     https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view")
    print("  2. Click the Download button")
    print("  3. Unzip the downloaded file")
    print(f"  4. Move 'flownet.pkl' to: d:\\Aviral\\models\\train_log\\flownet.pkl")
    sys.exit(1)

print()
print("=" * 60)
print("  NEXT STEP:")
print("  In ml_engine.py, change line 49 to:")
print('      USE_RIFE = True')
print("=" * 60)
