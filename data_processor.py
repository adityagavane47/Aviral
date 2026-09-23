"""
data_processor.py — Project Aviral
====================================
Loads two INSAT satellite NetCDF files (t=0 and t=30 min),
extracts the Thermal IR channel, and normalizes them into
3-channel RGB-like tensors compatible with standard PyTorch
vision models (e.g., RIFE).

HOW TO USE WITH REAL DATA:
---------------------------
1. Place your real INSAT .nc files at:
       data/insat_t0.nc   (timestamp t=00 min)
       data/insat_t30.nc  (timestamp t=30 min)

2. Run the inspector to see what's inside your file:
       python inspect_nc.py data/insat_t0.nc

3. Set VARIABLE_NAME below to the correct variable key.
   Use "AUTO" to let the script detect it automatically.

4. In app.py sidebar, toggle "Use Synthetic Data" OFF and
   click the Run button.

Common INSAT-3D variable names:
   "TIR1"  — Thermal IR 1 (~10.8 um)   ← most common
   "TIR2"  — Thermal IR 2 (~12.0 um)
   "WV"    — Water Vapour  (~6.7 um)
   "VIS"   — Visible       (~0.65 um)
   "SWIR"  — Short-wave IR (~1.6 um)
   "MIR"   — Mid IR        (~3.9 um)
   Set to "AUTO" to auto-detect the best 2D variable.
"""

import os
import numpy as np
import xarray as xr
import torch
import cv2
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION  ← Edit these to match your actual .nc files
# ─────────────────────────────────────────────────────────────
DATA_DIR = "data"
T0_FILE  = os.path.join(DATA_DIR, "insat_t0.nc")
T30_FILE = os.path.join(DATA_DIR, "insat_t30.nc")

# The NetCDF variable name for the Thermal IR channel.
# Set to "AUTO" to auto-detect the best 2D image variable.
# Common INSAT-3D names: "TIR1", "TIR2", "WV", "VIS", "SWIR", "MIR"
VARIABLE_NAME = "AUTO"

# Preferred variable priority order for AUTO detection
AUTO_PRIORITY = ["CMI", "TIR1", "TIR2", "WV", "MIR", "SWIR", "VIS",
                  "IMG_TIR1", "IMG_TIR2", "BT", "brightness_temperature"]

# Target spatial resolution (must be divisible by 32 for RIFE)
TARGET_H = 512
TARGET_W = 512

# ─────────────────────────────────────────────────────────────
#  DUMMY DATA GENERATOR  (used when real .nc files are absent)
# ─────────────────────────────────────────────────────────────
def _create_dummy_nc(filepath: str, seed: int = 0) -> None:
    """
    Creates a synthetic NetCDF file to simulate INSAT-3D TIR data.
    Brightness temperatures typically range from 200–310 K.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    rng = np.random.default_rng(seed)

    # Simulate a 512x512 scene with a realistic cold-cloud pattern
    lat = np.linspace(8.0, 37.0, TARGET_H)
    lon = np.linspace(68.0, 97.0, TARGET_W)

    # Base warm ocean/land background (~295 K)
    base = rng.normal(295, 5, (TARGET_H, TARGET_W)).astype(np.float32)

    # Add cold cloud clusters (~220–250 K)
    for _ in range(8):
        cy = rng.integers(50, TARGET_H - 50)
        cx = rng.integers(50, TARGET_W - 50)
        r  = rng.integers(30, 80)
        yy, xx = np.ogrid[:TARGET_H, :TARGET_W]
        mask = (yy - cy)**2 + (xx - cx)**2 < r**2
        base[mask] = rng.uniform(215, 250)

    # Slight temporal drift for t30 (seed changes)
    ds = xr.Dataset(
        {VARIABLE_NAME: (["lat", "lon"], base)},
        coords={"lat": lat, "lon": lon}
    )
    ds.attrs["description"] = f"Dummy INSAT-3D TIR1 — seed={seed}"
    ds.to_netcdf(filepath)
    print(f"  [Dummy] Created: {filepath}")


# ─────────────────────────────────────────────────────────────
#  AUTO-DETECTION HELPER
# ─────────────────────────────────────────────────────────────
def _auto_detect_variable(ds: "xr.Dataset", filepath: str) -> str:
    """
    Automatically finds the best 2D image variable in a NetCDF dataset.

    Priority order: AUTO_PRIORITY list first, then any 2D variable
    with both spatial dimensions > 50 pixels.
    """
    available = list(ds.data_vars)

    # 1. Try the priority list first
    for candidate in AUTO_PRIORITY:
        if candidate in available:
            arr = ds[candidate].values.squeeze()
            if arr.ndim == 2 and arr.shape[0] > 50 and arr.shape[1] > 50:
                print(f"  [AUTO] Detected variable: '{candidate}' "
                      f"(shape after squeeze: {arr.shape})")
                return candidate

    # 2. Fall back: any variable that squeezes to 2D image
    for vname in available:
        arr = ds[vname].values.squeeze()
        if arr.ndim == 2 and arr.shape[0] > 50 and arr.shape[1] > 50:
            print(f"  [AUTO] Detected variable: '{vname}' "
                  f"(shape after squeeze: {arr.shape})")
            return vname

    raise ValueError(
        f"Could not auto-detect a 2D image variable in {filepath}.\n"
        f"  Available variables: {available}\n"
        f"  Run: python inspect_nc.py {filepath}\n"
        f"  Then set VARIABLE_NAME explicitly in data_processor.py."
    )


# ─────────────────────────────────────────────────────────────
#  CORE LOADING & NORMALIZATION
# ─────────────────────────────────────────────────────────────
def load_nc_channel(filepath: str, variable: str) -> np.ndarray:
    """
    Opens a NetCDF file and extracts a calibrated 2D float32 array.

    Handles real INSAT-3D data quirks:
      - Auto-detection when variable = "AUTO"
      - _FillValue / missing_value masking  (set to NaN)
      - scale_factor / add_offset calibration (L1B packing)
      - Squeezes extra time / level / scan dimensions

    Returns:
        np.ndarray of shape (H, W), dtype float32
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"NetCDF file not found: {filepath}\n"
            f"  Place your INSAT .nc file there, or toggle 'Use Synthetic Data'"
            f" ON in the Streamlit sidebar."
        )

    # mask_and_scale=False lets us handle fill/scale manually for robustness
    ds = xr.open_dataset(filepath, mask_and_scale=False)

    # Resolve variable name
    resolved = variable
    if variable == "AUTO":
        resolved = _auto_detect_variable(ds, filepath)

    if resolved not in ds.data_vars:
        available = list(ds.data_vars)
        raise KeyError(
            f"Variable '{resolved}' not found in {filepath}.\n"
            f"  Available: {available}\n"
            f"  Run: python inspect_nc.py {filepath}\n"
            f"  Then update VARIABLE_NAME in data_processor.py."
        )

    var_obj = ds[resolved]
    data = var_obj.values.astype(np.float32)

    # ── Fill value masking ──────────────────────────────────────
    fill_val = var_obj.attrs.get("_FillValue",
               var_obj.attrs.get("missing_value", None))
    if fill_val is not None:
        fill_val = float(fill_val)
        # Replace fill values with NaN so they don't skew normalization
        data[data == fill_val] = np.nan
        # Also catch common sentinel values like -9999, 65535
        data[data < -1000] = np.nan
        data[data > 65000] = np.nan

    # ── Scale / offset calibration (INSAT L1B packing) ─────────
    scale  = float(var_obj.attrs.get("scale_factor",  1.0))
    offset = float(var_obj.attrs.get("add_offset",    0.0))
    if scale != 1.0 or offset != 0.0:
        data = data * scale + offset
        print(f"  [Calibration] Applied scale={scale}, offset={offset}")

    # ── Squeeze extra dimensions (time, level, scan_line) ──────
    while data.ndim > 2:
        if data.shape[0] == 1:
            data = data.squeeze(axis=0)
        else:
            # Take the first slice along leading dim
            data = data[0]
            print(f"  [Warning] Multiple slices in leading dim; using index 0")

    if data.ndim != 2:
        raise ValueError(
            f"Expected 2D array after processing, got shape {data.shape}.\n"
            f"  Run inspect_nc.py to diagnose the file structure."
        )

    valid_count = int(np.sum(~np.isnan(data)))
    total_count = data.size
    print(f"  Loaded '{resolved}' from {os.path.basename(filepath)} "
          f"— shape: {data.shape}, "
          f"valid pixels: {valid_count}/{total_count} "
          f"({100*valid_count/total_count:.1f}%), "
          f"range: [{np.nanmin(data):.1f}, {np.nanmax(data):.1f}]")

    ds.close()
    return data


def normalize_to_tensor(
    arr: np.ndarray,
    target_h: int = TARGET_H,
    target_w: int = TARGET_W,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> torch.Tensor:
    """
    Converts a 2D scientific array into a (1, 3, H, W) float32 tensor
    in [0, 1], suitable for RIFE / standard PyTorch vision models.

    Steps:
      1. Percentile clip   → removes sensor noise / bad pixels
      2. Min-max scale     → [0.0, 1.0]
      3. Resize            → TARGET_H × TARGET_W (bilinear)
      4. Replicate channel → 1-channel grayscale → 3-channel RGB
      5. Add batch dim     → (1, 3, H, W)
    """
    # --- 1. Percentile clip (robust to outliers / fill values) ---
    v_low  = np.nanpercentile(arr, p_low)
    v_high = np.nanpercentile(arr, p_high)
    arr = np.clip(arr, v_low, v_high)

    # --- 2. Min-max normalization ---
    span = v_high - v_low if (v_high - v_low) > 1e-6 else 1.0
    arr_norm = (arr - v_low) / span           # now in [0, 1]
    arr_norm = arr_norm.astype(np.float32)

    # --- 2.5 Fill NaNs before OpenCV resize ---
    # Space pixels (outside Earth disk) are NaN. OpenCV resize will propagate
    # NaN across the image. Fill them with 0 (black space).
    np.nan_to_num(arr_norm, copy=False, nan=0.0)

    # --- 3. Resize with OpenCV ---
    if arr_norm.shape != (target_h, target_w):
        arr_norm = cv2.resize(
            arr_norm, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

    # --- 4. Grayscale → 3-channel (H, W) → (3, H, W) ---
    arr_3ch = np.stack([arr_norm, arr_norm, arr_norm], axis=0)  # (3, H, W)

    # --- 5. Tensor + batch dim ---
    tensor = torch.from_numpy(arr_3ch).unsqueeze(0)  # (1, 3, H, W)
    return tensor


# ─────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────
def get_frame_tensors(use_dummy: bool = True):
    """
    Main entry point called by ml_engine.py and app.py.

    Args:
        use_dummy: If True and .nc files are missing, auto-generate
                   synthetic data so the pipeline can run immediately.

    Returns:
        tuple:
            frame_t0  (torch.Tensor)  shape (1, 3, H, W) — t=0  min
            frame_t30 (torch.Tensor)  shape (1, 3, H, W) — t=30 min
            raw_t0    (np.ndarray)    original 2D array t=0
            raw_t30   (np.ndarray)    original 2D array t=30
    """
    print("\n[DataProcessor] Loading satellite frames...")

    if use_dummy:
        # Auto-create dummy .nc files if they don't already exist
        if not os.path.exists(T0_FILE):
            print("  [Info] Real .nc files not found. Generating dummy data...")
            _create_dummy_nc(T0_FILE, seed=42)
        if not os.path.exists(T30_FILE):
            _create_dummy_nc(T30_FILE, seed=99)

    raw_t0  = load_nc_channel(T0_FILE,  VARIABLE_NAME)
    raw_t30 = load_nc_channel(T30_FILE, VARIABLE_NAME)

    tensor_t0  = normalize_to_tensor(raw_t0)
    tensor_t30 = normalize_to_tensor(raw_t30)

    print(f"  Tensor shapes: {tensor_t0.shape} | dtype: {tensor_t0.dtype}")
    print(f"  Value range : [{tensor_t0.min():.4f}, {tensor_t0.max():.4f}]")
    print("[DataProcessor] Done.\n")

    return tensor_t0, tensor_t30, raw_t0, raw_t30


def tensor_to_display_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a (1, 3, H, W) or (3, H, W) float32 tensor [0,1]
    into a uint8 HWC BGR numpy array for OpenCV / display.
    """
    t = tensor.squeeze(0).detach().cpu().numpy()   # (3, H, W)
    t = np.transpose(t, (1, 2, 0))                  # (H, W, 3)
    t = np.clip(t * 255.0, 0, 255).astype(np.uint8)
    return t  # RGB uint8


# ─────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0, t30, raw0, raw30 = get_frame_tensors(use_dummy=True)
    print(f"frame_t0  tensor: {t0.shape},  min={t0.min():.4f}, max={t0.max():.4f}")
    print(f"frame_t30 tensor: {t30.shape}, min={t30.min():.4f}, max={t30.max():.4f}")
    print("\n[OK] data_processor.py is working correctly.")
