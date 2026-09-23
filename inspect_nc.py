"""
inspect_nc.py — Project Aviral
================================
Utility script to probe any NetCDF file and show exactly what variables,
dimensions, coordinates, and value ranges it contains.

Run this FIRST on any new .nc file to identify the correct VARIABLE_NAME
before using it in data_processor.py.

Usage:
    python inspect_nc.py                        # inspects data/insat_t0.nc
    python inspect_nc.py path/to/yourfile.nc    # inspects a specific file
"""

import sys
import os
import numpy as np

def inspect(filepath: str) -> None:
    try:
        import xarray as xr
    except ImportError:
        print("ERROR: xarray not installed. Run: pip install xarray netCDF4")
        sys.exit(1)

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)

    filesize_mb = os.path.getsize(filepath) / 1024 / 1024
    print(f"\n{'='*60}")
    print(f"  NetCDF Inspector — Project Aviral")
    print(f"{'='*60}")
    print(f"  File   : {filepath}")
    print(f"  Size   : {filesize_mb:.2f} MB")
    print(f"{'='*60}\n")

    ds = xr.open_dataset(filepath, mask_and_scale=False)

    # ── Global attributes ─────────────────────────────────
    print("[1] Global Attributes")
    print("-" * 40)
    for k, v in ds.attrs.items():
        val_str = str(v)[:80] + ("..." if len(str(v)) > 80 else "")
        print(f"  {k:<30} = {val_str}")
    print()

    # ── Dimensions ───────────────────────────────────────
    print("[2] Dimensions")
    print("-" * 40)
    for dim, size in ds.dims.items():
        print(f"  {dim:<20} : {size}")
    print()

    # ── Coordinates ──────────────────────────────────────
    print("[3] Coordinates")
    print("-" * 40)
    for name, coord in ds.coords.items():
        vals = coord.values
        if vals.ndim == 1 and len(vals) > 0:
            print(f"  {name:<20} : shape={vals.shape}  "
                  f"range=[{float(vals.min()):.4f}, {float(vals.max()):.4f}]  "
                  f"dtype={vals.dtype}")
        else:
            print(f"  {name:<20} : shape={vals.shape}  dtype={vals.dtype}")
    print()

    # ── Data Variables ────────────────────────────────────
    print("[4] Data Variables  <-- USE ONE OF THESE AS VARIABLE_NAME")
    print("-" * 40)

    candidates = []  # 2D variables that look like images

    for vname, var in ds.data_vars.items():
        arr = var.values

        # Basic stats (skip if all NaN)
        valid = arr[~np.isnan(arr)] if arr.dtype.kind == 'f' else arr.flatten()
        if len(valid) == 0:
            stats = "all NaN"
        else:
            stats = (f"min={float(valid.min()):.2f}  "
                     f"max={float(valid.max()):.2f}  "
                     f"mean={float(valid.mean()):.2f}")

        fill_val = var.attrs.get("_FillValue", var.attrs.get("missing_value", "none"))
        units    = var.attrs.get("units", "")
        long_name = var.attrs.get("long_name", var.attrs.get("standard_name", ""))

        print(f"\n  Variable : {vname}")
        print(f"    shape    : {arr.shape}")
        print(f"    dtype    : {arr.dtype}")
        print(f"    units    : {units}")
        print(f"    long_name: {long_name}")
        print(f"    FillValue: {fill_val}")
        print(f"    stats    : {stats}")
        print(f"    dims     : {list(var.dims)}")

        # Flag 2D candidates (good for image interpolation)
        squeezed = arr.squeeze()
        if squeezed.ndim == 2:
            h, w = squeezed.shape
            if h > 50 and w > 50:
                candidates.append(vname)
                print(f"    *** GOOD CANDIDATE for VFI (2D image: {h}x{w}) ***")

    print()

    # ── Recommendation ────────────────────────────────────
    print("=" * 60)
    print("[5] Recommendation")
    print("-" * 40)
    if candidates:
        print(f"\n  Set in data_processor.py:")
        print(f'      VARIABLE_NAME = "{candidates[0]}"')
        if len(candidates) > 1:
            print(f"\n  Other valid options: {candidates[1:]}")
    else:
        print("  No obvious 2D image variable found.")
        print("  Try squeezing extra dimensions manually in data_processor.py")
        all_vars = list(ds.data_vars.keys())
        if all_vars:
            print(f"  Available variables: {all_vars}")
    print("=" * 60)
    ds.close()


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/insat_t0.nc"
    inspect(target)
