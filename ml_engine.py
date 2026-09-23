"""
ml_engine.py — Project Aviral
================================
Loads a pre-trained RIFE (Real-Time Intermediate Flow Estimation) model
and interpolates the t=15 min frame between t=0 and t=30 INSAT frames.

HOW TO USE:
-----------
Option A — Use the built-in fallback (optical flow via OpenCV):
    Works immediately. No weight files needed.
    Set USE_RIFE = False below.

Option B — Use real RIFE weights (recommended for demo quality):
    1. Clone the RIFE repo:
           git clone https://github.com/megvii-research/ECCV2022-RIFE.git
    2. Download pretrained weights from:
           https://github.com/megvii-research/ECCV2022-RIFE/releases
       or  https://drive.google.com/...  (see their README)
    3. Place the weights folder at:
           models/train_log/   (containing flownet.pkl, contextnet.pkl, etc.)
    4. Clone their model code or place:
           model/RIFE_HDv3.py  alongside this file.
    5. Set USE_RIFE = True below.

SSIM / MSE:
-----------
After interpolation, we compare the AI-interpolated t=15 frame against a
naive linear blend (average of t=0 and t=30) to quantify the improvement.
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import warnings
warnings.filterwarnings("ignore")

# skimage is imported lazily inside compute_metrics() to avoid IDE
# static-analysis false-positives when the analyser uses a different
# interpreter path than the runtime environment.

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
USE_RIFE    = True            # RIFE deep learning model ACTIVE
MODELS_DIR  = "models"       # Where RIFE weights/folders live
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[MLEngine] Device: {DEVICE}")


# ═════════════════════════════════════════════════════════════
#  SECTION 1: RIFE WRAPPER (Optional — heavyweight)
# ═════════════════════════════════════════════════════════════
class RIFEWrapper:
    """
    Thin wrapper around the official RIFE HD model (hzwer/ECCV2022-RIFE).

    Requires:
      - ECCV2022-RIFE source cloned to models/ECCV2022-RIFE/
      - Pre-trained weights at models/train_log/flownet.pkl

    Run download_rife_weights.py to get the weights automatically.
    """

    def __init__(self, model_dir: str = MODELS_DIR):
        # Add the RIFE source directory to Python path so model.* imports work
        rife_src = os.path.join(model_dir, "ECCV2022-RIFE")
        if not os.path.isdir(rife_src):
            raise FileNotFoundError(
                f"RIFE source not found at {rife_src}\n"
                f"  Run: git clone --depth=1 https://github.com/hzwer/ECCV2022-RIFE.git {rife_src}"
            )
        sys.path.insert(0, rife_src)

        # The downloaded HD weights came with their own exact model files 
        # (RIFE_HDv3.py and IFNet_HDv3.py) to avoid version mismatches.
        # They expect to be imported via 'train_log.xxx'
        sys.path.insert(0, model_dir)
        train_log = os.path.join(model_dir, "train_log")

        weight_file = os.path.join(train_log, "flownet.pkl")
        if not os.path.isfile(weight_file):
            raise FileNotFoundError(
                f"RIFE weights not found: {weight_file}\n"
                f"  Run: python download_rife_weights.py"
            )

        try:
            from train_log.RIFE_HDv3 import Model  # type: ignore
            self.model = Model()
            self.model.eval()

            # Load weights manually to handle PyTorch 2.0 backward compatibility
            raw = torch.load(
                weight_file,
                map_location="cpu",
                weights_only=False,
            )
            # Strip DDP module prefix if present
            if any("module." in k for k in raw.keys()):
                raw = {k.replace("module.", ""): v for k, v in raw.items() if "module." in k}
            self.model.flownet.load_state_dict(raw)
            print(f"[RIFE] Model loaded successfully from: {weight_file}")
        except ImportError as e:
            raise ImportError(f"Failed to import RIFE model: {e}")

    def interpolate(
        self, frame0: torch.Tensor, frame1: torch.Tensor, timestep: float = 0.5
    ) -> torch.Tensor:
        """
        Args:
            frame0, frame1: (1, 3, H, W) tensors in [0, 1]
            timestep:       0.5 → midpoint (t=15 min)
        Returns:
            interpolated:   (1, 3, H, W) tensor in [0, 1]
        """
        with torch.no_grad():
            f0 = frame0.to(DEVICE)
            f1 = frame1.to(DEVICE)
            # RIFE_HDv3 only does 2x interpolation (always computes t=0.5)
            mid = self.model.inference(f0, f1)
        return mid.clamp(0.0, 1.0).cpu()


# ═════════════════════════════════════════════════════════════
#  SECTION 2: OPTICAL FLOW FALLBACK (Zero-dependency, always works)
# ═════════════════════════════════════════════════════════════
class OpticalFlowInterpolator:
    """
    Fallback interpolator using dense Farneback optical flow (OpenCV).

    This is NOT deep learning, but produces visually convincing results
    for slowly-evolving satellite imagery and runs with zero extra downloads.

    Algorithm:
        1. Compute forward flow:  F(t0 → t30)
        2. Warp t0 by 0.5 * F    → estimate of t15 from t0 side
        3. Compute backward flow: F(t30 → t0)
        4. Warp t30 by 0.5 * F   → estimate of t15 from t30 side
        5. Blend the two warped frames 50/50 (bidirectional)
    """

    @staticmethod
    def _tensor_to_gray_uint8(tensor: torch.Tensor) -> np.ndarray:
        """Convert (1, 3, H, W) tensor → (H, W) uint8 grayscale."""
        arr = tensor.squeeze(0).numpy()          # (3, H, W)
        arr = np.transpose(arr, (1, 2, 0))       # (H, W, 3)
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def _warp_frame(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """
        Warp a (H, W, C) float image by a (H, W, 2) flow field.
        Uses cv2.remap with bilinear interpolation.
        """
        H, W = flow.shape[:2]
        map_x = np.arange(W, dtype=np.float32)[None, :] + flow[..., 0]
        map_y = np.arange(H, dtype=np.float32)[:, None] + flow[..., 1]
        map_x = np.clip(map_x, 0, W - 1)
        map_y = np.clip(map_y, 0, H - 1)

        if frame.ndim == 2:
            warped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
        else:
            warped = np.stack(
                [cv2.remap(frame[..., c], map_x, map_y, cv2.INTER_LINEAR)
                 for c in range(frame.shape[-1])],
                axis=-1,
            )
        return warped

    def interpolate(
        self, frame0: torch.Tensor, frame1: torch.Tensor, timestep: float = 0.5
    ) -> torch.Tensor:
        """
        Args:
            frame0, frame1: (1, 3, H, W) float32 tensors in [0, 1]
            timestep:       fraction along the interval (0.5 = midpoint)
        Returns:
            interpolated:   (1, 3, H, W) float32 tensor in [0, 1]
        """
        # Convert to uint8 grayscale for flow computation
        g0 = self._tensor_to_gray_uint8(frame0)
        g1 = self._tensor_to_gray_uint8(frame1)

        # Farneback dense optical flow parameters (tuned for large smooth fields)
        farn_params = dict(
            pyr_scale=0.5, levels=5, winsize=15,
            iterations=3, poly_n=7, poly_sigma=1.5,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        )

        # Forward & backward flows
        flow_fwd = cv2.calcOpticalFlowFarneback(g0, g1, None, **farn_params)
        flow_bwd = cv2.calcOpticalFlowFarneback(g1, g0, None, **farn_params)

        # Convert frame tensors → HWC float arrays for warping
        f0_np = np.transpose(frame0.squeeze(0).numpy(), (1, 2, 0))  # (H,W,3)
        f1_np = np.transpose(frame1.squeeze(0).numpy(), (1, 2, 0))  # (H,W,3)

        # Warp each frame toward the midpoint
        warped0 = self._warp_frame(f0_np,  flow_fwd * timestep)
        warped1 = self._warp_frame(f1_np,  flow_bwd * (1.0 - timestep))

        # Bidirectional blend
        blended = (1.0 - timestep) * warped0 + timestep * warped1
        blended = np.clip(blended, 0.0, 1.0).astype(np.float32)

        # Back to tensor (1, 3, H, W)
        result = torch.from_numpy(np.transpose(blended, (2, 0, 1))).unsqueeze(0)
        return result


# ═════════════════════════════════════════════════════════════
#  SECTION 3: METRICS
# ═════════════════════════════════════════════════════════════
def compute_metrics(
    pred: torch.Tensor, reference: torch.Tensor
) -> dict:
    """
    Computes SSIM and MSE between two (1, 3, H, W) tensors.

    We compare:
      - pred:      AI-interpolated t=15 frame
      - reference: naive linear blend (avg of t0 and t30) as baseline

    Returns:
        dict with keys: "ssim", "mse", "psnr"

    Note: skimage is imported lazily here to avoid IDE static-analysis
    errors when the analyser's interpreter path differs from the runtime.
    """
    # ── Lazy import: resolves IDE false-positive on skimage.metrics ──
    try:
        from skimage.metrics import structural_similarity as _ssim  # type: ignore
        _use_skimage = True
    except ImportError:
        _use_skimage = False

    # Convert to (H, W, 3) numpy arrays in [0, 1]
    def _to_numpy(t: torch.Tensor) -> np.ndarray:
        return np.transpose(t.squeeze(0).detach().cpu().numpy(), (1, 2, 0))

    pred_np = _to_numpy(pred).astype(np.float64)
    ref_np  = _to_numpy(reference).astype(np.float64)

    # SSIM — higher is better (max = 1.0)
    if _use_skimage:
        ssim_val = _ssim(
            pred_np, ref_np,
            data_range=1.0,
            channel_axis=2,      # scikit-image >= 0.19
            win_size=11,
        )
    else:
        # Fallback: channel-wise mean correlation coefficient (~SSIM proxy)
        mu1, mu2 = pred_np.mean(), ref_np.mean()
        sigma1   = pred_np.std()
        sigma2   = ref_np.std()
        sigma12  = float(np.mean((pred_np - mu1) * (ref_np - mu2)))
        c1, c2   = 0.01**2, 0.03**2
        ssim_val = (2*mu1*mu2 + c1) * (2*sigma12 + c2) / \
                   ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))

    # MSE — lower is better
    mse_val = float(np.mean((pred_np - ref_np) ** 2))

    # PSNR — higher is better
    psnr_val = 10 * math.log10(1.0 / mse_val) if mse_val > 1e-10 else float("inf")

    return {
        "ssim": round(float(ssim_val), 6),
        "mse":  round(mse_val,         8),
        "psnr": round(psnr_val,        4),
    }


# ═════════════════════════════════════════════════════════════
#  SECTION 4: PUBLIC API
# ═════════════════════════════════════════════════════════════
def get_interpolator():
    """
    Factory function — returns the best available interpolator.
    Tries RIFE first (if USE_RIFE=True), falls back to OpticalFlow.
    """
    if USE_RIFE:
        try:
            interp = RIFEWrapper(MODELS_DIR)
            print("[MLEngine] Using: RIFE HDv3 (deep learning)")
            return interp
        except Exception as e:
            print(f"[MLEngine] RIFE load failed: {e}")
            print("[MLEngine] Falling back to Optical Flow interpolator.")

    interp = OpticalFlowInterpolator()
    print("[MLEngine] Using: Optical Flow (Farneback bidirectional)")
    return interp


def run_interpolation(
    frame_t0: torch.Tensor,
    frame_t30: torch.Tensor,
    interpolator=None,
) -> dict:
    """
    Main entry point: runs interpolation and computes quality metrics.

    Args:
        frame_t0:     (1, 3, H, W) tensor — satellite image at t=0  min
        frame_t30:    (1, 3, H, W) tensor — satellite image at t=30 min
        interpolator: pre-loaded interpolator (or None to auto-create)

    Returns:
        dict with:
            "frame_t15"   : (1, 3, H, W) interpolated tensor
            "frame_naive" : (1, 3, H, W) naive linear blend (baseline)
            "metrics"     : {"ssim": ..., "mse": ..., "psnr": ...}
    """
    if interpolator is None:
        interpolator = get_interpolator()

    print("[MLEngine] Running frame interpolation...")
    frame_t15 = interpolator.interpolate(frame_t0, frame_t30, timestep=0.5)

    # Naive baseline: simple linear blend (average)
    frame_naive = ((frame_t0 + frame_t30) / 2.0).clamp(0.0, 1.0)

    # Metrics: AI output vs naive baseline
    metrics = compute_metrics(frame_t15, frame_naive)

    print(f"[MLEngine] Interpolation complete.")
    print(f"  SSIM : {metrics['ssim']:.4f}  (AI vs. linear blend)")
    print(f"  MSE  : {metrics['mse']:.6f}")
    print(f"  PSNR : {metrics['psnr']:.2f} dB")

    return {
        "frame_t15":   frame_t15,
        "frame_naive": frame_naive,
        "metrics":     metrics,
    }


# ─────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_processor import get_frame_tensors, tensor_to_display_image

    t0, t30, _, _ = get_frame_tensors(use_dummy=True)
    result = run_interpolation(t0, t30)

    t15    = result["frame_t15"]
    naive  = result["frame_naive"]
    metrics = result["metrics"]

    print(f"\n[OK] Interpolation succeeded.")
    print(f"   t15 tensor shape: {t15.shape}")
    print(f"   Metrics: {metrics}")
