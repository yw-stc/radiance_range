#!/usr/bin/env python3
"""
modis_channel_stats_l1_v2.py

Compute ONE combined statistic summary (median/min/max) across ALL files matching a glob.

- ALL bands processed as RADIANCE via radiance_scales/radiance_offsets
- Memory-safe: reads ONLY the requested band plane (hyperslab), not the full EV cube
- Masks fill/out-of-range using DN-space valid_range and _FillValue before scaling
- Median is approximate via streaming histogram; min/max are exact

Usage:
  python modis_channel_stats_l1_v2.py "/path/to/MYD021KM.A2019099.*.hdf"

Examples:
  python src/modis_channel_stats_l1_v2.py "/mnt/efs_clavrx/ywang/run/07272021/modisinput/day/MYD021KM.A2019099.*.hdf"
  python src/modis_channel_stats_l1_v2.py "/mnt/efs_clavrx/ywang/run/07272021/modisinput/night/MYD021KM.A2019099.*.hdf"
  python src/modis_channel_stats_l1_v2.py "/mnt/efs_clavrx/ywang/run/07272021/modisinput/*/MYD021KM.A2019100.*.hdf"
"""

import sys
import os
import glob
import numpy as np
from pyhdf.SD import SD, SDC

# -----------------------
# Channels and mapping
# -----------------------
CHANNELS = [3, 1, 2, 26, 6, 7, 20, 29, 31, 32, 33]

# band -> (SDS name, 0-based index within SDS)
BAND_TO_SDS = {
    1: ("EV_250_Aggr1km_RefSB", 0),
    2: ("EV_250_Aggr1km_RefSB", 1),
    3: ("EV_500_Aggr1km_RefSB", 0),
    6: ("EV_500_Aggr1km_RefSB", 3),
    7: ("EV_500_Aggr1km_RefSB", 4),
    26: ("EV_1KM_RefSB", 14),
    20: ("EV_1KM_Emissive", 0),
    29: ("EV_1KM_Emissive", 8),
    31: ("EV_1KM_Emissive", 10),
    32: ("EV_1KM_Emissive", 11),
    33: ("EV_1KM_Emissive", 12),
}

def is_thermal_band(b: int) -> bool:
    return 20 <= b <= 36

# -----------------------
# Histogram settings (for approximate median)
# -----------------------
HIST_NBINS = 4096

# Ranges are used ONLY for histogram-median; min/max are exact.
# Widen if you see clipping counts reported.
RADIANCE_RANGE_RSB = (0.0, 800.0)  # reflective solar bands radiance can be large
RADIANCE_RANGE_TEB = (0.0, 40.0)   # thermal emissive typical; widen if needed


# -----------------------
# HDF helpers
# -----------------------
def read_band_plane(sds, band_idx0: int) -> np.ndarray:
    """
    Read a single band plane from a 3D EV SDS using hyperslab.
    Expected order for MODIS EV: [band, row, col].
    Returns DN as float32 2D array (row, col).
    """
    name, rank, dims, dtype, nattrs = sds.info()
    if rank != 3:
        raise RuntimeError(f"SDS {name} expected rank=3, got rank={rank}, dims={dims}")

    nb, nr, nc = dims[0], dims[1], dims[2]
    if band_idx0 < 0 or band_idx0 >= nb:
        raise RuntimeError(f"band_idx0={band_idx0} out of range for {name} (nbands={nb})")

    dn = sds.get(start=(band_idx0, 0, 0), count=(1, nr, nc))
    dn = np.asarray(dn, dtype=np.float32).squeeze(axis=0)  # -> (nr, nc)
    return dn

def calibrated_radiance_2d(hdf: SD, sds_name: str, band_idx0: int) -> np.ndarray:
    """
    Returns radiance 2D float32 with invalid pixels as NaN.
    radiance = (DN - radiance_offsets[band]) * radiance_scales[band]
    """
    sds = hdf.select(sds_name)
    attrs = sds.attributes()

    dn = read_band_plane(sds, band_idx0)

    # Mask invalid DN BEFORE calibration
    mask = np.zeros(dn.shape, dtype=bool)

    fill = attrs.get("_FillValue", attrs.get("fill_value", None))
    if fill is not None:
        fill_val = float(np.array(fill).ravel()[0])
        mask |= (dn == fill_val)

    valid_range = attrs.get("valid_range", None)
    if valid_range is not None:
        vr = np.array(valid_range, dtype=np.float32).ravel()
        if vr.size >= 2:
            vmin, vmax = float(vr[0]), float(vr[1])
            mask |= (dn < vmin) | (dn > vmax)

    scales = attrs.get("radiance_scales", None)
    offsets = attrs.get("radiance_offsets", None)
    if scales is None or offsets is None:
        raise RuntimeError(f"Missing radiance_scales/offsets in {sds_name}")

    scale = float(np.array(scales).ravel()[band_idx0])
    offset = float(np.array(offsets).ravel()[band_idx0])

    rad = (dn - offset) * scale
    rad = rad.astype(np.float32)
    rad[mask] = np.nan
    return rad


# -----------------------
# Streaming stats (hist median + exact min/max)
# -----------------------
class RunningStatsWithHistMedian:
    def __init__(self, hmin: float, hmax: float, nbins: int):
        self.hmin = float(hmin)
        self.hmax = float(hmax)
        self.nbins = int(nbins)
        if not (self.hmax > self.hmin):
            raise ValueError("Histogram range must have hmax > hmin")

        self.hist = np.zeros(self.nbins, dtype=np.int64)
        self.count = 0
        self.vmin = np.inf
        self.vmax = -np.inf
        self.clipped_low = 0
        self.clipped_high = 0

    def update(self, arr2d: np.ndarray):
        vals = arr2d[np.isfinite(arr2d)]
        if vals.size == 0:
            return

        self.count += int(vals.size)

        mn = float(vals.min())
        mx = float(vals.max())
        if mn < self.vmin:
            self.vmin = mn
        if mx > self.vmax:
            self.vmax = mx

        self.clipped_low += int(np.sum(vals < self.hmin))
        self.clipped_high += int(np.sum(vals > self.hmax))

        vals_clipped = np.clip(vals, self.hmin, self.hmax)
        h, _ = np.histogram(vals_clipped, bins=self.nbins, range=(self.hmin, self.hmax))
        self.hist += h.astype(np.int64)

    def median(self) -> float:
        if self.count == 0:
            return np.nan
        cum = np.cumsum(self.hist)
        target = (self.count - 1) / 2.0
        idx = int(np.searchsorted(cum, target, side="left"))
        idx = max(0, min(self.nbins - 1, idx))
        bin_width = (self.hmax - self.hmin) / self.nbins
        return float(self.hmin + (idx + 0.5) * bin_width)

    def hist_range_str(self) -> str:
        s = f"[{self.hmin:g},{self.hmax:g}]"
        if self.clipped_low or self.clipped_high:
            s += f" clipped(<)={self.clipped_low} clipped(>)={self.clipped_high}"
        return s


# -----------------------
# Main
# -----------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python modis_channel_stats_l1_v2.py \"/path/pattern/MYD021KM.A2019099.*.hdf\"")
        sys.exit(2)

    pattern = sys.argv[1]
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files matched: {pattern}")

    acc = {}

    n_open_fail = 0
    n_warn_files = 0

    for fp in files:
        try:
            hdf = SD(fp, SDC.READ)
        except Exception as e:
            n_open_fail += 1
            print(f"[WARN] Cannot open {fp}: {e}")
            continue

        file_warn = False

        for band in CHANNELS:
            sds_name, bidx0 = BAND_TO_SDS[band]
            try:
                rad = calibrated_radiance_2d(hdf, sds_name, bidx0)
            except Exception as e:
                file_warn = True
                print(f"[WARN] {os.path.basename(fp)} band {band} ({sds_name}): {e}")
                continue

            if band not in acc:
                hmin, hmax = (RADIANCE_RANGE_TEB if is_thermal_band(band) else RADIANCE_RANGE_RSB)
                acc[band] = RunningStatsWithHistMedian(hmin=hmin, hmax=hmax, nbins=HIST_NBINS)

            acc[band].update(rad)

        if file_warn:
            n_warn_files += 1

    # Output
    print("\n" + "=" * 104)
    print(f"Pattern: {pattern}")
    print(f"Files matched: {len(files)} | open failures: {n_open_fail} | files w/ warnings: {n_warn_files}")
    print("All channels processed as RADIANCE via radiance_scales/radiance_offsets.")
    print(f"Median: histogram-approx ({HIST_NBINS} bins). Min/Max: exact.")
    print("=" * 104 + "\n")

    print(f"{'Band':>4}  {'median':>14}  {'min':>14}  {'max':>14}  {'N(valid)':>12}  {'HistRange':>34}")
    print("-" * 96)

    for band in CHANNELS:
        if band not in acc or acc[band].count == 0:
            print(f"{band:>4}  {'nan':>14}  {'nan':>14}  {'nan':>14}  {0:>12}  {'-':>34}")
            continue

        a = acc[band]
        print(f"{band:>4}  {a.median():>14.6g}  {a.vmin:>14.6g}  {a.vmax:>14.6g}  {a.count:>12d}  {a.hist_range_str():>34}")

if __name__ == "__main__":
    main()

