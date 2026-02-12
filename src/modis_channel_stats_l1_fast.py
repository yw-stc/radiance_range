#!/usr/bin/env python3
"""
modis_channel_stats_l1_v2.py

Compute ONE combined statistic summary (median/min/max) across ALL files matching a glob.

- ALL channels processed as RADIANCE via radiance_scales/radiance_offsets
- Memory-safe: reads ONLY the requested band plane (hyperslab)
- Masks fill/out-of-range using DN-space valid_range and _FillValue before scaling
- Median is approximate via streaming histogram; min/max are exact
- Uses per-channel upper clip limits for histogram range [0, upper_clip]
- Reports N(>clip) and Pct(>clip)=100*N(>clip)/N(valid)
- Adds columns: MODIS band number, MODIS band wavelength, SPECTRE band label

Usage:
  python modis_channel_stats_l1_v2.py "/path/to/MYD021KM.A2019099.*.hdf"
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

# User-provided upper clip limits, aligned to CHANNELS order above
UPPER_CLIP_LIST = [550, 425, 225, 40, 40, 10, 1.5, 14, 14, 10, 12]
assert len(UPPER_CLIP_LIST) == len(CHANNELS), "UPPER_CLIP_LIST must match CHANNELS length"
BAND_TO_UPPER_CLIP = dict(zip(CHANNELS, map(float, UPPER_CLIP_LIST)))

# User-provided SPECTRE band labels, aligned to CHANNELS order above
SPECTRE_LIST = ["A", "1", "2", "3", "4", "B", "5", "6", "7", "C", "8"]
assert len(SPECTRE_LIST) == len(CHANNELS), "SPECTRE_LIST must match CHANNELS length"
BAND_TO_SPECTRE = dict(zip(CHANNELS, SPECTRE_LIST))

# MODIS band wavelength ranges (µm), keyed by MODIS band number
BAND_TO_WAVELENGTH_UM = {
    1:  "0.620–0.670",
    2:  "0.841–0.876",
    3:  "0.459–0.479",
    6:  "1.628–1.652",
    7:  "2.105–2.155",
    20: "3.660–3.840",
    26: "1.360–1.390",
    29: "8.400–8.700",
    31: "10.780–11.280",
    32: "11.770–12.270",
    33: "13.185–13.485",
}

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

# Histogram settings (approx median)
HIST_NBINS = 4096
HIST_LOWER = 0.0  # radiance lower bound for histogram range


# -----------------------
# HDF helpers
# -----------------------
def read_band_plane(sds, band_idx0: int) -> np.ndarray:
    """Read a single band plane from a 3D EV SDS using hyperslab."""
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
            raise ValueError(f"Histogram range invalid: [{self.hmin},{self.hmax}]")

        self.hist = np.zeros(self.nbins, dtype=np.int64)
        self.count = 0
        self.vmin = np.inf
        self.vmax = -np.inf

        self.clipped_low = 0
        self.clipped_high = 0  # N(>clip)

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

    def pct_clipped_high(self) -> float:
        if self.count == 0:
            return np.nan
        return 100.0 * float(self.clipped_high) / float(self.count)


# -----------------------
# Main
# -----------------------
def main():
    if len(sys.argv) < 2:
        print('Usage: python modis_channel_stats_l1_v2.py "/path/to/MYD021KM.A2019099.*.hdf"')
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
                acc[band] = RunningStatsWithHistMedian(
                    hmin=HIST_LOWER,
                    hmax=BAND_TO_UPPER_CLIP[band],
                    nbins=HIST_NBINS,
                )
            acc[band].update(rad)

        if file_warn:
            n_warn_files += 1

    # Output
    print("\n" + "=" * 164)
    print(f"Pattern: {pattern}")
    print(f"Files matched: {len(files)} | open failures: {n_open_fail} | files w/ warnings: {n_warn_files}")
    print("All channels processed as RADIANCE via radiance_scales/radiance_offsets.")
    print(f"Median: histogram-approx ({HIST_NBINS} bins) over [0, upper_clip]. Min/Max: exact.")
    print("=" * 164 + "\n")

    print(
        f"{'MODIS':>5}  {'SPECTRE':>7}  {'Wavelength(µm)':>15}  "
        f"{'upper_clip':>10}  {'median':>14}  {'min':>14}  {'max':>14}  "
        f"{'N(valid)':>12}  {'N(>clip)':>10}  {'Pct(>clip)':>11}"
    )
    print("-" * 160)

    for band in CHANNELS:
        spectre = BAND_TO_SPECTRE.get(band, "-")
        wl = BAND_TO_WAVELENGTH_UM.get(band, "unknown")

        if band not in acc or acc[band].count == 0:
            print(
                f"{band:>5}  {spectre:>7}  {wl:>15}  "
                f"{BAND_TO_UPPER_CLIP[band]:>10g}  {'nan':>14}  {'nan':>14}  {'nan':>14}  "
                f"{0:>12}  {0:>10}  {'nan':>11}"
            )
            continue

        a = acc[band]
        print(
            f"{band:>5}  {spectre:>7}  {wl:>15}  "
            f"{a.hmax:>10g}  {a.median():>14.6g}  {a.vmin:>14.6g}  {a.vmax:>14.6g}  "
            f"{a.count:>12d}  {a.clipped_high:>10d}  {a.pct_clipped_high():>10.6g}%"
        )


if __name__ == "__main__":
    main()

