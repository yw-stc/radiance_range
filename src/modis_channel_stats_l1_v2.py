#!/usr/bin/env python3
import argparse
import glob
import math
import tempfile
import numpy as np
from pyhdf.SD import SD, SDC

DEFAULT_PATTERN = "/mnt/efs_clavrx/ywang/run/07272021/modisinput/day/MYD021KM.A2019099.*.hdf"
CHANNELS = [3, 1, 2, 26, 6, 7, 20, 29, 31, 32, 33]

# band -> (SDS name, 0-based index inside SDS)
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

def _arr(x):
    return np.array(x) if x is not None else None

def _get_attr(attrs, name, default=None):
    return attrs[name] if name in attrs else default

def _slice_band(data3d, band_idx0):
    # expected [band, row, col] but be defensive
    if data3d.shape[0] >= band_idx0 + 1:
        return data3d[band_idx0, :, :]
    if data3d.shape[-1] >= band_idx0 + 1:
        return data3d[:, :, band_idx0]
    raise RuntimeError(f"Cannot index band {band_idx0} in array with shape {data3d.shape}")

def calibrated_2d(hdf: SD, sds_name: str, band_idx0: int, band_number: int):
    """
    Returns (phys_2d, kind_str), where kind_str is 'radiance'.
    Invalid pixels masked to NaN.
    """
    sds = hdf.select(sds_name)
    data = sds.get()
    if data.ndim != 3:
        raise RuntimeError(f"Unexpected shape for {sds_name}: {data.shape} (expected 3D)")

    attrs = sds.attributes()

    dn = _slice_band(data, band_idx0).astype(np.float32)

    # Mask invalid DN values BEFORE scaling
    mask = np.zeros(dn.shape, dtype=bool)

    fill = _get_attr(attrs, "_FillValue", None)
    if fill is None:
        fill = _get_attr(attrs, "fill_value", None)
    if fill is not None:
        fill_val = float(_arr(fill).ravel()[0])
        mask |= (dn == fill_val)

    valid_range = _get_attr(attrs, "valid_range", None)
    if valid_range is not None:
        vr = _arr(valid_range).astype(np.float32).ravel()
        if vr.size >= 2:
            vmin, vmax = float(vr[0]), float(vr[1])
            mask |= (dn < vmin) | (dn > vmax)

    # 1) If scale_factor/add_offset exist, use them (generic convention)
    scale_factor = _get_attr(attrs, "scale_factor", None)
    add_offset = _get_attr(attrs, "add_offset", None)
    if scale_factor is not None and add_offset is not None:
        sf = float(_arr(scale_factor).ravel()[0])
        ao = float(_arr(add_offset).ravel()[0])
        phys = dn * sf + ao
        kind = "radiance"
    else:
        # 2) MODIS L1B radiance convention for all channels (VIS/SWIR/IR)
        scales = _get_attr(attrs, "radiance_scales", None)
        offsets = _get_attr(attrs, "radiance_offsets", None)
        if scales is None or offsets is None:
            raise RuntimeError(
                f"Missing radiance_scales/radiance_offsets in {sds_name} for band {band_number}."
            )
        scale = float(_arr(scales).ravel()[band_idx0])
        offset = float(_arr(offsets).ravel()[band_idx0])
        phys = (dn - offset) * scale
        kind = "radiance"

    phys = phys.astype(np.float32)
    phys[mask] = np.nan
    return phys, kind

class RunningStats:
    def __init__(self, tmp_path: str):
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self._tmp_path = tmp_path
        self._tmp_file = open(tmp_path, "wb")

    def update(self, values_1d: np.ndarray) -> None:
        if values_1d.size == 0:
            return
        self.count += int(values_1d.size)
        self.total += float(np.sum(values_1d, dtype=np.float64))
        self.total_sq += float(np.sum(values_1d * values_1d, dtype=np.float64))
        self.minimum = min(self.minimum, float(np.min(values_1d)))
        self.maximum = max(self.maximum, float(np.max(values_1d)))
        np.asarray(values_1d, dtype=np.float32).tofile(self._tmp_file)

    def mean(self):
        if self.count == 0:
            return float("nan")
        return self.total / self.count

    def std(self):
        if self.count == 0:
            return float("nan")
        mean = self.mean()
        var = max(0.0, self.total_sq / self.count - mean * mean)
        return math.sqrt(var)

    def min(self):
        return self.minimum if self.count else float("nan")

    def max(self):
        return self.maximum if self.count else float("nan")

    def close(self):
        if not self._tmp_file.closed:
            self._tmp_file.close()

    def median(self):
        if self.count == 0:
            return float("nan")
        self.close()
        arr = np.memmap(self._tmp_path, dtype=np.float32, mode="r+", shape=(self.count,))
        k = self.count // 2
        if self.count % 2 == 1:
            arr.partition(k)
            med = float(arr[k])
        else:
            arr.partition((k - 1, k))
            med = float((arr[k - 1] + arr[k]) * 0.5)
        del arr
        return med


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute one aggregated summary for selected MYD021KM channels across all matching HDF files."
    )
    parser.add_argument(
        "pattern",
        nargs="?",
        default=DEFAULT_PATTERN,
        help=f"Input file glob pattern (default: {DEFAULT_PATTERN})",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        type=int,
        default=CHANNELS,
        help=f"MODIS channels to process (default: {' '.join(str(ch) for ch in CHANNELS)})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {args.pattern}")

    missing = [b for b in args.channels if b not in BAND_TO_SDS]
    if missing:
        raise ValueError(f"No SDS mapping provided for bands: {missing}")

    with tempfile.TemporaryDirectory(prefix="modis_l1_stats_") as tmpdir:
        summaries = {
            band: RunningStats(f"{tmpdir}/band_{band}.bin")
            for band in args.channels
        }
        kinds = {}

        for path in paths:
            hdf = SD(path, SDC.READ)
            try:
                for band in args.channels:
                    sds_name, bidx0 = BAND_TO_SDS[band]
                    phys, kind = calibrated_2d(hdf, sds_name, bidx0, band)
                    kinds[band] = kind
                    valid = phys[np.isfinite(phys)]
                    summaries[band].update(valid.astype(np.float64))
            finally:
                hdf.end()

        for st in summaries.values():
            st.close()

        print(f"Pattern: {args.pattern}")
        print(f"Matched files: {len(paths)}")
        print("Summary is aggregated over all valid pixels from all matched files.")
        print("All bands are calibrated to radiance using radiance_scales/radiance_offsets.")
        print("")
        print(
            f"{'Band':>4}  {'Kind':<11}  {'SDS':<22}  {'SDS_band#':>8}  {'valid_count':>12}  {'mean':>14}  {'std':>14}  {'median':>14}  {'min':>14}  {'max':>14}"
        )
        print("-" * 148)

        for band in args.channels:
            sds_name, bidx0 = BAND_TO_SDS[band]
            st = summaries[band]
            kind = kinds.get(band, "unknown")
            print(
                f"{band:>4}  {kind:<11}  {sds_name:<22}  {bidx0+1:>8}  {st.count:>12}  {st.mean():>14.6g}  {st.std():>14.6g}  {st.median():>14.6g}  {st.min():>14.6g}  {st.max():>14.6g}"
            )

if __name__ == "__main__":
    main()
