#!/usr/bin/env python3
"""Compute min/max statistics for requested MODIS channels in CLAVR-x NetCDF files.

Usage:
  python modis_channel_stats.py \
    --pattern '/path/clavrx_MYD021KM.A2019099.0900.061.2019099*.level2.nc' \
    --channels 3 1 2 26 6 7 20 29 31 32 33
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class ChannelStats:
    minimum: float = math.inf
    maximum: float = -math.inf
    count: int = 0

    def update(self, values: Iterable[float]) -> None:
        for value in values:
            if value is None:
                continue
            if isinstance(value, float) and math.isnan(value):
                continue
            self.minimum = min(self.minimum, value)
            self.maximum = max(self.maximum, value)
            self.count += 1


def _candidate_variable_names(channel: int) -> List[str]:
    return [
        f"ch{channel}",
        f"channel_{channel}",
        f"radiance_ch{channel}",
        f"modis_ch{channel}",
        f"band_{channel}",
    ]


def _flatten(values):
    if isinstance(values, (list, tuple)):
        for item in values:
            yield from _flatten(item)
    else:
        yield values


def _find_channel_variable(dataset, channel: int) -> Optional[str]:
    var_names = list(dataset.variables.keys())
    for candidate in _candidate_variable_names(channel):
        if candidate in dataset.variables:
            return candidate

    pattern = re.compile(rf"(?:^|[_\-])(?:ch|channel|band)?0*{channel}(?:$|[_\-])", re.IGNORECASE)
    for name in var_names:
        if pattern.search(name):
            return name

    for name in var_names:
        var = dataset.variables[name]
        attr_candidates = [
            getattr(var, "channel_number", None),
            getattr(var, "band_number", None),
            getattr(var, "modis_channel", None),
        ]
        for attr in attr_candidates:
            if attr is None:
                continue
            try:
                if int(attr) == channel:
                    return name
            except (TypeError, ValueError):
                continue
    return None


def compute_stats(pattern: str, channels: List[int]) -> Dict[int, Tuple[Optional[float], Optional[float], str]]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    try:
        from netCDF4 import Dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "netCDF4 is required to read CLAVR-x .nc files. "
            "Install it with: pip install netCDF4"
        ) from exc

    stats: Dict[int, ChannelStats] = {ch: ChannelStats() for ch in channels}
    resolved_vars: Dict[int, str] = {}

    for path in paths:
        with Dataset(path) as ds:
            for channel in channels:
                var_name = resolved_vars.get(channel)
                if var_name is None:
                    var_name = _find_channel_variable(ds, channel)
                    if var_name:
                        resolved_vars[channel] = var_name
                if var_name is None:
                    continue
                values = ds.variables[var_name][:]
                stats[channel].update(_flatten(values.tolist() if hasattr(values, "tolist") else values))

    output: Dict[int, Tuple[Optional[float], Optional[float], str]] = {}
    for channel in channels:
        stat = stats[channel]
        variable = resolved_vars.get(channel, "<not found>")
        if stat.count == 0:
            output[channel] = (None, None, variable)
        else:
            output[channel] = (stat.minimum, stat.maximum, variable)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute min/max for MODIS channels from CLAVR-x NetCDF files")
    parser.add_argument("--pattern", required=True, help="File glob for input NetCDF files")
    parser.add_argument("--channels", nargs="+", type=int, required=True, help="MODIS channel numbers")
    args = parser.parse_args()

    results = compute_stats(args.pattern, args.channels)
    print("channel,variable,min,max")
    for ch in args.channels:
        min_v, max_v, variable = results[ch]
        min_text = "NA" if min_v is None else f"{min_v:.6g}"
        max_text = "NA" if max_v is None else f"{max_v:.6g}"
        print(f"{ch},{variable},{min_text},{max_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
