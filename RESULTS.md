# MODIS Channel Statistics Request

Requested dataset pattern:
`clavrx_MYD021KM.A2019099.0900.061.2019099*.level2.nc`

Requested channels:
`3, 1, 2, 26, 6, 7, 20, 29, 31, 32, 33`

## Current environment status

The requested CLAVR-x NetCDF files are not present in this container, so the min/max values cannot be computed here yet.

## How to compute once files are available

Run:

```bash
python modis_channel_stats.py \
  --pattern '/path/to/clavrx_MYD021KM.A2019099.0900.061.2019099*.level2.nc' \
  --channels 3 1 2 26 6 7 20 29 31 32 33
```

The script prints a CSV-style table with `channel, variable, min, max`.
