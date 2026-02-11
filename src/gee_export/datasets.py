"""Dataset metadata mirrors the original UM6P gee-exporter script."""

from __future__ import annotations

DEFAULT_OPT_URL = "https://earthengine-highvolume.googleapis.com"
NE_COUNTRIES_URL = (
    "https://raw.githubusercontent.com/nvkelso/"
    "natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
)

SENTINEL_CLOUD_THRESHOLD = 15
LANDSAT_CLOUD_THRESHOLD = 15

DATASETS: dict[str, dict] = {
    "landsat": {
        "dataset": "LANDSAT/LE07/C02/T1_TOA",
        "RGB": ["B3", "B2", "B1"],
        "NIR": ["B4"],
        "SI1": ["B5"],
        "SI2": ["B7"],
        "resolution": 30,
        "panchromatic": ["B8"],
        "min": 0.0,
        "max": 0.4,
    },
    "naip": {
        "dataset": "USDA/NAIP/DOQQ",
        "resolution": 0.6,
        "RGB": ["R", "G", "B"],
        "IR": ["N", "R", "G"],
        "NIR": ["N"],
        "panchromatic": None,
        "min": 0.0,
        "max": 255.0,
    },
    "sentinel": {
        "dataset": "COPERNICUS/S2_SR_HARMONIZED",
        "resolution": 10,
        "RGB": ["B4", "B3", "B2"],
        "RE": ["B7", "B6", "B5"],
        "RE4": ["B8A"],
        "NIR": ["B8"],
        "SWIR1": ["B11"],
        "SWIR2": ["B12"],
        "IR": ["B8", "B4", "B3"],
        "panchromatic": None,
        "min": 0.0,
        "max": 4500.0,
    },
    "gwl_fcs30": {
        "dataset": "projects/sat-io/open-datasets/GWL_FCS30",
        "resolution": 30,
        "RGB": None,
        "min": 0,
        "max": 1,
    },
    "soilgrids": {
        "layers": [
            "nitrogen_mean",
            "sand_mean",
            "silt_mean",
            "clay_mean",
            "bdod_mean",
            "cec_mean",
            "cfvo_mean",
            "ocd_mean",
            "phh2o_mean",
        ],
        "scale": 9300,
        "crs": "EPSG:3857",
    },
    "chirps": {
        "dataset": "UCSB-CHG/CHIRPS/DAILY",
        "select": "precipitation",
        "resolution": 5566,
        "min": 0.0,
        "max": 200.0,
    },
    "era5": {
        "dataset": "ECMWF/ERA5_LAND/DAILY_AGGR",
        "select": "temperature_2m",
        "resolution": 11132,
        "min": 180.0,
        "max": 330.0,
    },
    "terraclimate": {
        "dataset": "IDAHO_EPSCOR/TERRACLIMATE",
        "resolution": 4638.3,
        "min": None,
        "max": None,
    },
    "modis_ndvi_evi": {
        "dataset": "MODIS/061/MOD13Q1",
        "select": ["NDVI", "EVI"],
        "resolution": 250,
        "NDVI": ["NDVI"],
        "EVI": ["EVI"],
        "min": -2000,
        "max": 10000,
    },
    "modis_lai_fapar": {
        "dataset": "MODIS/061/MCD15A3H",
        "select": ["Lai", "Fpar"],
        "resolution": 500,
        "LAI": ["Lai"],
        "fAPAR": ["Fpar"],
        "min": 0,
        "max": 10000,
    },
    "modis_lst": {
        "dataset": "MODIS/061/MOD11A1",
        "select": ["LST_Day_1km", "LST_Night_1km"],
        "resolution": 1000,
        "Day_LST": ["LST_Day_1km"],
        "Night_LST": ["LST_Night_1km"],
        "min": 7500,
        "max": 65535,
    },
    "modis_et": {
        "dataset": "MODIS/061/MOD16A2",
        "select": ["ET"],
        "resolution": 500,
        "ET": ["ET"],
        "min": 0,
        "max": 1000,
    },
    "sentinel1": {
        "dataset": "COPERNICUS/S1_GRD",
        "resolution": 10,
        "VV": ["VV"],
        "VH": ["VH"],
        "min": -30,
        "max": 0,
        "filter": {"instrumentMode": "IW"},
    },
    "srtm": {
        "dataset": "USGS/SRTMGL1_003",
        "resolution": 30,
        "elevation": ["elevation"],
        "min": 0,
        "max": 4000,
    },
    "jrc_gsw": {
        "dataset": "JRC/GSW1_4/GlobalSurfaceWater",
        "select": ["occurrence", "recurrence", "seasonality"],
        "resolution": 30,
        "occurrence": ["occurrence"],
        "recurrence": ["recurrence"],
        "seasonality": ["seasonality"],
        "min": 0,
        "max": 100,
    },
    "worldcover": {
        "dataset": "ESA/WorldCover/v100",
        "select": ["Map"],
        "resolution": 10,
        "Map": ["Map"],
        "min": 0,
        "max": 100,
    },
    "gpw_population": {
        "dataset": "CIESIN/GPWv411/GPW_Population_Density",
        "select": "population_density",
        "resolution": 927.67,
        "population_density": ["population_density"],
        "min": 0,
        "max": 1000,
    },
    "nasapower": {
        "dataset": "NASA/POWER/MONTHLY",
        "select": ["T2M", "PRECTOT"],
        "resolution": 50000,
        "min": None,
        "max": None,
    },
}

SPLIT_TILE_DATASETS = {
    "modis_ndvi_evi",
    "modis_lai_fapar",
    "modis_lst",
    "modis_et",
    "sentinel1",
    "srtm",
    "jrc_gsw",
    "worldcover",
}

REGION_SUMMARY_DATASETS = {
    "chirps",
    "era5",
    "terraclimate",
    "gpw_population",
}

RAW_EXPORT_DATASETS = {
    "gwl_fcs30",
    "chirps",
    "era5",
    "terraclimate",
}
