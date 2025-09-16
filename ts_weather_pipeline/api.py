import requests
import requests_cache
import pandas as pd
from .config import CACHE_FILE, CACHE_EXPIRE, DEFAULT_HOURLY

requests_cache.install_cache(CACHE_FILE, expire_after=CACHE_EXPIRE)

def fetch_open_meteo_archive(lat, lon, start_date, end_date, hourly_vars=None, timezone="auto", retries=3):
    if hourly_vars is None:
        hourly_vars = DEFAULT_HOURLY

    base = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_vars),
        "timezone": timezone
    }

    for attempt in range(retries):
        r = requests.get(base, params=params, timeout=60)
        if r.status_code == 200:
            data = r.json()
            break
        elif attempt == retries - 1:
            r.raise_for_status()

    hourly = data.get("hourly")
    if hourly is None:
        raise ValueError("No hourly data returned. Response keys: " + ", ".join(data.keys()))

    times = pd.to_datetime(hourly["time"])
    df = pd.DataFrame({k: hourly[k] for k in hourly if k != "time"}, index=times)
    df.index.name = "ds"
    return df
