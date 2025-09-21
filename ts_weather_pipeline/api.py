import requests
import requests_cache
import pandas as pd
from config.base import CACHE_FILE, CACHE_EXPIRE

requests_cache.install_cache(CACHE_FILE, expire_after=CACHE_EXPIRE)

def fetch_open_meteo_archive(lat, lon, start_date, end_date, hourly_vars=None, timezone="auto", retries=3):
    if hourly_vars is None:
        from config.base import BASE_HOURLY
        hourly_vars = BASE_HOURLY
    
    # Determinar el endpoint según el tipo de variables solicitadas
    from config.base import AIR_QUALITY_HOURLY
    if any(var in hourly_vars for var in AIR_QUALITY_HOURLY):
        # Usar API de calidad del aire
        base = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(hourly_vars),
            "timezone": timezone,
            "domains": "auto"  # Selección automática de dominio (europeo o global)
        }
    else:
        # Usar API de clima histórico
        base = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(hourly_vars),
            "timezone": timezone,
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