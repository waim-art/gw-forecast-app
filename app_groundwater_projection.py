# app_groundwater_projection.py
# Groundwater projection app: Ouranos SSP hourly → daily → spatial baseline → Δ-level RF
from __future__ import annotations
import os, json, math, time, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
import requests
import joblib
from joblib import load as joblib_load
from math import pi
from pykrige.uk import UniversalKriging
from pyproj import CRS, Transformer

import streamlit as st
import folium

warnings.filterwarnings("ignore", message=".*to_datetimeindex will default.*")

# -------------------------------------------------------------------
# Paths / artifacts
# -------------------------------------------------------------------
DATA_ROOT = Path("data")   # expects data/ssp1, ssp2, ssp3, ssp5 with *_clean.txt

# Anchor paths to this file's folder so Streamlit runs from anywhere
APP_DIR = Path(__file__).resolve().parent
DP      = APP_DIR / "data_products"
DP.mkdir(parents=True, exist_ok=True)
REGK_JOBLIB = DP / "regkrig_baseline.joblib"
REGK_META   = DP / "regkrig_meta.json"
REGK_BIAS   = DP / "regkrig_bias.joblib"
REGKRIG_GSE_MODEL_P = DP / "regkrig_gse.joblib"
REGKRIG_GSE_META_P  = DP / "regkrig_gse_meta.json"

# Legacy: reg-kriging baseline (residual-corrected spatial baseline)
REGK_CFG   = DP / "regkrig_baseline_config.json" # old config
REGK_TRAIN = DP / "regkrig_baseline_train.csv"
if not REGK_TRAIN.exists():
    alt = DP / "krig_baseline_train.csv"
    if alt.exists():
        REGK_TRAIN = alt

# Δ-level RF (Ouranos) artifacts
DL_MODEL  = DP / "levelchange_rf_ouranos.joblib"
DL_FEATS  = DP / "levelchange_rf_ouranos_features.json"
DL_PWBias = DP / "levelchange_rf_ouranos_perwell_train_bias.csv"  # optional (well,bias)

# Optional: RF baseline (Ouranos) for local slow drift (clicked point)
RF_MODEL  = DP / "model_rf_baseline_ouranos.joblib"
RF_FEATS  = DP / "model_rf_baseline_ouranos_features.json"

# Optional: precomputed future RF predictions by well (enables per-day reg-kriging)
RF_FUTURE_WELLS = DP / "rf_baseline_future_by_well.csv"  # Date,Well_Name,pred_rf

# Optional: river emulator
RIV_MODEL = DP / "river_emulator.joblib"
RIV_FEATS = DP / "river_emulator_features.json"
RIV_TGT   = DP / "river_emulator_target.json"

# === Re-anchor artifacts ===
REANCHOR_MODEL_PATH  = DP / "reanchor_monthly.joblib"
REANCHOR_FEATS_PATH  = DP / "reanchor_monthly_features.json"
REANCHOR_SCALER_PATH = DP / "reanchor_monthly_scaler.joblib"

# Ouranos variable map (hourly names)
VAR_MAP = {"tas":"tas","pr":"pr","hurs":"hurs","huss":"huss",
           "uas":"uas","vas":"vas","rsds":"rsds","rlds":"rlds","ps":"ps","prfr":"pr"}

# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
def slog(msg: str):
    st.write(f"🟢 {msg}")

def to_local_utm(lat, lon):
    lat_m = float(lat); lon_m = float(lon)
    utm = int((lon_m + 180)//6) + 1
    south = lat_m < 0
    crs = CRS.from_dict({"proj":"utm","zone":utm,"south":south,"ellps":"WGS84"})
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = tf.transform(lon, lat)
    return float(x), float(y)


def read_url_list(p: Path) -> List[str]:
    if not p.exists(): return []
    return [ln.strip() for ln in p.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]

def open_ds(url: str, retries=3, wait=1.5) -> xr.Dataset:
    last = None
    for k in range(retries):
        try:
            return xr.open_dataset(url, engine="netcdf4")
        except Exception as e:
            last = e
            time.sleep(wait * (k + 1))
    raise RuntimeError(f"Failed to open dataset after {retries} tries: {url}\nlast_error={last}")

def to_datetime_index(time_var: xr.DataArray) -> pd.DatetimeIndex:
    try:
        return xr.coding.cftimeindex.CFTimeIndex(time_var.values).to_datetimeindex(
            unsafe=True, time_unit="ns"
        )
    except Exception:
        return pd.to_datetime(time_var.values, errors="coerce")

def nearest_2d_idx(lat2d: np.ndarray, lon2d: np.ndarray, lat0: float, lon0: float) -> Tuple[int,int]:
    lat_r, lon_r = np.radians(lat2d), np.radians(lon2d)
    lat0r, lon0r = math.radians(lat0), math.radians(lon0)
    dlat = lat_r - lat0r; dlon = lon_r - lon0r
    a = np.sin(dlat/2)**2 + np.cos(lat_r)*np.cos(lat0r)*np.sin(dlon/2)**2
    dist = 2*np.arcsin(np.sqrt(a))
    j,i = np.unravel_index(np.nanargmin(dist), dist.shape)
    return int(j), int(i)

def haversine_np(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance in meters between two points on the earth."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371000 * c # Earth radius in meters

def zero_mean(s: pd.Series) -> pd.Series:
    """Center a series to zero mean (ignoring NaNs)."""
    if s is None or len(s) == 0: 
        return s
    m = float(pd.to_numeric(s, errors="coerce").mean())
    return s - m

def clamp_levels_against_dem(level: pd.Series, dem: Optional[float], pad: float = 1.0, max_depth: float = 50.0) -> pd.Series:
    """
    Optional sanity clamp: keep level within [dem-max_depth, dem+pad] if DEM exists.
    pad>0 allows water to be slightly above ground in wet periods.
    """
    if dem is None or np.isnan(dem):
        return level
    lo = dem - float(max_depth)
    hi = dem + float(pad)
    return level.clip(lower=lo, upper=hi)

# --- Adaptive max-depth helper (AWwID-informed) ------------------------------
from math import radians, sin, cos, atan2, sqrt

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*atan2(sqrt(a), sqrt(1 - a))

def _load_awwid_train_table() -> Optional[pd.DataFrame]:
    """
    Try to load the same table you used to train reg-kriging baseline
    (columns: WellId, lat, lon, GSE_m, level_med, ...).
    We’ll compute depth = max(GSE_m - level_med, 0).
    """
    # Prefer your curated file if you already save one; otherwise use the reg-krig train csv you created
    candidates = [
        "data_products/awwid_krig_train.csv",              # your earlier export
        "data_products/awwid_krig_train.parquet",
        "data_products/awwid_krig_train.feather",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return pd.read_parquet(p) if p.endswith(".parquet") else (
                       pd.read_feather(p) if p.endswith(".feather") else pd.read_csv(p))
            except Exception:
                pass
    return None

def estimate_local_max_depth_m(lat: float, lon: float,
                               radius_km: float = 5.0,
                               pct: float = 98.0,
                               hard_min: float = 10.0,
                               hard_max: float = 80.0) -> float:
    """
    Use nearby wells’ depths (GSE - level_med) to pick a realistic clamp bound.
    Returns a robust percentile (default P95), constrained to [hard_min, hard_max].
    Falls back to 50 m if we can’t compute it.
    """
    tbl = _load_awwid_train_table()
    if tbl is None or not {"lat","lon","GSE_m","level_med"}.issubset(tbl.columns):
        return 50.0
    depth = (pd.to_numeric(tbl["GSE_m"], errors="coerce") - pd.to_numeric(tbl["level_med"], errors="coerce")).clip(lower=0)
    good = tbl.assign(_depth=depth).dropna(subset=["lat","lon","_depth"]).copy()
    dkm = good.apply(lambda r: _haversine_km(lat, lon, float(r["lat"]), float(r["lon"])), axis=1)
    local = good.loc[dkm.values <= radius_km, "_depth"].astype(float)
    if len(local) < 12: local = good.loc[dkm.values <= 10.0, "_depth"].astype(float)
    if local.empty: return 50.0
    est = float(np.nanpercentile(local, pct))
    return float(np.clip(est, hard_min, hard_max))

def _doy_sin_cos(dt: pd.Timestamp) -> tuple[float, float]:
    doy = dt.dayofyear
    ang = 2.0 * pi * (doy / 365.25)
    return float(np.sin(ang)), float(np.cos(ang))

def _month_agg(series: pd.Series) -> float:
    # Use the mean over the currently selected month window you’ve already built
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan

def _slope_limit(series: pd.Series, max_change_per_day: float) -> pd.Series:
    """Limit day-to-day change (m/day) to avoid sudden jumps."""
    if series.empty: return series
    out = series.copy()
    for i in range(1, len(out)):
        delta = out.iloc[i] - out.iloc[i-1]
        if delta > max_change_per_day:
            out.iloc[i] = out.iloc[i-1] + max_change_per_day
        elif delta < -max_change_per_day:
            out.iloc[i] = out.iloc[i-1] - max_change_per_day
    return out


# Reg-kriging: load & helpers
# -------------------------------------------------------------------
def _sig(p: Path) -> str:
    try: return f"{p.stat().st_mtime_ns}-{p.stat().st_size}"
    except Exception: return "na"

@st.cache_resource
def load_regkrig_bundle(sig: str):
    if not REGK_JOBLIB.exists() or not REGK_META.exists():
        missing = []
        if not REGK_JOBLIB.exists(): missing.append(str(REGK_JOBLIB.resolve()))
        if not REGK_META.exists():   missing.append(str(REGK_META.resolve()))
        raise FileNotFoundError("Missing reg-krig artifacts:\n" + "\n".join(missing))
    mdl = joblib.load(REGK_JOBLIB)
    meta = json.loads(REGK_META.read_text(encoding="utf-8"))
    return {"model": mdl, "meta": meta}

@st.cache_data
def load_regkrig_train_table() -> Optional[pd.DataFrame]:
    if not REGK_TRAIN.exists(): 
        return None
    df = pd.read_csv(REGK_TRAIN)
    need = {"WellId","lat","lon","x_m","y_m"}
    if not need.issubset(df.columns):
        # tolerate older files; derive x_m,y_m if needed
        if {"lat","lon"}.issubset(df.columns):
            x, y = zip(*[to_local_utm(r.lat, r.lon) for _,r in df.iterrows()])
            df["x_m"], df["y_m"] = x, y
        else:
            return None
    # canonical well name/id
    if "Well_Name" not in df.columns and "WellId" in df.columns:
        df["Well_Name"] = df["WellId"].astype(str)
    elif "Well_Name" not in df.columns:
        return None # Cannot proceed without a well identifier
    return df[["Well_Name","x_m","y_m"]].dropna().drop_duplicates()

def regkrig_from_daily_well_preds(lat: float, lon: float,
                                  df_well: pd.DataFrame,
                                  stamp: str = "") -> pd.Series:
    # This function is now unused because the logic was moved into the main pipeline
    # and depends on the legacy loader. It can be removed or refactored later.
    # For now, raising an error to indicate it's deprecated.
    raise NotImplementedError("regkrig_from_daily_well_preds is deprecated.")
    b = load_regkrig_bundle(stamp) # This will now use the new loader logic
    if b is None: raise RuntimeError("Reg-kriging artifacts not found.")
    cfg, train, drifts, anis_kw, tf = (b["cfg"], b["train"], b["drifts"], b["anis_kw"], b["tf"])

    xp, yp = tf.transform(lon, lat)
    xp, yp = float(xp), float(yp)

    if "Well_Name" not in train.columns:
        raise RuntimeError("Train table missing Well_Name for alignment with well predictions.")
    base = train[["Well_Name","x_m","y_m"]].copy()
    if drifts:
        for c in drifts:
            if c not in train.columns:
                raise RuntimeError(f"Drift '{c}' missing in train table.")
        base = base.join(train[drifts])

    out = []
    days = pd.to_datetime(df_well["Date"].unique()).sort_values()
    prog = st.progress(0.0, text="Per-day reg-kriging baseline (slow)…")
    for k, d in enumerate(days, start=1):
        sub = pd.merge(base, df_well.loc[pd.to_datetime(df_well["Date"])==d, ["Well_Name","pred_rf"]],
                       on="Well_Name", how="inner")
        if len(sub) < 5:
            out.append((d, np.nan))
            prog.progress(k/len(days)); continue

        uk = UniversalKriging(
            sub["x_m"].to_numpy(float), sub["y_m"].to_numpy(float), sub["pred_rf"].to_numpy(float),
            variogram_model=cfg.get("variogram_model","spherical"),
            variogram_parameters=cfg.get(
                "variogram_parameters",
                [np.nanvar(sub["pred_rf"]), (sub["x_m"].max()-sub["x_m"].min())*0.3, 0.0]),
            drift_terms=(["specified"] if drifts else None),
            specified_drift=([sub[c].to_numpy(float) for c in drifts] if drifts else None),
            enable_plotting=False, verbose=False, **anis_kw
        )

        drift_point = None
        if drifts:
            dx = sub["x_m"].to_numpy(float) - xp
            dy = sub["y_m"].to_numpy(float) - yp
            j = int(np.nanargmin(dx*dx + dy*dy))
            drift_point = [np.array([float(sub.iloc[j][c])]) for c in drifts]

        zhat, _ = uk.execute(
            "points",
            np.array([xp]), np.array([yp]),
            specified_drift_arrays=drift_point
        )
        out.append((d, float(zhat[0])))
        prog.progress(k/len(days))
    prog.empty()

    ser = pd.Series(dict(out)).sort_index()
    ser.index.name = "Date"
    ser.name = "pred_baseline"
    return ser

def regkrig_from_daily_well_preds_fast(train_xy: pd.DataFrame,
                                       daily_well_rf: pd.DataFrame,
                                       xp: float, yp: float) -> pd.Series:
    """
    Krige each day's field from well RF predictions.
    daily_well_rf columns: Date, Well_Name, pred_rf
    Returns pd.Series indexed by Date (float masl).
    """
    # fast nearest-drift (no explicit drift terms); UK on-the-fly per day
    days = pd.to_datetime(daily_well_rf["Date"].unique())
    out = []
    for d in days:
        sub = daily_well_rf.loc[pd.to_datetime(daily_well_rf["Date"]) == d, ["Well_Name","pred_rf"]]
        sub = sub.merge(train_xy, on="Well_Name", how="inner").dropna()
        if len(sub) < 6:
            out.append((d, np.nan))
            continue
        uk = UniversalKriging(
            sub["x_m"].to_numpy(float), sub["y_m"].to_numpy(float), sub["pred_rf"].to_numpy(float),
            variogram_model="spherical", enable_plotting=False, verbose=False
        )
        zhat, _ = uk.execute("points", np.array([xp]), np.array([yp]))
        out.append((d, float(zhat[0])))
    s = pd.Series(dict(out)).sort_index()
    s.index.name = "Date"; s.name = "baseline_from_daily_krig"
    return s
# -------------------------------------------------------------------
# DEM → depth
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def get_dem_open_elevation(lat: float, lon: float) -> Optional[float]:
    try:
        r = requests.get("https://api.open-elevation.com/api/v1/lookup",
                         params={"locations": f"{lat},{lon}"}, timeout=20)
        r.raise_for_status()
        return float(r.json()["results"][0]["elevation"])
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def get_dem_open_elevation_bulk(points: List[Tuple[float,float]]) -> List[Optional[float]]:
    """Bulk fetch DEM for a list of (lat, lon) points."""
    locs = "|".join([f"{lat},{lon}" for lat,lon in points])
    try:
        r = requests.post("https://api.open-elevation.com/api/v1/lookup",
                          json={"locations": [{"latitude":lat, "longitude":lon} for lat,lon in points]}, timeout=30)
        r.raise_for_status()
        return [res.get("elevation") for res in r.json()["results"]]
    except Exception:
        return [None] * len(points)
# -------------------------------------------------------------------
# Ouranos discovery & fetch

# --- Top-level model loading ---
REGKRIG_GSE_MODEL = None
REGKRIG_GSE_META  = None
try:
    if REGKRIG_GSE_MODEL_P.exists() and REGKRIG_GSE_META_P.exists():
        REGKRIG_GSE_MODEL = joblib.load(REGKRIG_GSE_MODEL_P)
        REGKRIG_GSE_META  = json.loads(REGKRIG_GSE_META_P.read_text())
        print("📦 GSE reg-kriging model loaded.")
except Exception as e:
    print(f"⚠️ GSE reg-kriging not available ({e}); will fall back to Open-Elevation if needed.")

try:
    if REANCHOR_MODEL_PATH.exists() and REANCHOR_FEATS_PATH.exists():
        reanchor_model = joblib_load(REANCHOR_MODEL_PATH)
        reanchor_feats = json.loads(Path(REANCHOR_FEATS_PATH).read_text())
        reanchor_scaler = joblib_load(REANCHOR_SCALER_PATH)
        print("📦 Monthly re-anchor model loaded.")
except Exception as e:
    print(f"⚠️ Monthly re-anchor model not available ({e}).")
# -------------------------------------------------------------------
def _discover_var_lists(ssp_folder: Path) -> Dict[str, Path]:
    want = ["tas","pr","hurs","huss","uas","vas","rsds","rlds","ps","prfr"]
    files = sorted(ssp_folder.glob("*.txt"),
                   key=lambda p: (0 if p.stem.endswith("_clean") else 1, p.name))
    out: Dict[str, Path] = {}
    for f in files:
        low = f.name.lower()
        for k in want:
            if k in low and k not in out:
                out[k] = f
    if "pr" not in out and "prfr" in out:
        out["pr"] = out["prfr"]
    return {k: v for k, v in out.items() if k in VAR_MAP}

def concat_hourly_slice(urls: List[str], varname: str, j: int, i: int,
                        date_min: str|None, date_max: str|None) -> pd.Series:
    frames=[]
    for u in urls:
        ds = open_ds(u)
        if varname not in ds.variables:
            ds.close(); continue
        if "y" in ds.dims and "x" in ds.dims:
            v = ds[varname].isel(y=j, x=i); t = ds["time"]
        else:
            v = ds[varname][:, j, i]; t = v["time"] if "time" in v.coords else ds["time"]
        ti = to_datetime_index(t)
        s = pd.Series(np.asarray(v.values), index=ti, name=varname)
        ds.close()
        if date_min: s = s[s.index >= pd.to_datetime(date_min)]
        if date_max: s = s[s.index <= pd.to_datetime(date_max)]
        frames.append(s)
    if not frames: return pd.Series(dtype=float, name=varname)
    ser = pd.concat(frames).sort_index()
    ser = ser[~ser.index.duplicated(keep="first")]
    return ser

def fetch_hourly_ouranos(lat: float, lon: float, ssp_folder: Path,
                         date_min: str, date_max: str,
                         max_files_per_var: int = 0) -> pd.DataFrame:
    """
    Fetches and combines hourly Ouranos data for multiple variables in parallel.
    """
    lists = _discover_var_lists(ssp_folder)
    if not lists:
        raise RuntimeError(f"No *_clean.txt lists found in {ssp_folder}")

    # Step 1: Probe grid sequentially to find nearest indices (j, i)
    urls0 = read_url_list(next(iter(lists.values())))
    if not urls0: raise RuntimeError("URL list is empty in SSP folder.")
    ds0 = open_ds(urls0[0])
    lat2d = ds0["lat"].values if "lat" in ds0 else ds0["latitude"].values
    lon2d = ds0["lon"].values if "lon" in ds0 else ds0["longitude"].values
    ds0.close()
    j, i = nearest_2d_idx(lat2d, lon2d, lat, lon)
    st.caption(f"Nearest Ouranos grid cell: (j={j}, i={i})")

    # Step 2: Submit fetch tasks to a thread pool for parallel execution
    parts = []
    prog = st.progress(0.0, text="Submitting Ouranos fetch tasks…")
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for short, path in lists.items():
            urls = read_url_list(path)
            if max_files_per_var and len(urls) > max_files_per_var:
                urls = urls[:max_files_per_var]
            if not urls:
                continue
            var = VAR_MAP[short]
            # Submit the task and store the future with its variable name
            future = executor.submit(concat_hourly_slice, urls, var, j, i, date_min, date_max)
            futures[future] = var

        # Step 3: Collect results as they complete
        completed_count = 0
        total_tasks = len(futures)
        for future in as_completed(futures):
            var_name = futures[future]
            try:
                result_series = future.result()
                if not result_series.empty:
                    parts.append(result_series)
                st.write(f"✓ Completed fetching: {var_name}")
            except Exception as e:
                st.warning(f"Could not fetch or process {var_name}: {e}")
            
            completed_count += 1
            prog.progress(completed_count / total_tasks, text=f"Fetching Ouranos hourly ({completed_count}/{total_tasks})…")

    prog.empty()

    if not parts: raise RuntimeError("No hourly variables were successfully extracted.")
    
    # Step 4: Combine and process final DataFrame
    H = pd.concat(parts, axis=1).sort_index()
    if "uas" in H or "vas" in H:
        u = H.get("uas", 0.0)
        v = H.get("vas", 0.0)
        H["wind_ms"] = np.sqrt(u**2 + v**2)
    if "pr" in H:
        H["pr_mm_h"] = H["pr"] * 3600.0
        
    return H

def hourly_to_daily(H: pd.DataFrame) -> pd.DataFrame:
    out={}
    if "tas" in H:
        tC = H["tas"] - 273.15
        out["tmean_c_ouranos"] = tC.resample("D").mean()
        out["tmax_c_ouranos"]  = tC.resample("D").max()
        out["tmin_c_ouranos"]  = tC.resample("D").min()
    if "pr_mm_h" in H: out["precip_mm_ouranos"] = H["pr_mm_h"].resample("D").sum()
    if "hurs" in H:    out["rh_mean_pct"]       = H["hurs"].resample("D").mean()
    if "huss" in H:    out["huss_mean"]         = H["huss"].resample("D").mean()   # <-- add missing 28th feat
    if "wind_ms" in H: out["wind_ms"]           = H["wind_ms"].resample("D").mean()
    if "rsds" in H:    out["rsds_MJ_m2_d"]      = (H["rsds"].resample("D").mean() * 86400.0 / 1e6)
    if "rlds" in H:    out["rlds_W_m2_mean"]    = H["rlds"].resample("D").mean()
    if "ps" in H:      out["ps_mean_Pa"]        = H["ps"].resample("D").mean()

    D = pd.DataFrame(out)
    D.index.name = "Date"
    doy = D.index.dayofyear
    D["doy"]     = doy
    D["doy_sin"] = np.sin(2*np.pi*doy/365.25)
    D["doy_cos"] = np.cos(2*np.pi*doy/365.25)

    for c in ["tmean_c_ouranos","tmax_c_ouranos","tmin_c_ouranos",
              "precip_mm_ouranos","rh_mean_pct","wind_ms"]:
        if c in D: D[c+"_d1"] = D[c].diff(1)
    if "precip_mm_ouranos" in D:
        D["precip_sum3"] = D["precip_mm_ouranos"].shift(1).rolling(3,min_periods=1).sum()
        D["precip_sum7"] = D["precip_mm_ouranos"].shift(1).rolling(7,min_periods=1).sum()
    if "tmean_c_ouranos" in D:
        D["tmean_mean7"] = D["tmean_c_ouranos"].shift(1).rolling(7,min_periods=1).mean()

    return D.reset_index()

# -------------------------------------------------------------------
# River emulator — name-normalized output for Δ-RF
# -------------------------------------------------------------------
def maybe_predict_river(D_daily: pd.DataFrame) -> Optional[pd.Series]:
    if not (RIV_MODEL.exists() and RIV_FEATS.exists() and RIV_TGT.exists()):
        return None
    feats = json.loads(RIV_FEATS.read_text())
    tgt  = json.loads(RIV_TGT.read_text()).get("target", "river_level_m")

    X = (
        D_daily
        .reindex(columns=feats)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(float)
    )
    model = joblib.load(RIV_MODEL)
    pred  = model.predict(X)

    try:
        dl_feats = json.loads(DL_FEATS.read_text()) if DL_FEATS.exists() else []
    except Exception:
        dl_feats = []
    preferred_name = "river_level_m" if "river_level_m" in dl_feats else tgt
    return pd.Series(pred, index=pd.to_datetime(D_daily["Date"]), name=preferred_name)

# -------------------------------------------------------------------
# Δ-level RF rollout anchored on baseline (with drift-guard)
# -------------------------------------------------------------------
@st.cache_resource
def _load_delta_rf():
    if not (DL_MODEL.exists() and DL_FEATS.exists()):
        raise RuntimeError("Missing Δ-level RF artifacts in data_products/.")
    model = joblib.load(DL_MODEL)
    feats = json.loads(DL_FEATS.read_text())
    pw = None
    if DL_PWBias.exists():
        try: pw = pd.read_csv(DL_PWBias)
        except: pw = None
    return model, feats, pw

def _rolling_zero_mean(arr: np.ndarray, win: int = 90) -> np.ndarray:
    """Subtract rolling mean to keep long-horizon drift near 0 (soft guard)."""
    if win <= 1 or len(arr) < win: 
        return arr - np.nanmean(arr)
    s = pd.Series(arr)
    mu = s.rolling(win, min_periods=1, center=True).mean().to_numpy()
    return arr - mu

def rollout_delta_from_baseline(
    baseline_series: pd.Series,
    D: pd.DataFrame,
    feats: List[str],
    model,
    river_daily: Optional[pd.Series],
    donor_well: Optional[str],
    perwell_bias_tbl: Optional[pd.DataFrame],
    drift_guard_win: int = 90
) -> pd.DataFrame:
    df = D.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Inject river features if present
    bias = 0.0
    if donor_well and perwell_bias_tbl is not None and "well" in perwell_bias_tbl.columns:
        row = perwell_bias_tbl.loc[perwell_bias_tbl["well"] == donor_well]
        if not row.empty and "bias" in row.columns:
            try: bias = float(row["bias"].iloc[0])
            except: bias = 0.0

    if river_daily is not None:
        rname = river_daily.name or "river_level_m"
        df = df.merge(river_daily.rename_axis("Date").reset_index(), on="Date", how="left")
        if rname in df.columns:
            if f"{rname}_lag1" not in df.columns:
                df[f"{rname}_lag1"] = df[rname].shift(1)
            if f"{rname}_roll7" not in df.columns:
                df[f"{rname}_roll7"] = df[rname].shift(1).rolling(7, min_periods=1).mean()

    # Feature gap diagnostics (we’ll still fill with 0)
    missing = [c for c in feats
               if c not in ["dlevel_lag1","dlevel_lag3","dlevel_lag7","level_masl_lag1_well"]
               and c not in df.columns]
    if missing:
        st.warning(f"Δ-RF expects {len(feats)} features; {len(missing)} not found "
                   f"in daily drivers and will default to 0: {missing}")

    # Rollout
    pred_delta_raw, pred_level = [], []
    d_lags = [0.0, 0.0, 0.0]
    base_series = baseline_series.reindex(pd.to_datetime(df["Date"])).fillna(method="ffill").fillna(method="bfill")
    level_prev = float(base_series.iloc[0])
    prog = st.progress(0.0, text="Rolling Δ-level (RF)…")
    n = len(df)

    for t in range(n):
        row = df.iloc[t]
        x_vals = []
        for c in feats:
            if c == "dlevel_lag1": x_vals.append(d_lags[0])
            elif c == "dlevel_lag3": x_vals.append(d_lags[1])
            elif c == "dlevel_lag7": x_vals.append(d_lags[2])
            elif c == "level_masl_lag1_well": x_vals.append(level_prev)
            else:
                v = row.get(c, 0.0)
                try: v = float(v) if (v is not None and not pd.isna(v)) else 0.0
                except Exception: v = 0.0
                x_vals.append(v)
        X = np.array([x_vals], dtype=float)

        d = float(model.predict(X)[0]) - bias
        pred_delta_raw.append(d)

        # (we will apply drift-guard after predicting the whole sequence)
        level_prev = level_prev + d
        pred_level.append(level_prev)
        d_lags = [d, d_lags[0], d_lags[1]]
        prog.progress((t+1)/n if n else 1.0)
    prog.empty()

    # Soft drift-guard on deltas, then re-integrate on the given baseline
    pred_delta = _rolling_zero_mean(np.array(pred_delta_raw), win=drift_guard_win)
    base0 = float(base_series.iloc[0])
    level = base0 + np.cumsum(pred_delta)

    out = df[["Date"]].copy()
    out["pred_delta_raw"] = pred_delta_raw
    out["pred_delta"] = pred_delta
    out["pred_level"] = level
    return out

# -------------------------------------------------------------------
# Local RF baseline slow drift (point-based)
# -------------------------------------------------------------------
@st.cache_resource
def _load_rf_baseline_local():
    if not (RF_MODEL.exists() and RF_FEATS.exists()):
        return None, None
    model = joblib.load(RF_MODEL)
    feats = json.loads(RF_FEATS.read_text())
    return model, feats

def rf_baseline_local_series(D: pd.DataFrame) -> Optional[pd.Series]:
    model, feats = _load_rf_baseline_local()
    if model is None: return None
    X = D.reindex(columns=feats).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    y = model.predict(X.to_numpy(float))
    return pd.Series(y, index=pd.to_datetime(D["Date"]), name="rf_drift_local")

# -------------------------------------------------------------------
# Hourly downscale (shape proxy)
# -------------------------------------------------------------------
def et_proxy(rsds=None, tas=None, hurs=None, wind=None):
    parts=[]
    if rsds is not None: parts.append(np.maximum(rsds,0.0))
    if tas  is not None: parts.append(np.maximum(tas-273.15,0.0))
    if hurs is not None: parts.append(np.maximum(100-hurs,0.0))
    if wind is not None: parts.append(np.maximum(wind,0.0))
    if not parts: return None
    S=np.zeros_like(parts[0],float)
    for p in parts:
        s=(p-np.nanmean(p))/(np.nanstd(p)+1e-9); S+=np.nan_to_num(s)
    S=np.maximum(S,0.0); m=np.nanmean(S)
    return S/(m if (m and np.isfinite(m)) else 1.0)

def shape_for_day(Hday: pd.DataFrame) -> np.ndarray:
    pr = Hday.get("pr_mm_h"); rs = Hday.get("rsds"); ta=Hday.get("tas"); hu=Hday.get("hurs"); wi=Hday.get("wind_ms")
    etp = et_proxy(rsds=rs, tas=ta, hurs=hu, wind=wi)
    if etp is None:
        H=len(Hday); h=np.arange(H); I=-np.cos(2*np.pi*(h-15)/24.0)
        I -= np.mean(I); L1=np.sum(np.abs(I))
        return I/(L1 if L1 else 1.0)
    I = pr.fillna(0.0).to_numpy() if pr is not None else np.zeros(len(Hday))
    I = I - etp
    I -= np.mean(I); L1 = np.sum(np.abs(I))
    return I/(L1 if L1 else 1.0)

def downscale_to_hourly(daily: pd.DataFrame, H: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily.sort_values("Date").set_index("Date")
    rows=[]
    for day, dday in daily["pred_delta"].items():
        Hday = H.loc(H.index.date == day.date()) if hasattr(H.index, "date") else H.loc[day:day+pd.Timedelta(hours=23)]
        if isinstance(Hday, pd.DataFrame) and not Hday.empty:
            pass
        else:
            idx = pd.date_range(day, periods=24, freq="H")
            rows.append(pd.DataFrame({"level_masl_hourly": (daily.loc[day,"pred_level"] + np.cumsum(np.full(24, dday/24.0)) - dday/24.0)}, index=idx))
            continue
        s = shape_for_day(Hday)
        dh = dday * s
        base = float(daily.loc[day, "pred_level"])
        lvls = base + np.cumsum(dh) - dh[0]
        out = pd.DataFrame({"level_masl_hourly": lvls}, index=Hday.index)
        rows.append(out)
    return pd.concat(rows).sort_index()

# --- Re-anchor monthly: load + predict, then broadcast to daily ---
def _add_degree_days(df, tmean="tmean_c_ouranos"):
    # 30-day cumulative heating/cooling degree-days (base 18°C)
    ddh = (18 - df[tmean]).clip(lower=0)
    ddc = (df[tmean] - 18).clip(lower=0)
    df["heating_dd30"] = ddh.rolling(30, min_periods=1).sum()
    df["cooling_dd30"] = ddc.rolling(30, min_periods=1).sum()
    return df

def _month_cyc(df, col="Date"):
    m = pd.to_datetime(df[col]).dt.month
    df["month_sin"] = np.sin(2*np.pi*m/12.0)
    df["month_cos"] = np.cos(2*np.pi*m/12.0)
    return df

@st.cache_resource
def load_reanchor_artifacts(prefix="data_products/reanchor_monthly"):
    model   = joblib.load(f"{prefix}.joblib")
    scaler  = joblib.load(f"{prefix}_scaler.joblib")
    feats   = json.loads(Path(f"{prefix}_features.json").read_text())
    meta    = json.loads(Path(f"{prefix}_meta.json").read_text())
    return model, scaler, feats, meta

def predict_reanchor_daily(
    df_daily: pd.DataFrame,
    lat: float,
    lon: float,
    dem_elev_m: float,
    slope_deg_dem: float,
    *,
    model_prefix: str = "data_products/reanchor_monthly",
    ema_span_days: int = 14,
):
    """
    df_daily must contain: Date (datetime), and columns for climate/river drivers.
    Returns: anchor_daily (pd.Series, meters), diag (dict)
    """
    df = df_daily.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # 1) Load artifacts & align feature order
    model, scaler, feat_list, meta = load_reanchor_artifacts(model_prefix)

    # 2) Prepare feature set
    df = _month_cyc(df, "Date")
    for base in ["precip_mm_ouranos", "tmean_c_ouranos", "river_level_m"]:
        if base in df.columns:
            for w in (7, 30):
                df[f"{base}_roll{w}"] = df[base].rolling(w, min_periods=1).mean()

    # Add statics
    df["dem_elev_m"] = dem_elev_m
    df["slope_deg_dem"] = slope_deg_dem
    df["lat"] = lat
    df["lon"] = lon

    # 3) Make sure all expected features exist (fill if needed)
    for f in feat_list:
        if f not in df.columns:
            df[f] = 0.0

    # 4) Predict on daily features, then broadcast to monthly anchor
    X_daily = df[feat_list]
    X_scaled = scaler.transform(X_daily)
    anchor_monthly = model.predict(X_scaled)

    # 5) Create daily series, reindex to match input, and smooth
    anchor_daily = pd.Series(anchor_monthly, index=df["Date"])
    anchor_daily = anchor_daily.reindex(df_daily["Date"].values)
    if ema_span_days and ema_span_days > 1:
        anchor_daily = anchor_daily.ewm(span=ema_span_days, adjust=False).mean()

    return anchor_daily.values, {} # Return numpy array and empty diag for now

# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
st.set_page_config(page_title="GW Projection — spatial+temporal (Ouranos)", layout="wide")
st.title("🌎 Groundwater Projection — reg-kriging baseline + RF slow drift + Δ-RF")

with st.sidebar:
    st.header("Scenario & Dates")
    ssp_dirs = [p for p in (DATA_ROOT).glob("ssp*") if p.is_dir()]
    if not ssp_dirs:
        st.error("No SSP folders found in ./data/. Add data/ssp2, data/ssp3, etc. with *_clean.txt lists.")
        st.stop()
    ssp = st.selectbox("SSP folder", options=ssp_dirs, format_func=lambda p: p.name)

    # Free selection (no clamping)
    start = st.date_input("Start date", value=pd.to_datetime("2021-01-01"))
    end   = st.date_input("End date",   value=pd.to_datetime("2100-12-31"))

    st.markdown("---")
    st.header("Debug / speed controls")
    debug_mode = st.checkbox("⚡ Debug mode (faster fetch)", value=False,
                             help="Use fewer files per variable and smaller time slices while debugging.")
    max_files_per_var = st.number_input("Max files per variable (0 = all)", value=8 if debug_mode else 0, min_value=0, step=1)

    st.markdown("---")
    st.header("Δ donor bias (optional)")
    donor_well = st.text_input("Donor well ID (from training) to transfer Δ bias", value="")

    st.markdown("---")
    st.header("Δ-RF Rollout Controls")
    disable_drift_guard = st.checkbox("Disable drift guard in Δ-RF rollout", value=False, help="Bypass the rolling zero-mean guard to see raw Δ-RF predictions.")

    st.markdown("---")
    st.header("Baseline mode")
    use_future_regkrig = st.checkbox(
        "Use future-informed reg-kriging (requires rf_baseline_future_by_well.csv)",
        value=RF_FUTURE_WELLS.exists(),
        help="If available, krige per-day using well RF predictions. Else: static reg-krig + local RF drift."
    )
    enable_awwid_dem_bias = st.checkbox(
        "Correct DEM with AWwID bias", value=True,
        help="Use nearby AWwID wells to correct local DEM bias. Requires regkrig_baseline_train.csv."
    )

    st.markdown("---")
    st.header("Depth (DEM)")
    dem_override = st.number_input("Override DEM (m a.s.l., optional)", value=0.0, step=1.0,
                                   help="Leave 0 to auto-fetch from Open-Elevation.")

    st.markdown("---")
    with st.sidebar.expander("Artifacts status", expanded=False):
        st.code(
            "CWD:  " + os.getcwd() + "\n"
            "APP:  " + str(APP_DIR) + "\n"
            "DP:   " + str(DP) + "\n"
            "JOB:  " + str(REGK_JOBLIB) + f"  | exists={REGK_JOBLIB.exists()}\n"
            "META: " + str(REGK_META)   + f"  | exists={REGK_META.exists()}",
            language="text"
        )
        # (optional) list a few files in data_products
        try:
            if DP.exists():
                files = "\n".join(sorted(p.name for p in DP.iterdir()))
                st.text_area("data_products/", files, height=120)
        except Exception as e:
            st.caption(f"ls data_products failed: {e}")

    if st.sidebar.button("Clear reg-krig cache"):
        load_regkrig_bundle.clear()
        st.experimental_rerun()

@st.cache_data
def load_awwid_wells(p: Path):
    if not p.exists(): return None
    df = pd.read_csv(p)
    # Ensure required columns exist and have the right names
    # Assuming AWwID has 'Latitude', 'Longitude', 'Elevation' (in feet)
    # And your training data has harmonized them to 'lat', 'lon', 'GSE_m'
    if {"lat", "lon", "GSE_m"}.issubset(df.columns):
        return df[["lat", "lon", "GSE_m"]].dropna().copy()
    return None

# This needs to be imported after the sidebar is defined for the map to work
from streamlit_folium import st_folium

st.subheader("1) Pick a location")
colL, colR = st.columns([2,1], gap="large")
with colL:
    fmap = folium.Map(location=[53.55, -113.50], zoom_start=10, tiles="CartoDB positron")
    folium.LatLngPopup().add_to(fmap)
    map_data = st_folium(fmap, height=520, use_container_width=True, key="gw_map_v2")
with colR:
    lat = lon = None
    if map_data and map_data.get("last_clicked"):
        lat = float(map_data["last_clicked"]["lat"])
        lon = float(map_data["last_clicked"]["lng"])
        st.success(f"Selected: lat={lat:.5f}, lon={lon:.5f}")
    else:
        st.info("Click on the map to select a point.")
    run = st.button("🚀 Run projection", type="primary", disabled=(lat is None))

# -------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------
if run and (lat is not None) and (lon is not None):
    regk = None
    try:
        regk = load_regkrig_bundle(_sig(REGK_JOBLIB) + "|" + _sig(REGK_META))
        st.sidebar.success("Reg-kriging baseline: loaded")
    except FileNotFoundError as e:
        st.sidebar.error(str(e))

    try:
        # 2) Ouranos hourly @ point -> daily drivers
        st.subheader("2) Ouranos hourly at point")
        dmin = pd.to_datetime(start).strftime("%Y-%m-%d")
        dmax = pd.to_datetime(end).strftime("%Y-%m-%d")
        H = fetch_hourly_ouranos(lat, lon, Path(ssp), dmin, dmax, max_files_per_var=max_files_per_var)
        st.success(f"Hourly forcing rows: {len(H):,}")
        if not H.empty:
            t0, t1 = H.index.min(), H.index.max()
            st.caption(f"Fetched hourly window: {t0.date()} → {t1.date()} ({len(H):,} hours)")

        st.subheader("3) Daily climate drivers")
        D = hourly_to_daily(H)
        D = D[(D["Date"]>=pd.to_datetime(dmin)) & (D["Date"]<=pd.to_datetime(dmax))].reset_index(drop=True)

        # Optional river emulator
        riv = maybe_predict_river(D)
        if riv is not None:
            st.info("River emulator applied (daily river level features).")

        st.subheader("4) Spatial baseline (two modes)")

        # (a) DEM for spatial features (unchanged except capped bias)
        if dem_override and dem_override > 0:
            dem_m = float(dem_override)
        else:
            dem_m = get_dem_open_elevation(lat, lon)
        if dem_m is None:
            st.error("DEM fetch failed. Cannot proceed with baseline calculation without ground elevation.")
            st.stop()
        
        if enable_awwid_dem_bias and dem_m is not None:
            wells_df = load_awwid_wells(REGK_TRAIN)
            if wells_df is not None:
                dists = haversine_np(lon, lat, wells_df['lon'].values, wells_df['lat'].values)
                cand = wells_df[dists < 1000].copy()
                if len(cand) >= 3:
                    cand['dem_at_well'] = get_dem_open_elevation_bulk(list(zip(cand['lat'], cand['lon'])))
                    cand = cand.dropna(subset=['dem_at_well'])
                    if len(cand) >= 3:
                        bias = float((cand['GSE_m'] - cand['dem_at_well']).clip(-3,3).median())  # cap ±3 m
                        dem_m += bias
                        st.caption(f"DEM corrected by {bias:+.2f} m using {len(cand)} nearby wells.")

        # (b) Static reg-kriging baseline as fallback/seed
        static_baseline_masl = 0.0
        x_m, y_m = to_local_utm(lat, lon)
        if regk is None:
            st.error("Reg-krig artifacts not found; using 0.0 as static baseline.")
        else:
            try:
                reg_pipe = regk["model"].get("reg") if isinstance(regk["model"], dict) else regk["model"]
                model_meta = getattr(reg_pipe, 'meta', regk["meta"])
                if model_meta.get("input_mode") == "xy+gse":
                    X = np.array([[x_m, y_m, dem_m]], float)
                    used_feats = ["x_m","y_m","GSE_m"]
                else:
                    X = np.array([[x_m, y_m]], float); used_feats = ["x_m","y_m"]
                static_baseline_masl = float(reg_pipe.predict(X)[0])
                st.success(f"Reg-kriging baseline: {static_baseline_masl:.2f} masl")
                with st.expander("Reg-kriging debug", expanded=False):
                    st.write("Features used for prediction:\n\n", used_feats)
                    st.write("X:\n\n", dict(zip(used_feats, X[0])))
            except Exception as e:
                st.error(f"Reg-kriging prediction failed: {e}")
                static_baseline_masl = 0.0

        # (c) Optional spatial bias surface
        if REGK_BIAS.exists():
            try:
                bias_art = joblib.load(REGK_BIAS) # This can fail
                gp = bias_art["gp"]
                center = bias_art["center"]
                bias_hat = float(gp.predict(np.array([[x_m, y_m]]))[0] + center)
                static_baseline_masl += bias_hat
                st.caption(f"Applied bias correction: {bias_hat:+.2f} m") # Use the float directly
            except Exception as e:
                st.warning(f"Could not apply bias correction: {e}")

        dates_idx = pd.to_datetime(D["Date"])
        baseline_series = pd.Series(static_baseline_masl, index=dates_idx, name="baseline_static")

        # (d) NEW: future-informed per-day reg-kriging
        if use_future_regkrig and RF_FUTURE_WELLS.exists():
            train_xy = load_regkrig_train_table()
            try:
                future_df = pd.read_csv(RF_FUTURE_WELLS, parse_dates=["Date"])
                future_df["Well_Name"] = future_df["Well_Name"].astype(str)
                perday = regkrig_from_daily_well_preds_fast(train_xy, future_df, x_m, y_m)
                perday = perday.reindex(dates_idx).interpolate().fillna(method="bfill").fillna(method="ffill")
                baseline_series = perday.rename("baseline_perday_krig")
                st.info("Using **per-day reg-kriging** from well RF predictions.")
            except Exception as e:
                st.warning(f"Per-day reg-kriging failed, falling back to static+drift: {e}")

        # (e) Local RF slow drift (point) if we are in static mode
        if baseline_series.name == "baseline_static":
            rf_drift = rf_baseline_local_series(D)
            if rf_drift is not None and not rf_drift.empty:
                rf_drift_zm = zero_mean(rf_drift.reindex(dates_idx).fillna(0.0))
                baseline_series = (baseline_series + rf_drift_zm).rename("baseline_static_plus_drift")
                with st.expander("RF slow drift (debug)", expanded=False):
                    st.write({
                        "drift_mean": float(rf_drift_zm.mean()),
                        "drift_std": float(rf_drift_zm.std()),
                        "min": float(rf_drift_zm.min()),
                        "max": float(rf_drift_zm.max())
                    })

        # (f) Compute GSE first (needed for re-anchor features)
        gse_pred = None
        dem_used_source = "n/a"
        if REGKRIG_GSE_MODEL is not None:
            x_m_gse, y_m_gse = to_local_utm(lat, lon)
            gse_pred = float(REGKRIG_GSE_MODEL.predict(np.array([[x_m_gse, y_m_gse]]))[0])
            dem_used_source = "AWwID-GSE (reg-krig)"
            st.caption(f"GSE from AWwID reg-krig: {gse_pred:.2f} masl")
        else:
            # Fallback to DEM if GSE model unavailable
            gse_pred = dem_m
            dem_used_source = "DEM (Open-Elevation fallback)"
            st.caption(f"GSE from DEM fallback: {gse_pred:.2f} masl")

        # (g) Monthly re-anchor (only if model loaded)
        anchor_adjustment = 0.0
        try:
            if 'reanchor_model' in globals() and 'reanchor_scaler' in globals():
                st.info("Applying seasonal re-anchor adjustment...")
                # Create a copy of D to add river features for the reanchor function
                D_for_reanchor = D.copy()
                if riv is not None:
                    D_for_reanchor = D_for_reanchor.merge(riv.rename("river_level_m").rename_axis("Date"), on="Date", how="left")

                anchor_adjustment, anchor_diag = predict_reanchor_daily(
                    D_for_reanchor,
                    lat=lat,
                    lon=lon,
                    dem_elev_m=dem_m,
                    slope_deg_dem=0, # Assuming 0 if not available
                    model_prefix="data_products/reanchor_monthly",
                    ema_span_days=7
                )
                baseline_series = baseline_series + anchor_adjustment
                st.caption(f"Seasonal anchor applied. Mean adjustment: {np.mean(anchor_adjustment):.3f} m")

            else:
                st.info("Monthly re-anchor model not loaded; skipping adjustment.")
        except Exception as e:
            st.warning(f"Re-anchor failed ({e}); proceeding without adjustment.")

        # 5) Δ-level RF rollout (with drift-guard)
        st.subheader("5) Δ-level RF rollout")
        dl_model, dl_feats, pw_bias = _load_delta_rf()
        st.caption(f"Δ-RF feature count = {len(dl_feats)}")
        daily_pred = rollout_delta_from_baseline(
            baseline_series=baseline_series,
            D=D,
            feats=dl_feats,
            model=dl_model,
            river_daily=riv,
            donor_well=(donor_well.strip() or None),
            perwell_bias_tbl=pw_bias, # type: ignore
            drift_guard_win=0 if disable_drift_guard else 90
        )

        with st.expander("Δ-RF rollout diagnostics", expanded=False):
            delta_raw = daily_pred["pred_delta_raw"]
            delta_final = daily_pred["pred_delta"]
            st.json({
                "raw_delta_stats": {"min": delta_raw.min(), "max": delta_raw.max(), "mean": delta_raw.mean(), "std": delta_raw.std()},
                "final_delta_stats": {"min": delta_final.min(), "max": delta_final.max(), "mean": delta_final.mean(), "std": delta_final.std()},
                "drift_guard_effect_mean": (delta_raw - delta_final).mean(),
                "drift_guard_effect_std": (delta_raw - delta_final).std(),
            })
        # The raw deltas are no longer needed
        daily_pred = daily_pred.drop(columns=["pred_delta_raw"])

        # 6) Depth = GSE (from AWwID reg-krig) − predicted level
        st.subheader("6) Depth (GSE − level)")

        daily_pred["gse_m"] = gse_pred if gse_pred is not None else np.nan

        # Pick an adaptive max-depth from nearby wells (AWwID)
        max_depth_local = estimate_local_max_depth_m(lat=float(lat), lon=float(lon))

        # Apply a "soft", one-sided clamp to prevent unrealistic depths
        level_unclamped = daily_pred["pred_level"].copy()
        depth_unclamped = daily_pred["gse_m"] - level_unclamped
        
        # Add a small buffer to the adaptive cap to avoid "always-on" clamping
        max_depth_with_buffer = max_depth_local + 1.0
        
        # One-sided clamp: only clip when predicted depth is greater than the cap
        depth_clamped = np.where(depth_unclamped > max_depth_with_buffer, max_depth_with_buffer, depth_unclamped)
        
        # Recalculate final level and depth
        daily_pred["pred_level"] = daily_pred["gse_m"] - depth_clamped
        daily_pred["gw_depth_m_bgs"] = pd.Series(depth_clamped).clip(lower=0.0)
        
        st.caption(f"DEM source used: {dem_used_source}")
        if gse_pred is not None:
            st.caption(f"Predicted GSE: {gse_pred:.2f} masl")
        st.caption(f"Adaptive clamp: max_depth_used = {max_depth_with_buffer:.1f} m (from nearby wells)")

        with st.expander("Diagnostics", expanded=False):
            diag = {
                "static_baseline": float(static_baseline_masl),
                "delta_mean": float(daily_pred["pred_delta"].mean()),
                "delta_std": float(daily_pred["pred_delta"].std()),
                "depth_unclamped_min": float(depth_unclamped.min()),
                "depth_unclamped_max": float(depth_unclamped.max()),
                "max_depth_used": float(max_depth_with_buffer),
                "pct_days_clamped": float((depth_unclamped > max_depth_with_buffer).mean())
            }
            st.json(diag)

        # Save + preview
        tag = f"{Path(ssp).name}_{lat:.4f}_{lon:.4f}"
        daily_out = DP / f"gw_daily_projection_{tag}.csv"
        daily_pred.to_csv(daily_out, index=False)
        st.success(f"Saved **daily** → {daily_out}")
        st.dataframe(daily_pred.head())

        st.subheader("7) Plots")
        st.line_chart(daily_pred.set_index("Date")[["pred_level"]].rename(columns={"pred_level":"GW level (masl)"}))
        if not np.isnan(daily_pred["gw_depth_m_bgs"]).all():
            st.line_chart(daily_pred.set_index("Date")[["gw_depth_m_bgs"]].rename(columns={"gw_depth_m_bgs":"GW depth (m b.g.s.)"}))

        # 8) Optional hourly downscale
        if st.checkbox("Also produce hourly series (shape downscale)", value=False):
            Hh = downscale_to_hourly(daily_pred[["Date","pred_delta","pred_level"]], H)
            if dem_m is not None:
                Hh["gw_depth_m_bgs"] = dem_m - Hh["level_masl_hourly"]
            hour_out = DP / f"gw_hourly_projection_{tag}.csv"
            Hh.to_csv(hour_out, index_label="time")
            st.success(f"Saved **hourly** → {hour_out}")
            st.dataframe(Hh.head())

        st.balloons()

    except Exception as e:
        st.error(f"Run failed: {e}")