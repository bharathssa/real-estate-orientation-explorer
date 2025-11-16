import requests, pandas as pd, numpy as np
from datetime import datetime, timezone
import re, math, textwrap

# ----------------------------
# 1) Fetch
# ----------------------------
def fetch_suburb_properties(suburb: str, token: str = "test") -> list[dict]:
    url = "https://www.microburbs.com.au/report_generator/api/suburb/properties"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    r = requests.get(url, params={"suburb": suburb}, headers=headers, timeout=60)
    r.raise_for_status()
    payload = r.json()
    return payload.get("results", [])

# ----------------------------
# 2) Helpers: parsing / cleaning
# ----------------------------
def to_number_m2(x):
    """Convert strings like '605 m²', '697 m²', '556SQM', '708.0', None → float sqm."""
    if x is None:
        return np.nan
    s = str(x).strip().lower()
    if s in {"none", "nan", ""}:
        return np.nan
    # keep digits, ., and thousand separators
    s = s.replace(",", "")
    # pull leading numeric token
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else np.nan

def to_number_building(x):
    """Same idea for building_size; treat 0.0 as NaN if looks like placeholder."""
    v = to_number_m2(x)
    if v == 0:
        return np.nan
    return v

def safe_int(x):
    try:
        return int(x)
    except Exception:
        return np.nan

def days_since(date_str):
    """YYYY-MM-DD → integer days since listing (UTC)."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).days
    except Exception:
        return np.nan

def flag_kw(desc, *keywords):
    if not isinstance(desc, str):
        return 0
    low = desc.lower()
    return int(any(k.lower() in low for k in keywords))

# ----------------------------
# 3) Feature engineering
# ----------------------------
def engineer_features(raw: list[dict]) -> pd.DataFrame:
    rows = []
    for r in raw:
        addr = r.get("address", {})
        attrs = r.get("attributes", {}) or {}
        desc  = attrs.get("description", "")

        land_sqm     = to_number_m2(attrs.get("land_size"))
        bldg_sqm     = to_number_building(attrs.get("building_size"))
        beds         = safe_int(attrs.get("bedrooms"))
        baths        = safe_int(attrs.get("bathrooms"))
        garages      = safe_int(attrs.get("garage_spaces"))
        price        = r.get("price", np.nan)
        prop_type    = r.get("property_type")
        listing_date = r.get("listing_date")

        row = {
            # identifiers
            "address": f"{addr.get('street')}, {addr.get('sal')}, {addr.get('state')}",
            "street": addr.get("street"),
            "suburb": addr.get("sal"),
            "state": addr.get("state"),
            "gnaf_pid": r.get("gnaf_pid"),
            "lat": r.get("coordinates", {}).get("latitude"),
            "lon": r.get("coordinates", {}).get("longitude"),
            "property_type": prop_type,

            # raw numeric
            "price_aud": price,
            "bedrooms": beds,
            "bathrooms": baths,
            "garage_spaces": garages,
            "land_sqm": land_sqm,
            "building_sqm": bldg_sqm,
            "listing_date": listing_date,
            "days_on_market": days_since(listing_date),

            # quick text flags (amenities / value drivers)
            "has_pool":  flag_kw(desc, "pool"),
            "has_spa":   flag_kw(desc, "spa"),
            "has_solar": flag_kw(desc, "solar"),
            "teen_retreat_or_studio": flag_kw(desc, "teenage retreat", "teen retreat", "studio", "granny"),
            "coastal_view": flag_kw(desc, "ocean", "beach", "coastal", "sea view", "panoramic"),
        }

        # price-based ratios (guard against division by zero)
        row["price_per_land_sqm"] = (price / land_sqm) if land_sqm and not math.isclose(land_sqm, 0.0) else np.nan
        row["price_per_bedroom"]  = (price / beds) if beds and beds > 0 else np.nan
        row["beds_per_100sqm_land"] = (100 * beds / land_sqm) if land_sqm and land_sqm > 0 and beds else np.nan
        row["baths_per_bedroom"] = (baths / beds) if beds and beds > 0 and baths is not None else np.nan
        row["parking_per_bedroom"] = (garages / beds) if beds and beds > 0 and garages is not None else np.nan

        # simple quality signals
        row["has_renovation_hint"] = flag_kw(desc, "renovated", "refurb", "updated", "modern", "new kitchen", "new bathroom")
        row["entertainer_outdoor"] = flag_kw(desc, "deck", "alfresco", "covered", "verandah", "entertaining")

        rows.append(row)

    df = pd.DataFrame(rows)

    # Outlier diagnostics (IQR on price)
    q1, q3 = df["price_aud"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df["price_outlier_iqr"] = ((df["price_aud"] < (q1 - 1.5*iqr)) | (df["price_aud"] > (q3 + 1.5*iqr))).astype(int)

    # Z-score as secondary check (robust to small samples, still informative)
    if df["price_aud"].notna().sum() >= 2:
        mu = df["price_aud"].mean()
        sd = df["price_aud"].std(ddof=0)
        df["price_zscore"] = (df["price_aud"] - mu) / sd if sd and sd > 0 else np.nan
    else:
        df["price_zscore"] = np.nan

    # An ultra-compact investor “readiness” score (0–100) you can iterate later:
    #   Lower land price per sqm (cheaper land), more bedrooms, pool/solar, not an outlier, shorter DOM
    #   This is a *starting* heuristic; tweak weights to your thesis.
    w = {
        "land_cost": 0.35, "beds": 0.25, "amenities": 0.20, "days": 0.10, "outlier_penalty": 0.10
    }
    # normalise components
    land_pps = df["price_per_land_sqm"].copy()
    land_pps_norm = 1 - (land_pps - land_pps.min()) / (land_pps.max() - land_pps.min()) if land_pps.notna().sum() > 1 else 0.5
    beds_norm = (df["bedrooms"] - df["bedrooms"].min()) / (df["bedrooms"].max() - df["bedrooms"].min()) if df["bedrooms"].notna().sum() > 1 else 0.5
    amen_norm = (df["has_pool"] + df["has_solar"] + df["entertainer_outdoor"]).clip(0, 3) / 3
    days_norm = 1 - (df["days_on_market"] - df["days_on_market"].min()) / (df["days_on_market"].max() - df["days_on_market"].min()) if df["days_on_market"].notna().sum() > 1 else 0.5
    outlier_pen = (1 - df["price_outlier_iqr"])  # 1 if normal, 0 if outlier

    df["investor_readiness_score"] = (
        100 * (
            w["land_cost"] * land_pps_norm.fillna(0.5) +
            w["beds"]      * beds_norm.fillna(0.5) +
            w["amenities"] * amen_norm.fillna(0) +
            w["days"]      * days_norm.fillna(0.5) +
            w["outlier_penalty"] * outlier_pen.fillna(1.0)
        )
    ).round(1)

    # Sort nicely
    df = df.sort_values(["price_outlier_iqr", "price_aud"], ascending=[True, True]).reset_index(drop=True)
    return df

# ----------------------------
# 4) Run for a suburb & export
# ----------------------------
suburb = "Belmont North"   # change as needed
raw = fetch_suburb_properties(suburb)
df = engineer_features(raw)

display_cols = [
    "address","property_type","price_aud","bedrooms","bathrooms","garage_spaces",
    "land_sqm","building_sqm","price_per_land_sqm","price_per_bedroom",
    "days_on_market","has_pool","has_solar","teen_retreat_or_studio",
    "coastal_view","price_outlier_iqr","price_zscore","investor_readiness_score"
]
df_display = df[display_cols].copy()

# Save one-row-per-property CSV (repo-local)
out_path = f"{suburb.replace(' ','_').lower()}_properties_clean.csv"
df_display.to_csv(out_path, index=False)
out_path, df_display.head(10)
