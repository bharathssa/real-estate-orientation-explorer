# streamlit_app.py
# Real Estate Investor Visuals & Orientation Explorer

import os, math, re, requests
import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Dict

# Optional: GIS imports (skip if not using orientation module)
try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    from shapely.ops import nearest_points
    GIS_AVAILABLE = True
except Exception:
    GIS_AVAILABLE = False

from datetime import datetime, timezone

st.set_page_config(page_title='Investor Explorer & Orientation', layout='wide')

# ----------------------------
# Helpers: fetch & parsing
# ----------------------------
API_URL = "https://www.microburbs.com.au/report_generator/api/suburb/properties"

def fetch_properties(suburb: str, token: str="test") -> List[Dict]:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    r = requests.get(API_URL, params={"suburb": suburb}, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json().get("results", [])

def to_number_m2(x):
    if x is None: return np.nan
    s = str(x).strip().lower()
    if s in {"none", "nan", ""}: return np.nan
    s = s.replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else np.nan

def to_number_building(x):
    v = to_number_m2(x)
    if v == 0: return np.nan
    return v

def safe_int(x):
    try: return int(x)
    except Exception: return np.nan

def days_since(date_str):
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).days
    except Exception:
        return np.nan

def flag_kw(desc, *keywords):
    if not isinstance(desc, str): return 0
    low = desc.lower()
    return int(any(k.lower() in low for k in keywords))

def engineer_features(raw: List[Dict]) -> pd.DataFrame:
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
            "address": f"{addr.get('street')}, {addr.get('sal')}, {addr.get('state')}",
            "street": addr.get("street"),
            "suburb": addr.get("sal"),
            "state": addr.get("state"),
            "gnaf_pid": r.get("gnaf_pid"),
            "lat": r.get("coordinates", {}).get("latitude"),
            "lon": r.get("coordinates", {}).get("longitude"),
            "property_type": prop_type,
            "price_aud": price,
            "bedrooms": beds,
            "bathrooms": baths,
            "garage_spaces": garages,
            "land_sqm": land_sqm,
            "building_sqm": bldg_sqm,
            "listing_date": listing_date,
            "days_on_market": days_since(listing_date),
            "has_pool":  flag_kw(desc, "pool"),
            "has_spa":   flag_kw(desc, "spa"),
            "has_solar": flag_kw(desc, "solar"),
            "teen_retreat_or_studio": flag_kw(desc, "teenage retreat", "teen retreat", "studio", "granny"),
            "coastal_view": flag_kw(desc, "ocean", "beach", "coastal", "sea view", "panoramic"),
            "renovation_hint": flag_kw(desc, "renovated", "refurb", "updated", "modern", "new kitchen", "new bathroom"),
            "entertainer_outdoor": flag_kw(desc, "deck", "alfresco", "covered", "verandah", "entertaining"),
        }
        row["price_per_land_sqm"] = (price / land_sqm) if land_sqm and not math.isclose(land_sqm, 0.0) else np.nan
        row["price_per_bedroom"]  = (price / beds) if beds and beds > 0 else np.nan
        row["beds_per_100sqm_land"] = (100 * beds / land_sqm) if land_sqm and land_sqm > 0 and beds else np.nan
        row["baths_per_bedroom"] = (baths / beds) if beds and beds > 0 and baths is not None else np.nan
        row["parking_per_bedroom"] = (garages / beds) if beds and beds > 0 and garages is not None else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)

    # Outliers and z-score
    if df["price_aud"].notna().sum() >= 2:
        q1, q3 = df["price_aud"].quantile([0.25, 0.75])
        iqr = q3 - q1
        df["price_outlier_iqr"] = ((df["price_aud"] < (q1 - 1.5*iqr)) | (df["price_aud"] > (q3 + 1.5*iqr))).astype(int)
        mu = df["price_aud"].mean()
        sd = df["price_aud"].std(ddof=0)
        df["price_zscore"] = (df["price_aud"] - mu) / sd if sd and sd > 0 else np.nan
    else:
        df["price_outlier_iqr"] = 0
        df["price_zscore"] = np.nan

    # Investor Readiness Score
    w = {"land_cost": 0.35, "beds": 0.25, "amenities": 0.20, "days": 0.10, "outlier_penalty": 0.10}
    land_pps = df["price_per_land_sqm"]
    land_pps_norm = 1 - (land_pps - land_pps.min()) / (land_pps.max() - land_pps.min()) if land_pps.notna().sum() > 1 else 0.5
    beds_norm = (df["bedrooms"] - df["bedrooms"].min()) / (df["bedrooms"].max() - df["bedrooms"].min()) if df["bedrooms"].notna().sum() > 1 else 0.5
    amen_norm = (df["has_pool"] + df["has_solar"] + df["entertainer_outdoor"]).clip(0, 3) / 3
    days_norm = 1 - (df["days_on_market"] - df["days_on_market"].min()) / (df["days_on_market"].max() - df["days_on_market"].min()) if df["days_on_market"].notna().sum() > 1 else 0.5
    outlier_pen = (1 - df["price_outlier_iqr"])
    df["investor_readiness_score"] = (
        100 * (
            w["land_cost"] * pd.Series(land_pps_norm).fillna(0.5) +
            w["beds"]      * pd.Series(beds_norm).fillna(0.5) +
            w["amenities"] * amen_norm.fillna(0) +
            w["days"]      * pd.Series(days_norm).fillna(0.5) +
            w["outlier_penalty"] * outlier_pen.fillna(1.0)
        )
    ).round(1)

    return df

# ----------------------------
# Orientation module (optional GIS)
# ----------------------------
def compute_orientation(gnaf_df, cadastre_gdf, roads_gdf, snap_dist=30):
    if not GIS_AVAILABLE:
        st.warning("Geospatial libraries not available.")
        return pd.DataFrame()
    target_crs = roads_gdf.crs or "EPSG:4326"
    gnaf = gnaf_df.to_crs(target_crs)
    roads = roads_gdf.to_crs(target_crs)
    road_geom = roads.geometry.unary_union

    def closest_bearing(pt: Point):
        nearest = nearest_points(pt, road_geom)[1]
        roads["dist_tmp"] = roads.distance(nearest)
        idx = roads["dist_tmp"].idxmin()
        seg = roads.loc[idx].geometry
        if isinstance(seg, LineString):
            x0, y0 = list(seg.coords)[0]
            x1, y1 = list(seg.coords)[-1]
        else:
            x0, y0 = pt.x, pt.y
            x1, y1 = nearest.x, nearest.y
        angle = math.degrees(math.atan2(x1 - x0, y1 - y0))
        bearing = (angle + 360) % 360
        return bearing

    def to_cardinal(bearing):
        dirs = ['N','NE','E','SE','S','SW','W','NW','N']
        ix = int(round(bearing / 45.0))
        return dirs[ix]

    out_rows = []
    for i, row in gnaf.iterrows():
        geom = row.geometry
        try:
            b = closest_bearing(geom)
            card = to_cardinal(b)
        except Exception:
            b, card = np.nan, None
        out_rows.append({
            "gnaf_pid": row.get("gnaf_pid"),
            "address": row.get("address") or row.get("street"),
            "bearing_deg": b,
            "orientation": card
        })
    out = pd.DataFrame(out_rows)
    return out

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Controls")
suburb = st.sidebar.text_input("Suburb", value="Belmont North")
api_token = st.sidebar.text_input("API token", value="test")
min_beds = st.sidebar.slider("Min bedrooms", 0, 6, 0)
hide_outliers = st.sidebar.checkbox("Hide price outliers", value=True)
show_map = st.sidebar.checkbox("Show map", value=True)
enable_orientation = st.sidebar.checkbox("Enable orientation module (needs local GIS files)", value=False)
st.sidebar.markdown("---")
st.sidebar.caption("Tip: Toggle 'Hide price outliers' to see the effect on ranks and distributions.")

# ----------------------------
# Data section
# ----------------------------
with st.spinner("Fetching listings..."):
    try:
        raw = fetch_properties(suburb, api_token)
        df = engineer_features(raw)
    except Exception as e:
        st.error(f"API error: {e}")
        df = pd.DataFrame()

st.title("Investor Explorer")
st.caption("Turns noisy listing data into investor-ready signals.")

if df.empty:
    st.warning("No data returned. Check suburb or token.")
    st.stop()

if min_beds:
    df = df[df["bedrooms"].fillna(0) >= min_beds]
if hide_outliers and "price_outlier_iqr" in df.columns:
    df = df[df["price_outlier_iqr"] == 0]

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Median price (filtered)", f"${int(df['price_aud'].median()):,}" if df['price_aud'].notna().any() else "—")
with col2:
    st.metric("Median land size", f"{int(df['land_sqm'].median())} m²" if df['land_sqm'].notna().any() else "—")
with col3:
    st.metric("Median DOM", int(df['days_on_market'].median()) if df['days_on_market'].notna().any() else "—")
with col4:
    st.metric("Avg Investor Score", f"{df['investor_readiness_score'].mean():.1f}" if 'investor_readiness_score' in df else "—")

st.markdown("### Ranked properties")
rank_cols = [
    "address","property_type","price_aud","bedrooms","bathrooms","garage_spaces",
    "land_sqm","building_sqm","price_per_land_sqm","price_per_bedroom",
    "days_on_market","has_pool","has_solar","entertainer_outdoor","price_zscore","investor_readiness_score"
]
tbl = df[rank_cols].sort_values("investor_readiness_score", ascending=False)
st.dataframe(tbl, use_container_width=True)

# Export
csv = tbl.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", data=csv, file_name=f"{suburb.replace(' ','_').lower()}_ranked.csv", mime="text/csv")

# Charts
st.markdown("### Distributions & Relationships")
c1, c2 = st.columns(2)
with c1:
    st.bar_chart(df[["price_aud"]].dropna(), height=280)
    st.caption("Distribution of prices (bars auto-binned by Streamlit).")
with c2:
    if df[["land_sqm","price_aud"]].dropna().shape[0] > 0:
        st.scatter_chart(df[["land_sqm","price_aud"]].dropna(), x="land_sqm", y="price_aud", height=280)
        st.caption("Price vs land size.")

if show_map and df[["lat","lon"]].dropna().shape[0] > 0:
    st.markdown("### Map")
    st.map(df.rename(columns={"lat":"latitude","lon":"longitude"}), use_container_width=True, height=380)

# ----------------------------
# Orientation module (optional)
# ----------------------------
if enable_orientation:
    st.markdown("---")
    st.header("Orientation Explorer (one row per property)")
    if not GIS_AVAILABLE:
        st.error("Install geopandas & shapely to enable this module.")
    else:
        cad_path = st.text_input("Cadastre .gpkg path", value="cadastre.gpkg")
        gnaf_path = st.text_input("G-NAF addresses .parquet path", value="gnaf_prop.parquet")
        roads_path = st.text_input("Roads .gpkg path", value="roads.gpkg")
        try:
            cad_gdf = gpd.read_file(cad_path)

            # ---- THIS IS THE ONLY WAY TO READ THE PARQUET FILE ----
            import pandas as pd
            from shapely.geometry import Point

            gdf_tmp = pd.read_parquet(gnaf_path)
            # For debugging: show first 5 rows and column names
            st.write("GNAF address parquet preview:", gdf_tmp.head())
            st.write("Columns:", gdf_tmp.columns.tolist())

            # ----- CHANGE THESE if your file uses other names -----
            # Try these in order until one works:
            if {'longitude', 'latitude'}.issubset(gdf_tmp.columns):
                lon_col, lat_col = 'longitude', 'latitude'
            elif {'lon', 'lat'}.issubset(gdf_tmp.columns):
                lon_col, lat_col = 'lon', 'lat'
            elif {'LON', 'LAT'}.issubset(gdf_tmp.columns):
                lon_col, lat_col = 'LON', 'LAT'
            else:
                raise ValueError("Can't find longitude/latitude columns in your Parquet file! Please check column names above.")

            gdf_tmp['geometry'] = gdf_tmp.apply(lambda r: Point(r[lon_col], r[lat_col]), axis=1)
            gnaf_df = gpd.GeoDataFrame(gdf_tmp, geometry='geometry', crs='EPSG:4326')
            roads_gdf = gpd.read_file(roads_path)

            st.success("Loaded all GIS files.")

            orient_df = compute_orientation(gnaf_df, cad_gdf, roads_gdf)
            if not orient_df.empty:
                st.success("Orientation results calculated.")

                # Merge with address info for richer display (optional if already present)
                if 'address' in orient_df.columns:
                    display_cols = ['gnaf_pid', 'address', 'bearing_deg', 'orientation']
                else:
                    display_cols = ['gnaf_pid', 'bearing_deg', 'orientation']

                st.dataframe(orient_df[display_cols].head(100))

                # Orientation distribution chart
                orient_counts = orient_df['orientation'].value_counts().reindex(['N','NE','E','SE','S','SW','W','NW']).fillna(0)
                st.bar_chart(orient_counts)

                # Allow filter by orientation
                selected_orient = st.multiselect(
                    "Filter by orientation",
                    options=orient_counts.index.tolist(),
                    default=orient_counts.index.tolist()
                )
                filtered = orient_df[orient_df['orientation'].isin(selected_orient)]
                st.write("Filtered Properties:", filtered.head(20))

                # Download filtered
                out_csv = filtered.to_csv(index=False).encode("utf-8")
                st.download_button("Download filtered orientation CSV", data=out_csv, file_name=f"{suburb.replace(' ','_').lower()}_orientation_filtered.csv")

            else:
                st.warning("No orientation computed—check join keys and CRS.")

        except Exception as e:
            st.error(f"Orientation module error: {e}")
