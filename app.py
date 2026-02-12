import os
import re
import time
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

# ============================
# CONFIG
# ============================
st.set_page_config(page_title="Basel Café Karte", layout="wide")

DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "cafes.db")
os.makedirs(DB_DIR, exist_ok=True)

# Fixe Basel-BBox (S, W, N, E)
BASEL_BBOX = (47.52, 7.54, 47.58, 7.62)

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"

# Basel PLZ -> Quartier
PLZ_TO_QUARTIER = {
    "4001": "Altstadt Grossbasel",
    "4051": "Am Ring / Bachletten",
    "4052": "St. Alban / Breite",
    "4053": "Gundeldingen",
    "4054": "Bachletten / Am Ring",
    "4055": "Iselin / Gotthelf",
    "4056": "St. Johann / Am Ring",
    "4057": "Klybeck / Kleinhüningen",
    "4058": "Wettstein / Hirzbrunnen",
    "4059": "Bruderholz",
}
PLZ_RE = re.compile(r"\b(4001|4051|4052|4053|4054|4055|4056|4057|4058|4059)\b")

# opening_hours basics
DAYS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
DAY_TO_INDEX = {d: i for i, d in enumerate(DAYS)}
INDEX_TO_DAY = {i: d for i, d in enumerate(DAYS)}

# Common cases (Builder-compatible) part regex:
_OPENING_PART_RE = re.compile(
    r"^\s*([A-Za-z]{2}(?:-[A-Za-z]{2})?)\s+(\d{2}:\d{2}-\d{2}:\d{2}(?:,\d{2}:\d{2}-\d{2}:\d{2})*)\s*$"
)

# ============================
# STYLE: Karte "fast Vollbild"
# ============================
st.markdown(
    """
    <style>
      .block-container {padding-top: 1rem; padding-left: 1.2rem; padding-right: 1.2rem; padding-bottom: 0.2rem;}
      [data-testid="stDataFrame"] {margin-top: 0.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================
# HELPERS
# ============================
def _ua_headers():
    return {"User-Agent": "BaselCafeKarte/1.0 (Streamlit)"}


def quartier_from_plz(plz: Optional[str]) -> Optional[str]:
    if not plz:
        return None
    return PLZ_TO_QUARTIER.get(str(plz).strip())


def extract_plz_from_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = PLZ_RE.search(text)
    return m.group(1) if m else None


def find_nearest_cafe_id(df: pd.DataFrame, lat: float, lon: float, max_dist_deg: float = 0.002) -> Optional[int]:
    """
    Nimmt den nächsten Café-Punkt zu (lat,lon) in einem (sehr) groben Distanzfenster.
    max_dist_deg ~ 0.002 entspricht grob ~200m (für Basel ok).
    """
    if df.empty:
        return None
    d = (df["lat"] - lat) ** 2 + (df["lon"] - lon) ** 2
    idx = d.idxmin()
    if float(d.loc[idx]) > (max_dist_deg ** 2):
        return None
    return int(df.loc[idx, "id"])


def marker_color(overall: Optional[float], rating_count: int) -> str:
    if rating_count == 0 or overall is None:
        return "blue"
    if overall >= 4.0:
        return "green"
    if overall >= 2.0:
        return "orange"
    return "red"


# ============================
# Öffnungszeiten-Filter (Common Cases)
# ============================
def _day_matches(day_token: str, day: str) -> bool:
    # day_token: "Mo" oder "Mo-Fr"
    if "-" in day_token:
        a, b = day_token.split("-")
        if a not in DAY_TO_INDEX or b not in DAY_TO_INDEX:
            return False
        i, j = DAY_TO_INDEX[a], DAY_TO_INDEX[b]
        di = DAY_TO_INDEX[day]
        return i <= di <= j
    return day_token == day


def _time_in_ranges(hhmm: str, ranges: List[Tuple[str, str]]) -> bool:
    # hhmm: "14:30", ranges: [("08:00","12:00"), ...]
    t_h, t_m = map(int, hhmm.split(":"))
    t = t_h * 60 + t_m

    for s, e in ranges:
        sh, sm = map(int, s.split(":"))
        eh, em = map(int, e.split(":"))
        start = sh * 60 + sm
        end = eh * 60 + em

        if start <= end:
            if start <= t <= end:
                return True
        else:
            # über Mitternacht (z.B. 22:00-02:00)
            if t >= start or t <= end:
                return True
    return False


def is_open_at(opening_hours: Optional[str], day: str, hhmm: str) -> bool:
    """
    Unterstützt 'common cases' wie sie dein Builder erzeugt:
    'Mo-Fr 08:00-18:00; Sa 09:00-16:00'
    Mehrere Slots via Komma, mehrere Segmente via ;
    """
    if not opening_hours or not opening_hours.strip():
        return False

    parts = [p.strip() for p in opening_hours.split(";") if p.strip()]
    for p in parts:
        m = _OPENING_PART_RE.match(p)
        if not m:
            continue  # unbekanntes Format -> ignorieren
        day_part, times_part = m.group(1), m.group(2)

        if not _day_matches(day_part, day):
            continue

        ranges = []
        for block in times_part.split(","):
            block = block.strip()
            if not re.match(r"^\d{2}:\d{2}-\d{2}:\d{2}$", block):
                continue
            s, e = block.split("-")
            ranges.append((s, e))

        if _time_in_ranges(hhmm, ranges):
            return True

    return False


# ============================
# NOMINATIM SEARCH (Adresse finden)
# ============================
@st.cache_data(ttl=300)
def nominatim_search_basel(query: str, limit: int = 8) -> List[Dict[str, Any]]:
    """
    Liefert Vorschläge in/um Basel (bounding box + CH).
    """
    q = (query or "").strip()
    if len(q) < 3:
        return []

    south, west, north, east = BASEL_BBOX
    params = {
        "format": "jsonv2",
        "q": q,
        "limit": limit,
        "addressdetails": 1,
        "countrycodes": "ch",
        "bounded": 1,
        "viewbox": f"{west},{north},{east},{south}",
    }
    r = requests.get(NOMINATIM_SEARCH_URL, params=params, headers=_ua_headers(), timeout=20)
    r.raise_for_status()
    data = r.json()
    out = []
    for it in data:
        addr = it.get("address") or {}
        postcode = addr.get("postcode") or extract_plz_from_text(it.get("display_name"))
        out.append({
            "label": it.get("display_name"),
            "lat": float(it["lat"]),
            "lon": float(it["lon"]),
            "postcode": postcode,
        })
    return out


# ============================
# OPENING HOURS: UI -> OSM string
# ============================
def _time_to_str(t) -> str:
    return f"{t.hour:02d}:{t.minute:02d}"


def build_opening_hours_from_week(week: Dict[str, Dict[str, Any]]) -> str:
    """
    week: dict day -> {open: bool, ranges: [(start,end), ...]}
    Produziert opening_hours String, gruppiert gleiche Zeiten aufeinanderfolgend.
    """
    day_str = {}
    for d in DAYS:
        info = week.get(d, {})
        if not info.get("open"):
            day_str[d] = None
            continue
        ranges = info.get("ranges", [])
        parts = []
        for (s, e) in ranges:
            if s and e:
                parts.append(f"{_time_to_str(s)}-{_time_to_str(e)}")
        day_str[d] = ",".join(parts) if parts else None

    groups = []
    i = 0
    while i < len(DAYS):
        d = DAYS[i]
        hrs = day_str[d]
        if hrs is None:
            i += 1
            continue
        j = i
        while j + 1 < len(DAYS) and day_str[DAYS[j + 1]] == hrs:
            j += 1
        if i == j:
            day_part = DAYS[i]
        else:
            day_part = f"{DAYS[i]}-{DAYS[j]}"
        groups.append(f"{day_part} {hrs}")
        i = j + 1

    return "; ".join(groups)


def parse_opening_hours_simple(oh: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Sehr vereinfachter Parser für häufige Fälle:
    'Mo-Fr 08:00-18:00; Sa 09:00-16:00'
    Unterstützt auch mehrere Zeitfenster via Komma.
    Falls nicht parsebar -> None
    """
    if not oh or not oh.strip():
        return None

    week = {d: {"open": False, "ranges": []} for d in DAYS}
    parts = [p.strip() for p in oh.split(";") if p.strip()]
    for p in parts:
        m = _OPENING_PART_RE.match(p)
        if not m:
            return None
        day_part, times_part = m.group(1), m.group(2)

        # expand days
        if "-" in day_part:
            a, b = day_part.split("-")
            if a not in DAY_TO_INDEX or b not in DAY_TO_INDEX:
                return None
            start_i, end_i = DAY_TO_INDEX[a], DAY_TO_INDEX[b]
            if end_i < start_i:
                return None
            day_list = [INDEX_TO_DAY[i] for i in range(start_i, end_i + 1)]
        else:
            if day_part not in DAY_TO_INDEX:
                return None
            day_list = [day_part]

        ranges = []
        for block in times_part.split(","):
            block = block.strip()
            if not re.match(r"^\d{2}:\d{2}-\d{2}:\d{2}$", block):
                return None
            s, e = block.split("-")
            ranges.append((s, e))

        for d in day_list:
            week[d]["open"] = True
            week[d]["ranges"] = ranges[:]  # copy

    return week


def _str_to_time(hhmm: str):
    hh, mm = hhmm.split(":")
    return datetime(2000, 1, 1, int(hh), int(mm)).time()


def week_from_parsed(parsed_week: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    week = {d: {"open": False, "ranges": []} for d in DAYS}
    for d in DAYS:
        info = parsed_week.get(d, {})
        if not info.get("open"):
            continue
        week[d]["open"] = True
        rr = []
        for s, e in info.get("ranges", []):
            rr.append((_str_to_time(s), _str_to_time(e)))
        week[d]["ranges"] = rr
    return week


def opening_hours_editor(key_prefix: str, initial_oh: Optional[str]) -> str:
    """
    UI Editor: returns opening_hours string
    - versucht initial_oh zu parsen, sonst default leer
    """
    parsed = parse_opening_hours_simple(initial_oh or "")
    if parsed is not None:
        week = week_from_parsed(parsed)
        parse_ok = True
    else:
        week = {d: {"open": False, "ranges": []} for d in DAYS}
        parse_ok = False

    st.caption("Standard: OSM opening_hours (z.B. `Mo-Fr 08:00-18:00; Sa 09:00-16:00`).")

    if initial_oh and not parse_ok:
        st.warning("Vorhandene Öffnungszeiten konnte ich nicht sicher mappen. Du kannst sie unten manuell übernehmen oder neu erfassen.")

    raw = st.text_input("opening_hours (manuell, optional)", value=(initial_oh or ""), key=f"{key_prefix}_raw")
    use_raw = st.checkbox("Manuellen opening_hours-Text verwenden", value=False, key=f"{key_prefix}_use_raw")

    st.divider()
    st.markdown("**Öffnungszeiten-Builder** (empfohlen)")

    week_out: Dict[str, Dict[str, Any]] = {d: {"open": False, "ranges": []} for d in DAYS}

    for d in DAYS:
        col_a, col_b, col_c, col_d = st.columns([1.2, 1.2, 1.2, 1.2])
        with col_a:
            is_open = st.checkbox(d, value=week[d]["open"], key=f"{key_prefix}_{d}_open")
        ranges = []
        if is_open:
            default1 = week[d]["ranges"][0] if len(week[d]["ranges"]) >= 1 else (datetime(2000,1,1,8,0).time(), datetime(2000,1,1,18,0).time())
            default2 = week[d]["ranges"][1] if len(week[d]["ranges"]) >= 2 else (None, None)

            with col_b:
                s1 = st.time_input(f"{d} Start 1", value=default1[0], key=f"{key_prefix}_{d}_s1")
            with col_c:
                e1 = st.time_input(f"{d} Ende 1", value=default1[1], key=f"{key_prefix}_{d}_e1")
            ranges.append((s1, e1))

            with col_d:
                add_second = st.checkbox(f"{d} 2. Slot", value=(len(week[d]["ranges"]) >= 2), key=f"{key_prefix}_{d}_slot2")

            if add_second:
                c2b, c2c = st.columns([1, 1])
                with c2b:
                    s2 = st.time_input(f"{d} Start 2", value=(default2[0] or datetime(2000,1,1,0,0).time()), key=f"{key_prefix}_{d}_s2")
                with c2c:
                    e2 = st.time_input(f"{d} Ende 2", value=(default2[1] or datetime(2000,1,1,0,0).time()), key=f"{key_prefix}_{d}_e2")
                ranges.append((s2, e2))

        week_out[d]["open"] = bool(is_open)
        week_out[d]["ranges"] = ranges

    built = build_opening_hours_from_week(week_out)
    st.write(f"**Builder-Ergebnis:** `{built or ''}`")

    if use_raw:
        return (raw.strip() or "")
    return (built or "")


# ============================
# DB
# ============================
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def add_column_if_missing(conn, table: str, column: str, coltype: str):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table});")
    cols = {row[1] for row in cur.fetchall()}
    if column not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype};")


def init_db():
    with get_conn() as conn:
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS cafes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            osm_id TEXT,
            name TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            address TEXT,
            postcode TEXT,
            quartier TEXT,
            opening_hours TEXT,
            created_at TEXT NOT NULL
        );
        """)

        cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_cafes_osm_unique
        ON cafes(osm_id) WHERE osm_id IS NOT NULL;
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cafe_id INTEGER NOT NULL,
            reviewer_name TEXT,
            coffee_quality INTEGER NOT NULL CHECK(coffee_quality BETWEEN 1 AND 5),
            ambience INTEGER NOT NULL CHECK(ambience BETWEEN 1 AND 5),
            service INTEGER NOT NULL CHECK(service BETWEEN 1 AND 5),
            value_for_money INTEGER NOT NULL CHECK(value_for_money BETWEEN 1 AND 5),
            food INTEGER NOT NULL CHECK(food BETWEEN 1 AND 5),
            comment TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(cafe_id) REFERENCES cafes(id)
        );
        """)

        add_column_if_missing(conn, "cafes", "address", "TEXT")
        add_column_if_missing(conn, "cafes", "postcode", "TEXT")
        add_column_if_missing(conn, "cafes", "quartier", "TEXT")
        add_column_if_missing(conn, "cafes", "opening_hours", "TEXT")
        add_column_if_missing(conn, "ratings", "reviewer_name", "TEXT")

        conn.commit()


def cafes_count() -> int:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM cafes;")
        return int(cur.fetchone()[0])


def missing_postcode_count() -> int:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM cafes WHERE postcode IS NULL OR TRIM(postcode) = ''")
        return int(cur.fetchone()[0])


def upsert_osm_cafes(rows: List[Dict[str, Any]]):
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        cur = conn.cursor()
        for r in rows:
            cur.execute("""
                INSERT OR IGNORE INTO cafes
                (source, osm_id, name, lat, lon, address, postcode, quartier, opening_hours, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "osm",
                r.get("osm_id"),
                r.get("name") or "Unbekannt",
                float(r["lat"]),
                float(r["lon"]),
                r.get("address"),
                r.get("postcode"),
                r.get("quartier"),
                r.get("opening_hours"),
                now
            ))

            cur.execute("""
                UPDATE cafes
                SET
                    name = COALESCE(NULLIF(name, ''), ?),
                    address = COALESCE(address, ?),
                    postcode = COALESCE(postcode, ?),
                    quartier = COALESCE(quartier, ?),
                    opening_hours = COALESCE(opening_hours, ?)
                WHERE osm_id = ?
            """, (
                r.get("name"),
                r.get("address"),
                r.get("postcode"),
                r.get("quartier"),
                r.get("opening_hours"),
                r.get("osm_id"),
            ))

        conn.commit()


def update_cafe_location_fields(cafe_id: int, address: Optional[str], postcode: Optional[str], quartier: Optional[str]):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE cafes
            SET address = COALESCE(address, ?),
                postcode = COALESCE(postcode, ?),
                quartier = COALESCE(quartier, ?)
            WHERE id = ?
        """, (address, postcode, quartier, cafe_id))
        conn.commit()


def update_cafe(
    cafe_id: int,
    name: str,
    lat: float,
    lon: float,
    address: Optional[str],
    postcode: Optional[str],
    quartier: Optional[str],
    opening_hours: Optional[str],
):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE cafes
            SET name = ?,
                lat = ?,
                lon = ?,
                address = ?,
                postcode = ?,
                quartier = ?,
                opening_hours = ?
            WHERE id = ?
        """, (name, lat, lon, address, postcode, quartier, opening_hours, cafe_id))
        conn.commit()


def insert_user_cafe(
    name: str,
    lat: float,
    lon: float,
    address: Optional[str],
    opening_hours: Optional[str],
    postcode: Optional[str],
    quartier: Optional[str],
):
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO cafes
            (source, osm_id, name, lat, lon, address, postcode, quartier, opening_hours, created_at)
            VALUES ('user', NULL, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, lat, lon, address, postcode, quartier, opening_hours, now))
        conn.commit()


def insert_rating_multi(
    cafe_id: int,
    reviewer_name: Optional[str],
    coffee_quality: int,
    ambience: int,
    service: int,
    value_for_money: int,
    food: int,
    comment: Optional[str]
):
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO ratings
            (cafe_id, reviewer_name, coffee_quality, ambience, service, value_for_money, food, comment, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (cafe_id, reviewer_name, coffee_quality, ambience, service, value_for_money, food, comment, now))
        conn.commit()


def load_cafes_with_stats() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query("""
            SELECT
                c.*,
                AVG((r.coffee_quality + r.ambience + r.service + r.value_for_money + r.food) / 5.0) AS overall,
                COUNT(r.id) AS rating_count
            FROM cafes c
            LEFT JOIN ratings r ON r.cafe_id = c.id
            GROUP BY c.id
        """, conn)
    return df


def load_cafes_missing_postcode(limit: int = 5000) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, lat, lon
            FROM cafes
            WHERE postcode IS NULL OR TRIM(postcode) = ''
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
    return [{"id": r[0], "lat": float(r[1]), "lon": float(r[2])} for r in rows]


def load_rating_averages_for_cafe(cafe_id: int) -> Dict[str, Optional[float]]:
    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                AVG(coffee_quality) AS avg_coffee_quality,
                AVG(ambience) AS avg_ambience,
                AVG(service) AS avg_service,
                AVG(value_for_money) AS avg_value_for_money,
                AVG(food) AS avg_food,
                AVG((coffee_quality + ambience + service + value_for_money + food) / 5.0) AS avg_overall,
                COUNT(*) AS n
            FROM ratings
            WHERE cafe_id = ?
            """,
            conn,
            params=(cafe_id,),
        )
    return df.iloc[0].to_dict()


def load_ratings_for_cafe(cafe_id: int) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                created_at,
                reviewer_name,
                coffee_quality,
                ambience,
                service,
                value_for_money,
                food,
                ROUND((coffee_quality + ambience + service + value_for_money + food) / 5.0, 1) AS overall,
                comment
            FROM ratings
            WHERE cafe_id = ?
            ORDER BY datetime(created_at) DESC
            """,
            conn,
            params=(cafe_id,),
        )
    return df


# ============================
# OSM IMPORT
# ============================
def fetch_osm_cafes_basel_bbox(bbox: Tuple[float, float, float, float]) -> List[Dict[str, Any]]:
    south, west, north, east = bbox

    query = f"""
    [out:json][timeout:60];
    (
      node["amenity"="cafe"]({south},{west},{north},{east});
      way["amenity"="cafe"]({south},{west},{north},{east});
      relation["amenity"="cafe"]({south},{west},{north},{east});
    );
    out tags center;
    """

    last_error = None

    for base_url in OVERPASS_URLS:
        for attempt in range(1, 4):
            try:
                r = requests.post(
                    base_url,
                    data=query.encode("utf-8"),
                    headers={"Content-Type": "text/plain"},
                    timeout=90
                )
                r.raise_for_status()
                data = r.json()

                rows: List[Dict[str, Any]] = []
                for el in data.get("elements", []):
                    tags = el.get("tags") or {}
                    name = tags.get("name")
                    if not name:
                        continue

                    if "lat" in el and "lon" in el:
                        lat, lon = el["lat"], el["lon"]
                    else:
                        center = el.get("center")
                        if not center:
                            continue
                        lat, lon = center["lat"], center["lon"]

                    osm_id = f'{el.get("type")}:{el.get("id")}'

                    postcode = tags.get("addr:postcode")
                    street = tags.get("addr:street")
                    hnr = tags.get("addr:housenumber")
                    city = tags.get("addr:city") or "Basel"

                    address = None
                    if street and hnr and postcode:
                        address = f"{street} {hnr}, {postcode} {city}"
                    elif street and hnr:
                        address = f"{street} {hnr}, {city}"
                    elif street and postcode:
                        address = f"{street}, {postcode} {city}"
                    elif street:
                        address = f"{street}, {city}"

                    if not postcode:
                        postcode = extract_plz_from_text(address)

                    quartier = quartier_from_plz(postcode)

                    rows.append({
                        "osm_id": osm_id,
                        "name": name,
                        "lat": float(lat),
                        "lon": float(lon),
                        "opening_hours": tags.get("opening_hours"),
                        "address": address,
                        "postcode": postcode,
                        "quartier": quartier,
                    })

                return rows

            except Exception as e:
                last_error = e
                time.sleep(2 ** (attempt - 1))

    raise RuntimeError(f"Overpass nicht erreichbar (mehrere Server probiert). Letzter Fehler: {last_error}")


# ============================
# Reverse Geocoding (Nominatim)
# ============================
def nominatim_reverse(lat: float, lon: float, session: requests.Session) -> Tuple[Optional[str], Optional[str]]:
    params = {
        "format": "jsonv2",
        "lat": lat,
        "lon": lon,
        "zoom": 18,
        "addressdetails": 1
    }
    resp = session.get(NOMINATIM_REVERSE_URL, params=params, headers=_ua_headers(), timeout=20)
    resp.raise_for_status()
    js = resp.json()

    addr = js.get("address") or {}
    postcode = addr.get("postcode")
    display = js.get("display_name")

    if not postcode:
        postcode = extract_plz_from_text(display)

    return display, postcode


def enrich_missing_postcodes_with_progress():
    missing = load_cafes_missing_postcode(limit=5000)
    if not missing:
        return

    st.info(f"Ergänze PLZ/Adresse für {len(missing)} Cafés (Nominatim, 1 Anfrage/Sek.).")

    progress = st.progress(0)
    status = st.empty()
    session = requests.Session()

    total = len(missing)
    for i, row in enumerate(missing, start=1):
        cafe_id = row["id"]
        lat = row["lat"]
        lon = row["lon"]

        try:
            address_str, postcode = nominatim_reverse(lat, lon, session)
            quartier = quartier_from_plz(postcode)
            update_cafe_location_fields(cafe_id, address_str, postcode, quartier)
        except Exception as e:
            status.write(f"Fehler bei ID {cafe_id}: {e}")

        time.sleep(1.05)
        progress.progress(i / total)
        status.write(f"{i}/{total} verarbeitet...")

    status.success("PLZ/Quartier-Anreicherung abgeschlossen.")


def bootstrap_db_once():
    if st.session_state.get("bootstrapped", False):
        return

    init_db()

    if cafes_count() == 0:
        st.info("Initialer Basel-Import läuft (OpenStreetMap)…")
        rows = fetch_osm_cafes_basel_bbox(BASEL_BBOX)
        upsert_osm_cafes(rows)
        st.success(f"Import abgeschlossen: {len(rows)} Cafés gefunden.")

    if missing_postcode_count() > 0:
        enrich_missing_postcodes_with_progress()

    st.session_state["bootstrapped"] = True


# ============================
# APP
# ============================
bootstrap_db_once()
cafes = load_cafes_with_stats()

st.markdown("## 👁️ Basel Café Karte")
st.caption("Cafés (OSM) sind automatisch importiert. Bewertungen & Filter sind lokal in SQLite gespeichert.")

# ----------------------------
# Filter: Popover oben links
# ----------------------------
with st.popover("🔎 Filter", use_container_width=False):
    quartiere = sorted([q for q in cafes["quartier"].dropna().unique()])
    selected_quartier = st.selectbox("Quartier", ["(alle)"] + quartiere)

    min_overall = st.slider("Mindest-Overall", 0.0, 5.0, 0.0, 0.5)
    only_with_hours = st.checkbox("Nur Cafés mit Öffnungszeiten", value=False)
    hours_contains = st.text_input("Öffnungszeiten enthält (Text)", value="")

    st.divider()
    st.markdown("**Öffnungszeiten**")
    open_now = st.checkbox("Nur aktuell offene Cafés", value=False)
    custom_time_filter = st.checkbox("Nach Wochentag/Uhrzeit filtern", value=False)
    filter_day = st.selectbox("Wochentag", DAYS, index=DAYS.index("Mo"))
    filter_time = st.time_input("Uhrzeit", value=datetime(2000, 1, 1, 12, 0).time())

# Apply filters (Map)
filtered = cafes.copy()
filtered = filtered[(filtered["overall"].fillna(0.0)) >= float(min_overall)]

if selected_quartier != "(alle)":
    filtered = filtered[filtered["quartier"] == selected_quartier]

if only_with_hours:
    filtered = filtered[filtered["opening_hours"].fillna("").str.strip() != ""]

if hours_contains.strip():
    filtered = filtered[filtered["opening_hours"].fillna("").str.contains(hours_contains.strip(), case=False, na=False)]

# Öffnungszeiten: jetzt offen
if open_now:
    now = datetime.now()
    day = DAYS[now.weekday()]  # Mo..Su
    hhmm = f"{now.hour:02d}:{now.minute:02d}"
    filtered = filtered[
        filtered["opening_hours"].fillna("").apply(lambda oh: is_open_at(oh, day, hhmm))
    ]

# Öffnungszeiten: custom Tag/Uhrzeit
if custom_time_filter:
    hhmm = f"{filter_time.hour:02d}:{filter_time.minute:02d}"
    filtered = filtered[
        filtered["opening_hours"].fillna("").apply(lambda oh: is_open_at(oh, filter_day, hhmm))
    ]

# ----------------------------
# Layout: Karte groß + Controls rechts
# ----------------------------
map_col, ui_col = st.columns([3.6, 1.4], gap="large")

with map_col:
    if len(filtered) > 0:
        center_lat = float(filtered["lat"].mean())
        center_lon = float(filtered["lon"].mean())
    else:
        center_lat, center_lon = 47.5596, 7.5886

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    for _, row in filtered.iterrows():
        color = marker_color(row["overall"], int(row["rating_count"]))
        icon = folium.Icon(color=color, icon="coffee", prefix="fa")

        rating_txt = "-" if row["overall"] is None else f"{float(row['overall']):.1f} ⭐"
        popup_html = f"""
        <b>{row['name']}</b><br>
        Adresse: {row['address'] or '-'}<br>
        Bewertung: {rating_txt} ({int(row['rating_count'])})<br>
        Öffnungszeiten: {row['opening_hours'] or '-'}
        """

        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=folium.Popup(popup_html, max_width=420),
            tooltip=row["name"],
            icon=icon
        ).add_to(m)

    st.caption("Klick auf Marker (oder in die Nähe) → Café auswählen. Klick auf Karte → Koordinaten übernehmen.")
    map_state = st_folium(m, width=None, height=720)

    # Auswahl per Klick (Marker oder Karte)
    clicked_obj = None
    if map_state:
        clicked_obj = map_state.get("last_object_clicked") or map_state.get("last_clicked")

    if clicked_obj:
        lat_click = float(clicked_obj["lat"])
        lon_click = float(clicked_obj.get("lng", clicked_obj.get("lon")))
        selected_id = find_nearest_cafe_id(filtered, lat_click, lon_click)
        if selected_id is not None:
            st.session_state["selected_cafe_id"] = int(selected_id)

    # Koordinaten für "Neues Café"
    clicked = map_state.get("last_clicked") if map_state else None

with ui_col:
    st.markdown("### 🧰 Aktionen")
    tab_manage, tab_rate = st.tabs(["➕ / ✏️ Café", "⭐ Bewerten"])

    # ----------------------------
    # TAB: Create / Edit Cafe
    # ----------------------------
    with tab_manage:
        st.caption("Neu erstellen oder bestehendes Café bearbeiten. Adresse über Nominatim suchen (Basel).")

        cafes_for_select = cafes.copy()
        cafes_for_select["label"] = cafes_for_select.apply(
            lambda r: f"{r['name']} — {r['address'] or ''}".strip(" —"),
            axis=1
        )

        manage_mode = st.selectbox("Modus", ["Neu erstellen", "Bearbeiten"], index=0)

        edit_id = None
        if manage_mode == "Bearbeiten" and not cafes_for_select.empty:
            sel_id = st.session_state.get("selected_cafe_id")
            options = cafes_for_select.sort_values("name")[["id", "label"]].to_records(index=False)
            id_to_label = {int(i): str(l) for i, l in options}
            all_labels = list(id_to_label.values())
            default_label = id_to_label.get(int(sel_id)) if sel_id else (all_labels[0] if all_labels else "")
            chosen = st.selectbox("Café wählen", all_labels, index=all_labels.index(default_label) if default_label in all_labels else 0)
            edit_id = [k for k, v in id_to_label.items() if v == chosen][0]
            st.session_state["selected_cafe_id"] = int(edit_id)

        # Initial values
        if manage_mode == "Bearbeiten" and edit_id is not None:
            row = cafes[cafes["id"] == int(edit_id)].iloc[0]
            init_name = str(row["name"])
            init_address = None if pd.isna(row["address"]) else str(row["address"])
            init_lat = float(row["lat"])
            init_lon = float(row["lon"])
            init_postcode = None if pd.isna(row["postcode"]) else str(row["postcode"])
            init_quartier = None if pd.isna(row["quartier"]) else str(row["quartier"])
            init_oh = None if pd.isna(row["opening_hours"]) else str(row["opening_hours"])
        else:
            init_name = ""
            init_address = ""
            init_lat = float(clicked["lat"]) if clicked else 47.5596
            init_lon = float(clicked["lng"]) if clicked else 7.5886
            init_postcode = None
            init_quartier = None
            init_oh = ""

        st.markdown("#### Adresse suchen (Nominatim)")
        addr_q = st.text_input("Suche", value="", placeholder="z.B. Steinentorstrasse 13")

        suggestions = []
        if addr_q.strip() and len(addr_q.strip()) >= 3:
            try:
                suggestions = nominatim_search_basel(addr_q.strip(), limit=8)
            except Exception as e:
                st.error(f"Nominatim Suche fehlgeschlagen: {e}")

        chosen_addr = None
        if suggestions:
            labels = [s["label"] for s in suggestions]
            chosen_label = st.selectbox("Vorschläge", labels, index=0)
            chosen_addr = next((s for s in suggestions if s["label"] == chosen_label), None)
            if st.button("Adresse übernehmen"):
                st.session_state["manage_address"] = chosen_addr["label"]
                st.session_state["manage_lat"] = float(chosen_addr["lat"])
                st.session_state["manage_lon"] = float(chosen_addr["lon"])
                st.session_state["manage_postcode"] = chosen_addr.get("postcode")
                st.session_state["manage_quartier"] = quartier_from_plz(chosen_addr.get("postcode"))

        name = st.text_input("Name", value=st.session_state.get("manage_name", init_name))
        address = st.text_input("Adresse", value=st.session_state.get("manage_address", init_address))
        lat = st.number_input("Latitude", value=float(st.session_state.get("manage_lat", init_lat)), format="%.6f")
        lon = st.number_input("Longitude", value=float(st.session_state.get("manage_lon", init_lon)), format="%.6f")

        postcode = st.session_state.get("manage_postcode", init_postcode) or extract_plz_from_text(address)
        quartier = st.session_state.get("manage_quartier", init_quartier) or quartier_from_plz(postcode)

        st.write(f"**PLZ:** {postcode or '-'}")
        st.write(f"**Quartier:** {quartier or '-'}")

        st.markdown("#### Öffnungszeiten")
        opening_hours = opening_hours_editor(
            key_prefix="manage_oh",
            initial_oh=init_oh if manage_mode == "Bearbeiten" else ""
        )

        if manage_mode == "Bearbeiten" and edit_id is not None:
            if st.button("Änderungen speichern"):
                if not name.strip():
                    st.error("Bitte Name eingeben.")
                else:
                    postcode2 = extract_plz_from_text(address) or postcode
                    quartier2 = quartier_from_plz(postcode2) or quartier
                    update_cafe(
                        cafe_id=int(edit_id),
                        name=name.strip(),
                        lat=float(lat),
                        lon=float(lon),
                        address=address.strip() or None,
                        postcode=postcode2,
                        quartier=quartier2,
                        opening_hours=opening_hours.strip() or None,
                    )
                    st.success("Café aktualisiert.")
                    for k in ["manage_name", "manage_address", "manage_lat", "manage_lon", "manage_postcode", "manage_quartier"]:
                        st.session_state.pop(k, None)
                    st.rerun()
        else:
            if st.button("Neues Café speichern"):
                if not name.strip():
                    st.error("Bitte Name eingeben.")
                else:
                    postcode2 = extract_plz_from_text(address) or postcode
                    quartier2 = quartier_from_plz(postcode2) or quartier
                    insert_user_cafe(
                        name=name.strip(),
                        lat=float(lat),
                        lon=float(lon),
                        address=address.strip() or None,
                        opening_hours=opening_hours.strip() or None,
                        postcode=postcode2,
                        quartier=quartier2,
                    )
                    st.success("Café gespeichert.")
                    for k in ["manage_name", "manage_address", "manage_lat", "manage_lon", "manage_postcode", "manage_quartier"]:
                        st.session_state.pop(k, None)
                    st.rerun()

    # ----------------------------
    # TAB: Rate
    # ----------------------------
    with tab_rate:
        st.caption("Café auswählen: auf Marker klicken ODER hier per Name suchen.")

        cafes_for_select = cafes.copy()
        cafes_for_select["label"] = cafes_for_select.apply(
            lambda r: f"{r['name']} — {r['address'] or ''}".strip(" —"),
            axis=1
        )

        search = st.text_input("Suche nach Name", value="")
        if search.strip():
            cafes_for_select = cafes_for_select[
                cafes_for_select["name"].str.contains(search.strip(), case=False, na=False)
            ]

        if cafes_for_select.empty:
            st.warning("Keine Cafés gefunden. Suchbegriff anpassen oder auf der Karte auswählen.")
        else:
            selected_cafe_id = st.session_state.get("selected_cafe_id")
            options = cafes_for_select.sort_values("name")[["id", "label"]].to_records(index=False)

            id_to_label = {int(i): str(l) for i, l in options}
            labels = list(id_to_label.values())

            default_label = id_to_label.get(int(selected_cafe_id)) if selected_cafe_id else labels[0]
            chosen_label = st.selectbox(
                "Café auswählen",
                labels,
                index=labels.index(default_label) if default_label in labels else 0
            )

            chosen_id = [k for k, v in id_to_label.items() if v == chosen_label][0]
            st.session_state["selected_cafe_id"] = int(chosen_id)

        selected_cafe_id = st.session_state.get("selected_cafe_id")

        if not selected_cafe_id:
            st.info("Bitte ein Café auswählen (Karte oder Dropdown), um es zu bewerten.")
        else:
            selected_row = cafes[cafes["id"] == selected_cafe_id].iloc[0]
            st.write(f"**Ausgewählt:** {selected_row['name']}")

            avgs = load_rating_averages_for_cafe(int(selected_cafe_id))
            n = int(avgs.get("n") or 0)

            if n == 0:
                st.info("Noch keine Bewertungen vorhanden.")
            else:
                st.markdown("#### Durchschnitt (alle Bewertungen)")
                st.write(f"**Overall:** {float(avgs['avg_overall']):.1f} ⭐  (Anzahl: {n})")

                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"Kaffee & Getränke: **{float(avgs['avg_coffee_quality']):.1f} ⭐**")
                    st.write(f"Ambiente: **{float(avgs['avg_ambience']):.1f} ⭐**")
                    st.write(f"Service: **{float(avgs['avg_service']):.1f} ⭐**")
                with c2:
                    st.write(f"Preis/Leistung: **{float(avgs['avg_value_for_money']):.1f} ⭐**")
                    st.write(f"Speisen: **{float(avgs['avg_food']):.1f} ⭐**")

                ratings_df = load_ratings_for_cafe(int(selected_cafe_id))
                if not ratings_df.empty:
                    st.markdown("#### Bewertungen (neueste zuerst)")
                    st.dataframe(
                        ratings_df.rename(columns={
                            "created_at": "Datum",
                            "reviewer_name": "Name",
                            "coffee_quality": "Kaffee & Getränke",
                            "ambience": "Ambiente",
                            "service": "Service",
                            "value_for_money": "Preis/Leistung",
                            "food": "Speisen",
                            "overall": "Overall",
                            "comment": "Kommentar",
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

            st.divider()
            st.markdown("#### Neue Bewertung")

            with st.form("rate_form"):
                reviewer_name = st.text_input("Dein Name", value="")

                coffee_quality = st.slider("Qualität von Kaffee & Getränken", 1, 5, 5)
                ambience = st.slider("Atmosphäre & Ambiente", 1, 5, 5)
                service = st.slider("Service", 1, 5, 5)
                value_for_money = st.slider("Preis-Leistungs-Verhältnis", 1, 5, 5)
                food = st.slider("Speisenangebot", 1, 5, 5)

                overall = (coffee_quality + ambience + service + value_for_money + food) / 5.0
                st.write(f"**Overall (diese Bewertung):** {overall:.1f} ⭐")

                comment = st.text_area("Kommentar (optional)", value="")
                submitted = st.form_submit_button("Bewertung speichern")

                if submitted:
                    insert_rating_multi(
                        int(selected_cafe_id),
                        (reviewer_name.strip() or "Anonym"),
                        int(coffee_quality),
                        int(ambience),
                        int(service),
                        int(value_for_money),
                        int(food),
                        comment.strip() or None,
                    )
                    st.success("Bewertung gespeichert.")
                    st.rerun()

# ----------------------------
# Tabelle (optional)
# ----------------------------
st.divider()
st.subheader("📋 Ergebnisliste")
cafes = load_cafes_with_stats()

filtered_tbl = cafes.copy()
filtered_tbl = filtered_tbl[(filtered_tbl["overall"].fillna(0.0)) >= float(min_overall)]

if selected_quartier != "(alle)":
    filtered_tbl = filtered_tbl[filtered_tbl["quartier"] == selected_quartier]

if only_with_hours:
    filtered_tbl = filtered_tbl[filtered_tbl["opening_hours"].fillna("").str.strip() != ""]

if hours_contains.strip():
    filtered_tbl = filtered_tbl[filtered_tbl["opening_hours"].fillna("").str.contains(hours_contains.strip(), case=False, na=False)]

if open_now:
    now = datetime.now()
    day = DAYS[now.weekday()]
    hhmm = f"{now.hour:02d}:{now.minute:02d}"
    filtered_tbl = filtered_tbl[
        filtered_tbl["opening_hours"].fillna("").apply(lambda oh: is_open_at(oh, day, hhmm))
    ]

if custom_time_filter:
    hhmm = f"{filter_time.hour:02d}:{filter_time.minute:02d}"
    filtered_tbl = filtered_tbl[
        filtered_tbl["opening_hours"].fillna("").apply(lambda oh: is_open_at(oh, filter_day, hhmm))
    ]

st.dataframe(
    filtered_tbl.sort_values(["overall", "rating_count"], ascending=False)[
        ["id", "name", "address", "opening_hours", "overall", "rating_count", "source", "postcode", "quartier"]
    ],
    use_container_width=True,
    hide_index=True
)

st.caption("Marker-Farben: blau=keine Bewertung, grün=Overall ≥4, orange=2–<4, rot=<2.")
st.caption("Datenquelle: OpenStreetMap (Overpass) + PLZ/Adresse via Nominatim. OSM-Attribution beachten.")
