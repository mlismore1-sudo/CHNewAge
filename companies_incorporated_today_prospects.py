import base64
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

st.set_page_config(page_title="Companies Incorporated Today", layout="wide")

TECH_SIC_CODES = {
    "62012", "62020", "58290", "58210", "61100", "61200", "61300", "61900",
    "62011", "62030", "62090", "63110", "63120", "71200", "72110", "72190",
    "72200", "71129",
}

HOLDINGS_SIC_CODES = {
    "64201", "64202", "64203", "64204", "64205", "64209", "66300",
}

TARGET_SIC_CODES = sorted(TECH_SIC_CODES | HOLDINGS_SIC_CODES)

EXCLUDED_DIRECTOR_COUNTRIES = {
    "PAKISTAN",
    "TURKEY",
    "CHINA",
    "NIGERIA",
}

COUNTRY_NORMALISATION = {
    "TURKIYE": "TURKEY",
    "PEOPLE'S REPUBLIC OF CHINA": "CHINA",
    "PRC": "CHINA",
    "P.R.C.": "CHINA",
}

TEAM_MEMBERS = ["Brad", "James"]
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "companies_cache.db"
APP_USER_AGENT = "streamlit-companies-house-today-app"
REQUEST_TIMEOUT = 30
ADVANCED_SEARCH_PAGE_SIZE = 1000
MAX_WORKERS = 6
RATE_WINDOW_SECONDS = 300
SAFE_REQUESTS_PER_WINDOW = 540

request_budget_lock = threading.Lock()
request_budget: Dict[str, List[float]] = {}
write_lock = threading.Lock()


def today_uk_str() -> str:
    return datetime.now().astimezone().date().isoformat()


def now_uk_str() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")


def get_api_keys() -> List[str]:
    keys: List[str] = []
    list_style_keys = st.secrets.get("COMPANIES_HOUSE_API_KEYS", [])
    if list_style_keys:
        keys.extend([str(k).strip() for k in list_style_keys if str(k).strip()])
    for key_name in ["CH_API_KEY_1", "CH_API_KEY_2", "CH_API_KEY_3"]:
        value = st.secrets.get(key_name, "")
        if value:
            keys.append(str(value).strip())
    deduped_keys = []
    seen = set()
    for key in keys:
        if key and key not in seen:
            deduped_keys.append(key)
            seen.add(key)
    return deduped_keys


def auth_header(api_key: str) -> Dict[str, str]:
    token = base64.b64encode(f"{api_key}:".encode()).decode()
    return {
        "Authorization": f"Basic {token}",
        "User-Agent": APP_USER_AGENT,
    }


def classify_sector(sic_codes: List[str]) -> Optional[str]:
    codes = {str(code) for code in (sic_codes or [])}
    if codes & HOLDINGS_SIC_CODES:
        return "Holdings"
    if codes & TECH_SIC_CODES:
        return "Tech"
    return None


def normalise_country(country: str) -> str:
    value = str(country or "").strip().upper()
    if not value:
        return ""
    return COUNTRY_NORMALISATION.get(value, value)


@st.cache_resource
def get_http_session() -> requests.Session:
    retry_strategy = Retry(
        total=4,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=20, pool_maxsize=20)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


@st.cache_resource
def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS director_country_cache (
            company_number TEXT PRIMARY KEY,
            excluded INTEGER NOT NULL,
            director_countries TEXT,
            checked_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_company_snapshots (
            run_date TEXT NOT NULL,
            company_number TEXT NOT NULL,
            company_name TEXT NOT NULL,
            sector TEXT NOT NULL,
            time_added_to_table TEXT NOT NULL,
            pull_order INTEGER NOT NULL,
            PRIMARY KEY (run_date, company_number)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_seen_companies (
            run_date TEXT NOT NULL,
            company_number TEXT NOT NULL,
            PRIMARY KEY (run_date, company_number)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prospect_list (
            company_number TEXT PRIMARY KEY,
            company_name TEXT NOT NULL,
            sector TEXT NOT NULL,
            added_by TEXT NOT NULL,
            added_at TEXT NOT NULL,
            notes TEXT NOT NULL DEFAULT ''
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshot_run_date ON daily_company_snapshots(run_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_seen_run_date ON daily_seen_companies(run_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prospect_added_by ON prospect_list(added_by)")
    conn.commit()
    return conn


def throttle_for_key(api_key: str) -> None:
    while True:
        now_ts = time.time()
        with request_budget_lock:
            timestamps = request_budget.setdefault(api_key, [])
            cutoff = now_ts - RATE_WINDOW_SECONDS
            while timestamps and timestamps[0] < cutoff:
                timestamps.pop(0)
            if len(timestamps) < SAFE_REQUESTS_PER_WINDOW:
                timestamps.append(now_ts)
                return
            sleep_for = max(0.25, RATE_WINDOW_SECONDS - (now_ts - timestamps[0]))
        time.sleep(min(sleep_for, 2.0))


def fetch_with_rotation(
    session: requests.Session,
    url: str,
    params: Optional[Dict[str, str]],
    api_keys: List[str],
    timeout: int = REQUEST_TIMEOUT,
) -> requests.Response:
    last_response = None
    for api_key in api_keys:
        throttle_for_key(api_key)
        response = session.get(
            url,
            headers=auth_header(api_key),
            params=params,
            timeout=timeout,
        )
        if response.status_code in (401, 429):
            last_response = response
            continue
        response.raise_for_status()
        return response
    if last_response is not None:
        last_response.raise_for_status()
    raise RuntimeError("No valid Companies House API keys were available.")


def get_cached_decisions_map(company_numbers: List[str]) -> Dict[str, bool]:
    if not company_numbers:
        return {}
    conn = get_db_connection()
    placeholders = ",".join("?" for _ in company_numbers)
    query = f"SELECT company_number, excluded FROM director_country_cache WHERE company_number IN ({placeholders})"
    rows = conn.execute(query, company_numbers).fetchall()
    return {str(company_number): bool(excluded) for company_number, excluded in rows}


def upsert_director_decision(company_number: str, excluded: bool, countries: List[str]) -> None:
    conn = get_db_connection()
    checked_at = now_uk_str()
    countries_str = ", ".join(sorted(set(countries)))
    with write_lock:
        conn.execute(
            """
            INSERT INTO director_country_cache (company_number, excluded, director_countries, checked_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(company_number) DO UPDATE SET
                excluded = excluded.excluded,
                director_countries = excluded.director_countries,
                checked_at = excluded.checked_at
            """,
            (company_number, int(excluded), countries_str, checked_at),
        )
        conn.commit()


def get_active_director_countries(session: requests.Session, company_number: str, api_keys: List[str]) -> List[str]:
    url = f"https://api.company-information.service.gov.uk/company/{company_number}/officers"
    params = {
        "register_view": "true",
        "register_type": "directors",
        "items_per_page": "100",
    }
    response = fetch_with_rotation(session, url, params, api_keys)
    payload = response.json()
    items = payload.get("items", []) or []
    countries: List[str] = []
    for officer in items:
        country = normalise_country(officer.get("country_of_residence", ""))
        if country:
            countries.append(country)
    return countries


def check_company_exclusion(company_number: str, api_keys: List[str]) -> Tuple[str, bool, List[str]]:
    session = get_http_session()
    try:
        countries = get_active_director_countries(session, company_number, api_keys)
        excluded = any(country in EXCLUDED_DIRECTOR_COUNTRIES for country in countries)
    except requests.RequestException:
        countries = []
        excluded = False
    upsert_director_decision(company_number, excluded, countries)
    return company_number, excluded, countries


def fetch_candidate_companies(api_keys: List[str], run_date: str) -> pd.DataFrame:
    session = get_http_session()
    url = "https://api.company-information.service.gov.uk/advanced-search/companies"
    start_index = 0
    rows = []

    while True:
        params = {
            "incorporated_from": run_date,
            "incorporated_to": run_date,
            "sic_codes": ",".join(TARGET_SIC_CODES),
            "size": str(ADVANCED_SEARCH_PAGE_SIZE),
            "start_index": str(start_index),
        }
        response = fetch_with_rotation(session, url, params, api_keys)
        payload = response.json()
        items = payload.get("items", []) or []

        for item in items:
            company_number = str(item.get("company_number", "")).strip()
            if not company_number:
                continue
            sic_codes = [str(code) for code in item.get("sic_codes", []) if code]
            sector = classify_sector(sic_codes)
            if not sector:
                continue
            rows.append(
                {
                    "company_number": company_number,
                    "company_name": item.get("company_name", ""),
                    "sector": sector,
                }
            )

        if len(items) < ADVANCED_SEARCH_PAGE_SIZE:
            break
        start_index += ADVANCED_SEARCH_PAGE_SIZE

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["company_number", "company_name", "sector"])
    return df.drop_duplicates(subset=["company_number"], keep="first").reset_index(drop=True)


def enrich_exclusions(candidate_df: pd.DataFrame, api_keys: List[str]) -> Tuple[pd.DataFrame, int, int]:
    if candidate_df.empty:
        return candidate_df.copy(), 0, 0

    company_numbers = candidate_df["company_number"].astype(str).tolist()
    cached_map = get_cached_decisions_map(company_numbers)
    cache_hits = len(cached_map)
    uncached_numbers = [n for n in company_numbers if n not in cached_map]

    if uncached_numbers:
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(uncached_numbers))) as executor:
            futures = {executor.submit(check_company_exclusion, company_number, api_keys): company_number for company_number in uncached_numbers}
            for future in as_completed(futures):
                company_number, excluded, _ = future.result()
                cached_map[company_number] = excluded

    checked_via_api = len(uncached_numbers)
    filtered_df = candidate_df[~candidate_df["company_number"].astype(str).map(cached_map).fillna(False)].copy()
    return filtered_df.reset_index(drop=True), checked_via_api, cache_hits


def load_snapshot(run_date: str) -> pd.DataFrame:
    conn = get_db_connection()
    query = """
        SELECT company_number, company_name, sector, time_added_to_table, pull_order
        FROM daily_company_snapshots
        WHERE run_date = ?
        ORDER BY pull_order DESC
    """
    df = pd.read_sql_query(query, conn, params=(run_date,))
    if df.empty:
        return pd.DataFrame(columns=["company_number", "company_name", "sector", "time_added_to_table", "pull_order"])
    df["time_added_to_table"] = pd.to_datetime(df["time_added_to_table"], errors="coerce")
    df["pull_order"] = pd.to_numeric(df["pull_order"], errors="coerce")
    return df


def load_seen(run_date: str) -> pd.DataFrame:
    conn = get_db_connection()
    return pd.read_sql_query("SELECT company_number FROM daily_seen_companies WHERE run_date = ?", conn, params=(run_date,))


def identify_new_rows(current_df: pd.DataFrame, seen_df: pd.DataFrame) -> pd.DataFrame:
    if current_df.empty:
        return current_df.copy()
    if seen_df.empty or "company_number" not in seen_df.columns:
        return current_df.copy()
    unseen = current_df[~current_df["company_number"].isin(seen_df["company_number"].astype(str))].copy()
    return unseen.reset_index(drop=True)


def save_daily_state(run_date: str, current_df: pd.DataFrame) -> None:
    conn = get_db_connection()
    export_df = current_df.copy()
    export_df["run_date"] = run_date
    export_df["time_added_to_table"] = export_df["time_added_to_table"].astype(str)
    with write_lock:
        conn.execute("DELETE FROM daily_company_snapshots WHERE run_date = ?", (run_date,))
        conn.execute("DELETE FROM daily_seen_companies WHERE run_date = ?", (run_date,))
        export_df[["run_date", "company_number", "company_name", "sector", "time_added_to_table", "pull_order"]].to_sql(
            "daily_company_snapshots", conn, if_exists="append", index=False
        )
        export_df[["run_date", "company_number"]].drop_duplicates().to_sql(
            "daily_seen_companies", conn, if_exists="append", index=False
        )
        conn.commit()


def get_cache_size() -> int:
    conn = get_db_connection()
    row = conn.execute("SELECT COUNT(*) FROM director_country_cache").fetchone()
    return int(row[0]) if row else 0


def add_to_prospect_list(df: pd.DataFrame, added_by: str, notes: str) -> int:
    if df.empty:
        return 0
    conn = get_db_connection()
    added_at = now_uk_str()
    rows_added = 0
    with write_lock:
        for _, row in df.iterrows():
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO prospect_list (company_number, company_name, sector, added_by, added_at, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row.get("company_number", "")).strip(),
                    str(row.get("company_name", "")).strip(),
                    str(row.get("sector", "")).strip(),
                    added_by,
                    added_at,
                    notes.strip(),
                ),
            )
            if cursor.rowcount and cursor.rowcount > 0:
                rows_added += 1
        conn.commit()
    return rows_added


def load_prospect_list() -> pd.DataFrame:
    conn = get_db_connection()
    query = """
        SELECT company_number, company_name, sector, added_by, added_at, notes
        FROM prospect_list
        ORDER BY added_at DESC, company_name ASC
    """
    df = pd.read_sql_query(query, conn)
    if df.empty:
        return pd.DataFrame(columns=["company_number", "company_name", "sector", "added_by", "added_at", "notes"])
    return df


def remove_from_prospect_list(company_numbers: List[str]) -> int:
    if not company_numbers:
        return 0
    conn = get_db_connection()
    placeholders = ",".join("?" for _ in company_numbers)
    with write_lock:
        cursor = conn.execute(f"DELETE FROM prospect_list WHERE company_number IN ({placeholders})", company_numbers)
        conn.commit()
    return int(cursor.rowcount or 0)


def render_table(df: pd.DataFrame, title: str) -> None:
    st.subheader(title)
    if df.empty:
        st.info("No companies to show yet.")
        return
    display_df = (
        df.sort_values("time_added_to_table", ascending=False, kind="stable")[["company_name", "sector", "time_added_to_table"]]
        .rename(columns={
            "company_name": "Company Name",
            "sector": "Sector",
            "time_added_to_table": "Time Added To Table",
        })
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_selectable_results_table(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=df.columns)

    editor_df = df[["company_number", "company_name", "sector", "time_added_to_table"]].copy()
    editor_df.insert(0, "Select", False)
    editor_df = editor_df.rename(columns={
        "company_number": "Company Number",
        "company_name": "Company Name",
        "sector": "Sector",
        "time_added_to_table": "Time Added To Table",
    })

    edited_df = st.data_editor(
        editor_df,
        key=key,
        hide_index=True,
        use_container_width=True,
        disabled=["Company Number", "Company Name", "Sector", "Time Added To Table"],
        column_config={
            "Select": st.column_config.CheckboxColumn("Select", help="Tick to add to prospect list"),
        },
    )

    selected = edited_df[edited_df["Select"]].copy()
    if selected.empty:
        return pd.DataFrame(columns=["company_number", "company_name", "sector", "time_added_to_table"])

    selected = selected.rename(columns={
        "Company Number": "company_number",
        "Company Name": "company_name",
        "Sector": "sector",
        "Time Added To Table": "time_added_to_table",
    })
    return selected[["company_number", "company_name", "sector", "time_added_to_table"]].reset_index(drop=True)


@st.cache_data
def convert_current_df_to_download_csv(df: pd.DataFrame) -> bytes:
    return (
        df.sort_values("time_added_to_table", ascending=False, kind="stable")[["company_name", "sector", "time_added_to_table"]]
        .rename(columns={
            "company_name": "Company Name",
            "sector": "Sector",
            "time_added_to_table": "Time Added To Table",
        })
        .to_csv(index=False)
        .encode("utf-8")
    )


@st.cache_data
def convert_prospect_df_to_download_csv(df: pd.DataFrame) -> bytes:
    return (
        df.rename(columns={
            "company_number": "Company Number",
            "company_name": "Company Name",
            "sector": "Sector",
            "added_by": "Added By",
            "added_at": "Added At",
            "notes": "Notes",
        })
        .to_csv(index=False)
        .encode("utf-8")
    )


def build_current_day_dataset(api_keys: List[str], run_date: str) -> Tuple[pd.DataFrame, pd.DataFrame, int, int]:
    candidate_df = fetch_candidate_companies(api_keys, run_date)
    filtered_df, checked_via_api, cache_hits = enrich_exclusions(candidate_df, api_keys)

    existing_df = load_snapshot(run_date)
    timestamp_now = now_uk_str()
    if existing_df.empty:
        current_df = filtered_df.copy()
        if not current_df.empty:
            current_df["time_added_to_table"] = timestamp_now
            current_df["pull_order"] = range(len(current_df))
    else:
        existing_numbers = set(existing_df["company_number"].astype(str)) if "company_number" in existing_df.columns else set()
        new_rows = filtered_df[~filtered_df["company_number"].astype(str).isin(existing_numbers)].copy()
        if not new_rows.empty:
            new_rows["time_added_to_table"] = timestamp_now
            new_rows["pull_order"] = range(len(new_rows))
        current_df = pd.concat([new_rows, existing_df], ignore_index=True)
        current_df = current_df.drop_duplicates(subset=["company_number"], keep="first").reset_index(drop=True)

    if current_df.empty:
        current_df = pd.DataFrame(columns=["company_number", "company_name", "sector", "time_added_to_table", "pull_order"])

    seen_df = load_seen(run_date)
    new_df = identify_new_rows(current_df, seen_df)
    save_daily_state(run_date, current_df)
    return current_df, new_df, checked_via_api, cache_hits


def main() -> None:
    st.title("Companies Incorporated Today")
    st.caption(
        "Lead prospect engine for Brad and James. Shows companies incorporated today that match your Tech and Holdings SIC code lists, excluding companies with active directors whose country of residence is Pakistan, Turkey, China, or Nigeria."
    )

    api_keys = get_api_keys()
    if not api_keys:
        st.error("Add COMPANIES_HOUSE_API_KEYS or CH_API_KEY_1/2/3 to your Streamlit secrets before running the app.")
        st.stop()

    run_date = today_uk_str()

    st.sidebar.header("Controls")
    st.sidebar.write(f"Run date: {run_date}")
    st.sidebar.write(f"API keys loaded: {len(api_keys)}")
    selected_user = st.sidebar.selectbox("Working as", TEAM_MEMBERS, index=0)
    refresh = st.sidebar.button("Refresh now", type="primary")

    if refresh or "latest_df" not in st.session_state:
        started = time.perf_counter()
        current_df, new_df, checked_via_api, cache_hits = build_current_day_dataset(api_keys, run_date)
        elapsed = time.perf_counter() - started

        st.session_state["latest_df"] = current_df
        st.session_state["new_df"] = new_df
        st.session_state["checked_this_run"] = checked_via_api
        st.session_state["cache_hits_this_run"] = cache_hits
        st.session_state["cache_size"] = get_cache_size()
        st.session_state["elapsed_seconds"] = round(elapsed, 2)
        st.session_state["last_refresh"] = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    else:
        current_df = load_snapshot(run_date)
        st.session_state.setdefault("latest_df", current_df)
        st.session_state.setdefault(
            "new_df",
            pd.DataFrame(columns=current_df.columns if not current_df.empty else ["company_number", "company_name", "sector", "time_added_to_table", "pull_order"]),
        )
        st.session_state.setdefault("checked_this_run", 0)
        st.session_state.setdefault("cache_hits_this_run", 0)
        st.session_state.setdefault("cache_size", get_cache_size())
        st.session_state.setdefault("elapsed_seconds", 0.0)
        st.session_state.setdefault("last_refresh", "Not refreshed in this session")

    current_df = st.session_state.get("latest_df", pd.DataFrame())
    new_df = st.session_state.get("new_df", pd.DataFrame())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total pulled today", int(len(current_df)))
    c2.metric("New on latest refresh", int(len(new_df)))
    c3.metric("Officer checks this run", int(st.session_state.get("checked_this_run", 0)))
    c4.metric("Cache hits this run", int(st.session_state.get("cache_hits_this_run", 0)))
    c5.metric("Refresh seconds", float(st.session_state.get("elapsed_seconds", 0.0)))

    st.write(f"Working as: {selected_user}")
    st.write(f"Last refresh: {st.session_state.get('last_refresh', 'Unknown')}")
    st.write(f"Persistent cached company decisions: {int(st.session_state.get('cache_size', 0))}")

    st.subheader("New companies found on the latest refresh")
    selected_new = render_selectable_results_table(new_df, key="new_companies_editor")

    st.subheader("All companies pulled so far today")
    selected_all = render_selectable_results_table(current_df, key="all_companies_editor")

    combined_selected = pd.concat([selected_new, selected_all], ignore_index=True).drop_duplicates(subset=["company_number"], keep="first") if not selected_new.empty or not selected_all.empty else pd.DataFrame(columns=["company_number", "company_name", "sector", "time_added_to_table"])

    st.subheader("Add selected companies to prospect list")
    notes = st.text_area("Optional notes for these selected companies", key="prospect_notes", placeholder="Example: Good fit for treasury stack, follow up next week")
    add_disabled = combined_selected.empty
    if st.button("Add selected to prospect list", type="primary", disabled=add_disabled):
        rows_added = add_to_prospect_list(combined_selected, selected_user, notes)
        if rows_added > 0:
            st.success(f"Added {rows_added} compan{'y' if rows_added == 1 else 'ies'} to the prospect list.")
        else:
            st.info("No new companies were added. They may already be in the prospect list.")

    if combined_selected.empty:
        st.caption("Select one or more companies above to add them to the shared prospect list.")
    else:
        st.caption(f"Currently selected companies ready to add: {len(combined_selected)}")

    st.divider()
    st.subheader("Shared prospect list")
    prospect_df = load_prospect_list()
    if prospect_df.empty:
        st.info("No companies have been added to the prospect list yet.")
    else:
        prospect_display = prospect_df.rename(columns={
            "company_number": "Company Number",
            "company_name": "Company Name",
            "sector": "Sector",
            "added_by": "Added By",
            "added_at": "Added At",
            "notes": "Notes",
        }).copy()
        prospect_display.insert(0, "Remove", False)

        edited_prospect_df = st.data_editor(
            prospect_display,
            key="prospect_list_editor",
            hide_index=True,
            use_container_width=True,
            disabled=["Company Number", "Company Name", "Sector", "Added By", "Added At", "Notes"],
            column_config={
                "Remove": st.column_config.CheckboxColumn("Remove", help="Tick rows to remove from prospect list"),
            },
        )

        selected_to_remove = edited_prospect_df[edited_prospect_df["Remove"]]
        if st.button("Remove selected from prospect list", disabled=selected_to_remove.empty):
            company_numbers_to_remove = selected_to_remove["Company Number"].astype(str).tolist()
            removed_count = remove_from_prospect_list(company_numbers_to_remove)
            if removed_count > 0:
                st.success(f"Removed {removed_count} compan{'y' if removed_count == 1 else 'ies'} from the prospect list.")
                st.rerun()
            else:
                st.info("No companies were removed.")

        prospect_csv = convert_prospect_df_to_download_csv(prospect_df)
        st.download_button(
            label="Download shared prospect list as CSV",
            data=prospect_csv,
            file_name=f"prospect_list_{run_date}.csv",
            mime="text/csv",
            key="download_prospect_csv_button",
        )

    if not current_df.empty:
        csv_bytes = convert_current_df_to_download_csv(current_df)
        st.download_button(
            label="Download today’s filtered results as CSV",
            data=csv_bytes,
            file_name=f"companies_incorporated_{run_date}.csv",
            mime="text/csv",
            key="download_current_csv_button",
        )

    with st.expander("Suggested .streamlit/secrets.toml"):
        secrets_example = """COMPANIES_HOUSE_API_KEYS = [
  \"your-first-key\",
  \"your-second-key\",
  \"your-third-key\"
]

# Optional legacy format
# CH_API_KEY_1 = \"your-first-key\"
# CH_API_KEY_2 = \"your-second-key\"
# CH_API_KEY_3 = \"your-third-key\"
"""
        st.code(secrets_example, language="toml")

    with st.expander("Architecture notes"):
        st.markdown(
            """
- Uses persistent SQLite caching across all days for company exclusion decisions.
- Reuses a pooled HTTP session with retries and backoff.
- Runs uncached officer lookups concurrently with a bounded worker pool.
- Applies rate-aware throttling to stay below Companies House API limits.
- Includes a shared prospect list for Brad and James, stored in SQLite and downloadable as CSV.
            """
        )


if __name__ == "__main__":
    main()
