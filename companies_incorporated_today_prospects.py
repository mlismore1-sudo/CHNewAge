import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

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
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
LEADS_DIR = DATA_DIR / "leads"
LEADS_DIR.mkdir(exist_ok=True)
TEAM_MEMBERS = ["Brad", "James"]
QUICK_ADD_DEFAULT = 15
RESULT_COLUMNS = ["company_number", "company_name", "sector", "time_added_to_table", "pull_order"]
LEAD_COLUMNS = ["company_number", "company_name", "sector", "added_by", "added_at"]


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
        "User-Agent": "streamlit-companies-house-today-app",
    }


def classify_sector(sic_codes: List[str]) -> Optional[str]:
    codes = {str(code) for code in (sic_codes or [])}
    if codes & HOLDINGS_SIC_CODES:
        return "Holdings"
    if codes & TECH_SIC_CODES:
        return "Tech"
    return None


@st.cache_resource(show_spinner=False)
def get_session() -> requests.Session:
    return requests.Session()


def fetch_with_rotation(url: str, params: Dict[str, str], api_keys: List[str], timeout: int = 30) -> requests.Response:
    session = get_session()
    last_response = None
    for api_key in api_keys:
        response = session.get(url, headers=auth_header(api_key), params=params, timeout=timeout)
        if response.status_code in (401, 429):
            last_response = response
            continue
        response.raise_for_status()
        return response
    if last_response is not None:
        last_response.raise_for_status()
    raise RuntimeError("No valid Companies House API keys were available.")


def fetch_companies_incorporated_today(api_keys: List[str], run_date: str) -> pd.DataFrame:
    url = "https://api.company-information.service.gov.uk/advanced-search/companies"
    start_index = 0
    page_size = 5000
    rows = []
    pull_counter = 0

    while True:
        params = {
            "incorporated_from": run_date,
            "incorporated_to": run_date,
            "sic_codes": ",".join(TARGET_SIC_CODES),
            "size": str(page_size),
            "start_index": str(start_index),
        }
        response = fetch_with_rotation(url, params, api_keys)
        payload = response.json()
        items = payload.get("items", []) or []

        for item in items:
            sic_codes = [str(code) for code in item.get("sic_codes", []) if code]
            sector = classify_sector(sic_codes)
            if not sector:
                continue
            rows.append({
                "company_number": str(item.get("company_number", "")),
                "company_name": str(item.get("company_name", "")),
                "sector": sector,
                # ✅ FIX: timestamp is recorded at fetch time; will be overridden
                # by the preserved value for already-known companies during merge.
                "time_added_to_table": now_uk_str(),
                "pull_order": pull_counter,
            })
            pull_counter += 1

        if len(items) < page_size:
            break
        start_index += page_size

    if not rows:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    return (
        df.sort_values("pull_order", ascending=False, kind="stable")
        .drop_duplicates(subset=["company_number"], keep="first")
        .reset_index(drop=True)
    )


def get_store_paths(run_date: str) -> Tuple[Path, Path]:
    snapshot_path = DATA_DIR / f"companies_{run_date}.csv"
    seen_path = DATA_DIR / f"seen_{run_date}.csv"
    return snapshot_path, seen_path


def lead_file_path(person: str, run_date: str) -> Path:
    return LEADS_DIR / f"{person.strip().lower()}_leads_{run_date}.csv"


@st.cache_data(show_spinner=False)
def load_results_csv(path_str: str, mtime: float) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.read_csv(path, dtype={c: "string" for c in RESULT_COLUMNS}).fillna("")


@st.cache_data(show_spinner=False)
def load_leads_csv(path_str: str, mtime: float) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame(columns=LEAD_COLUMNS)
    return pd.read_csv(path, dtype={c: "string" for c in LEAD_COLUMNS}).fillna("")


def load_results(path: Path) -> pd.DataFrame:
    mtime = path.stat().st_mtime if path.exists() else 0.0
    return load_results_csv(str(path), mtime)


def load_leads(person: str, run_date: str) -> pd.DataFrame:
    path = lead_file_path(person, run_date)
    mtime = path.stat().st_mtime if path.exists() else 0.0
    return load_leads_csv(str(path), mtime)


def identify_new_rows(current_df: pd.DataFrame, seen_df: pd.DataFrame) -> pd.DataFrame:
    if current_df.empty:
        return current_df.copy()
    if seen_df.empty or "company_number" not in seen_df.columns:
        return current_df.copy()
    unseen = current_df[~current_df["company_number"].isin(seen_df["company_number"].astype(str))].copy()
    return unseen.reset_index(drop=True)


def save_state(current_df: pd.DataFrame, snapshot_path: Path, seen_path: Path) -> None:
    current_df.to_csv(snapshot_path, index=False)
    current_df.to_csv(seen_path, index=False)


def add_company_to_leads(person: str, run_date: str, row: pd.Series, existing_leads: pd.DataFrame) -> bool:
    path = lead_file_path(person, run_date)
    company_number = str(row.get("company_number", "")).strip()
    if not company_number:
        return False

    if not existing_leads.empty and company_number in set(existing_leads["company_number"].astype(str)):
        return False

    new_row = pd.DataFrame([{
        "company_number": company_number,
        "company_name": str(row.get("company_name", "")).strip(),
        "sector": str(row.get("sector", "")).strip(),
        "added_by": person,
        "added_at": now_uk_str(),
    }], columns=LEAD_COLUMNS)

    if path.exists():
        new_row.to_csv(path, mode="a", index=False, header=False)
    else:
        new_row.to_csv(path, index=False)
    return True


@st.cache_data(show_spinner=False)
def convert_results_csv_bytes(df: pd.DataFrame) -> bytes:
    if df.empty:
        return b""
    export_df = df[["company_name", "sector", "time_added_to_table"]].rename(columns={
        "company_name": "Company Name",
        "sector": "Sector",
        "time_added_to_table": "Time Added To Table",
    })
    return export_df.to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False)
def convert_leads_csv_bytes(df: pd.DataFrame) -> bytes:
    if df.empty:
        return b""
    export_df = df.rename(columns={
        "company_number": "Company Number",
        "company_name": "Company Name",
        "sector": "Sector",
        "added_by": "Added By",
        "added_at": "Added At",
    })
    return export_df.to_csv(index=False).encode("utf-8")


def get_sorted_current_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values("pull_order", ascending=False, kind="stable").reset_index(drop=True)


def render_quick_add(df: pd.DataFrame, person: str, run_date: str, existing_leads: pd.DataFrame) -> None:
    st.subheader(f"Quick add to {person}'s leads")
    if df.empty:
        st.info("No companies available to add.")
        return

    existing_numbers = set(existing_leads["company_number"].astype(str)) if not existing_leads.empty else set()

    for idx, (_, row) in enumerate(df.iterrows()):
        company_number = str(row.get("company_number", "")).strip()
        already_added = company_number in existing_numbers
        c1, c2, c3, c4 = st.columns([5, 1.2, 2, 0.9])
        c1.write(f"**{row['company_name']}**")
        c2.write(str(row["sector"]))
        c3.write(str(row["time_added_to_table"]))
        if already_added:
            c4.caption("Added")
        else:
            if c4.button("Add", key=f"add_{person}_{company_number}_{idx}"):
                added = add_company_to_leads(person, run_date, row, existing_leads)
                if added:
                    st.rerun()


def merge_preserving_timestamps(fetched_df: pd.DataFrame, existing_df: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ FIX: Merge fetched results with the existing snapshot, keeping the
    original time_added_to_table for companies that already exist in the
    snapshot. Only genuinely new companies get the current fetch timestamp.
    """
    if existing_df.empty:
        return fetched_df.copy()

    existing_lookup = existing_df.set_index("company_number")[["time_added_to_table", "pull_order"]]
    existing_numbers = set(existing_df["company_number"].astype(str))

    # Split fetched rows into new vs already-known
    new_rows = fetched_df[~fetched_df["company_number"].astype(str).isin(existing_numbers)].copy()

    # For already-known rows, restore their original timestamps and pull_order
    known_rows = fetched_df[fetched_df["company_number"].astype(str).isin(existing_numbers)].copy()
    known_rows["time_added_to_table"] = known_rows["company_number"].map(
        existing_lookup["time_added_to_table"]
    )
    known_rows["pull_order"] = known_rows["company_number"].map(
        existing_lookup["pull_order"].astype(int)
    )

    merged = pd.concat([new_rows, known_rows], ignore_index=True)
    return merged.drop_duplicates(subset=["company_number"], keep="first").reset_index(drop=True)


def main() -> None:
    st.title("Companies Incorporated Today")
    st.caption("Ultra-fast version v2: session reuse, narrower cache invalidation, minimal rendering, top-15 quick add list.")

    api_keys = get_api_keys()
    if not api_keys:
        st.error("Add COMPANIES_HOUSE_API_KEYS or CH_API_KEY_1/2/3 to your Streamlit secrets before running the app.")
        st.stop()

    run_date = today_uk_str()
    snapshot_path, seen_path = get_store_paths(run_date)

    st.sidebar.header("Controls")
    selected_user = st.sidebar.selectbox("Working as", TEAM_MEMBERS, index=0)
    refresh = st.sidebar.button("Refresh now", type="primary")

    if refresh or not snapshot_path.exists():
        fetched_df = fetch_companies_incorporated_today(api_keys, run_date)
        existing_df = load_results(snapshot_path)

        # ✅ FIX: Use merge function that preserves original timestamps
        current_df = merge_preserving_timestamps(fetched_df, existing_df)

        seen_df = load_results(seen_path)
        new_df = identify_new_rows(current_df, seen_df)
        save_state(current_df, snapshot_path, seen_path)
        st.session_state["latest_df"] = current_df
        st.session_state["sorted_df"] = get_sorted_current_df(current_df)
        st.session_state["new_df"] = new_df
        st.session_state["last_refresh"] = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    else:
        current_df = load_results(snapshot_path)
        st.session_state.setdefault("latest_df", current_df)
        st.session_state.setdefault("sorted_df", get_sorted_current_df(current_df))
        st.session_state.setdefault("new_df", pd.DataFrame(columns=RESULT_COLUMNS))
        st.session_state.setdefault("last_refresh", "Not refreshed in this session")

    current_df = st.session_state.get("latest_df", pd.DataFrame(columns=RESULT_COLUMNS))
    sorted_df = st.session_state.get("sorted_df", pd.DataFrame(columns=RESULT_COLUMNS))
    leads_df = load_leads(selected_user, run_date)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total pulled today", int(len(current_df)))
    c2.metric(f"{selected_user}'s leads today", int(len(leads_df)))
    c3.metric("Quick add rows", QUICK_ADD_DEFAULT)

    st.caption(f"Working as {selected_user} | Last refresh: {st.session_state.get('last_refresh', 'Unknown')}")

    newest_df = sorted_df.head(QUICK_ADD_DEFAULT).reset_index(drop=True) if not sorted_df.empty else sorted_df
    render_quick_add(newest_df, selected_user, run_date, leads_df)

    with st.expander(f"{selected_user}'s leads for today", expanded=False):
        if leads_df.empty:
            st.info(f"No leads saved yet for {selected_user}.")
        else:
            leads_display = leads_df.rename(columns={
                "company_number": "Company Number",
                "company_name": "Company Name",
                "sector": "Sector",
                "added_by": "Added By",
                "added_at": "Added At",
            })
            st.dataframe(leads_display, use_container_width=True, hide_index=True)
            st.download_button(
                label=f"Download {selected_user}'s leads CSV",
                data=convert_leads_csv_bytes(leads_df),
                file_name=f"{selected_user.lower()}_leads_{run_date}.csv",
                mime="text/csv",
                key=f"download_{selected_user.lower()}_leads",
            )

    with st.expander("Today's results CSV", expanded=False):
        if not current_df.empty:
            st.download_button(
                label="Download today's results as CSV",
                data=convert_results_csv_bytes(current_df),
                file_name=f"companies_incorporated_{run_date}.csv",
                mime="text/csv",
                key="download_results_csv",
            )
        else:
            st.info("No results available yet.")

    with st.expander("Full table", expanded=False):
        if current_df.empty:
            st.info("No companies to show yet.")
        else:
            preview_df = sorted_df[["company_name", "sector", "time_added_to_table"]].rename(columns={
                "company_name": "Company Name",
                "sector": "Sector",
                "time_added_to_table": "Time Added To Table",
            })
            st.dataframe(preview_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
