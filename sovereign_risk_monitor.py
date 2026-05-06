import os
import sys
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import pytz
import requests
import yfinance as yf
from bs4 import BeautifulSoup


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_FILE = "raw_daily_market_data.csv"
TIMEZONE = pytz.timezone("US/Eastern")

# Keep this True while debugging BondBloX.
# Once yields look correct, change it to False.
DEBUG_BONDBLOX = True

UST_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/TextView?type=daily_treasury_yield_curve"
    "&field_tdr_date_value=2026"
)

COUNTRIES = [
    {
        "Country": "Brazil",
        "Bond_Name": "Brazil 6.625% 15-Mar-2035 USD",
        "Bond_URL": "https://bondblox.com/bond-market/Brazil,-Federative-Republic-Of-(Government)-US105756CL22",
        "FX_Ticker": "BRL=X",
        "FX_URL": "https://finance.yahoo.com/quote/BRL=X/",
    },
    {
        "Country": "Colombia",
        "Bond_Name": "Colombia 8.500% 25-Apr-2035 USD",
        "Bond_URL": "https://bondblox.com/bond-market/Colombia,-Republic-Of-(Government)-US195325ES00",
        "FX_Ticker": "COP=X",
        "FX_URL": "https://finance.yahoo.com/quote/COP=X/",
    },
    {
        "Country": "Chile",
        "Bond_Name": "Chile 3.500% 31-Jan-2034 USD",
        "Bond_URL": "https://bondblox.com/bond-market/Chile,-Republic-Of-(Government)-US168863DV76",
        "FX_Ticker": "CLP=X",
        "FX_URL": "https://finance.yahoo.com/quote/CLP=X/",
    },
    {
        "Country": "Peru",
        "Bond_Name": "Peru 5.375% 8-Feb-2035 USD",
        "Bond_URL": "https://bondblox.com/bond-market/Peru,-Republic-Of-(Government)-US715638EB48",
        "FX_Ticker": "PEN=X",
        "FX_URL": "https://finance.yahoo.com/quote/PEN=X/",
    },
    {
        "Country": "Argentina",
        "Bond_Name": "Argentina 4.125% 9-Jul-2035 USD",
        "Bond_URL": "https://bondblox.com/bond-market/Argentina,-Republic-Of-(Government)-US040114HT09",
        "FX_Ticker": "ARS=X",
        "FX_URL": "https://finance.yahoo.com/quote/ARS=X/",
    },
]


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


# ============================================================
# HELPERS
# ============================================================

def get_now_et() -> datetime:
    return datetime.now(TIMEZONE)


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    text = str(value).strip()
    text = text.replace("%", "").replace(",", "")

    if text.upper() in {"", "N/A", "NA", "--", "---.--", "NAN", "NOT APPLICABLE"}:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def looks_like_decimal_number(text: str) -> bool:
    """
    BondBloX has footnote-only lines like:
        3

    The actual yield usually appears as:
        6.06
        7.91
        5.28

    This function forces the candidate to look like a decimal value,
    which prevents us from accidentally using the footnote marker.
    """
    cleaned = str(text).strip().replace("%", "").replace(",", "")
    return cleaned.replace(".", "", 1).isdigit() and "." in cleaned


def is_footnote_marker(text: str) -> bool:
    cleaned = str(text).strip()
    return cleaned in {"1", "2", "3", "4", "5", "6", "7", "8", "9"}


def request_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    return response.text


# ============================================================
# BOND YIELD FETCH
# ============================================================

def print_debug_block(country_or_url: str, block: list[str]) -> None:
    if not DEBUG_BONDBLOX:
        return

    logger.info("========== BondBloX DEBUG BLOCK: %s ==========", country_or_url)

    for idx, line in enumerate(block):
        logger.info("[%02d] %s", idx, line)

    logger.info("======== END BondBloX DEBUG BLOCK: %s ========", country_or_url)


def extract_bondblox_yield_from_text(
    page_text: str,
    country_or_url: str = "Unknown",
) -> Tuple[Optional[float], str]:
    """
    Extracts the BondBloX yield from the Yield Analysis section.

    The relevant page text often looks like:

        Yield Analysis
        Yield
        3
        6.06
        Yield To Call
        Not Applicable

    In that structure:
        3    = footnote marker
        6.06 = actual yield

    Therefore:
        - Do NOT grab first percentage on page.
        - Do NOT grab coupon.
        - Do NOT grab standalone integer after Yield.
        - Prefer first decimal numeric value after Yield label.
    """

    lines = [
        line.strip()
        for line in page_text.splitlines()
        if line.strip()
    ]

    try:
        start_idx = next(
            i for i, line in enumerate(lines)
            if line.lower() == "yield analysis"
        )
    except StopIteration:
        return None, "Could not find 'Yield Analysis' section"

    end_idx = len(lines)

    stop_headers = {
        "bond additional information",
        "bond information",
        "price/yield",
        "issuer information",
    }

    for i in range(start_idx + 1, len(lines)):
        lower = lines[i].lower()

        if any(lower.startswith(header) for header in stop_headers):
            end_idx = i
            break

    block = lines[start_idx:end_idx]

    print_debug_block(country_or_url, block)

    for i, line in enumerate(block):
        normalized = line.lower().replace(" ", "")

        is_yield_label = (
            normalized == "yield"
            or normalized.startswith("yield^")
            or normalized.startswith("yield{")
        )

        is_not_yield_to_call = "tocall" not in normalized

        if is_yield_label and is_not_yield_to_call:
            candidates = block[i + 1:]

            for candidate in candidates:
                candidate_clean = candidate.strip()

                # Stop if we hit the next field.
                next_field = candidate_clean.lower().replace(" ", "")
                if next_field.startswith("yieldtocall"):
                    break

                # Skip footnote-only lines like 3.
                if is_footnote_marker(candidate_clean):
                    logger.info(
                        "Skipping likely BondBloX footnote marker after Yield label: %s",
                        candidate_clean,
                    )
                    continue

                # Require decimal to avoid footnote/coupon/label accidents.
                if not looks_like_decimal_number(candidate_clean):
                    continue

                value = safe_float(candidate_clean)

                if value is not None and 0 < value < 100:
                    return value, "OK"

            return None, "Found Yield label but no decimal numeric yield after it"

    return None, "Could not find Yield field inside Yield Analysis section"


def fetch_bond_yield(country: str, url: str) -> Tuple[Optional[float], str]:
    try:
        html = request_html(url)
        soup = BeautifulSoup(html, "lxml")
        page_text = soup.get_text(separator="\n")
        return extract_bondblox_yield_from_text(page_text, country)

    except requests.HTTPError as e:
        return None, f"BondBloX HTTP error: {e}"
    except requests.RequestException as e:
        return None, f"BondBloX request error: {e}"
    except Exception as e:
        return None, f"BondBloX parsing error: {e}"


# ============================================================
# UST FETCH
# ============================================================

def clean_treasury_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    if isinstance(cleaned.columns, pd.MultiIndex):
        cleaned.columns = [
            " ".join(str(part).strip() for part in col if str(part).strip())
            for col in cleaned.columns
        ]
    else:
        cleaned.columns = [str(col).strip() for col in cleaned.columns]

    return cleaned


def find_ust_10y_column(columns) -> Optional[str]:
    for col in columns:
        normalized = str(col).strip().lower()
        if normalized in {"10 yr", "10 year", "10-year", "10y", "10 y"}:
            return col

    for col in columns:
        normalized = str(col).strip().lower()
        if "10" in normalized and "yr" in normalized:
            return col

    return None


def fetch_ust_10y() -> Tuple[Optional[float], str]:
    try:
        tables = pd.read_html(UST_URL)

        if not tables:
            return None, "No tables found on Treasury page"

        treasury_table = None

        for table in tables:
            cleaned = clean_treasury_columns(table)
            columns_lower = [str(c).strip().lower() for c in cleaned.columns]

            has_date = "date" in columns_lower
            has_10y = any("10" in c and "yr" in c for c in columns_lower)

            if has_date and has_10y:
                treasury_table = cleaned
                break

        if treasury_table is None:
            return None, "Could not find Treasury yield curve table"

        date_col = next(
            col for col in treasury_table.columns
            if str(col).strip().lower() == "date"
        )

        ten_y_col = find_ust_10y_column(treasury_table.columns)

        if ten_y_col is None:
            return None, f"Could not identify 10Y column: {list(treasury_table.columns)}"

        treasury_table[date_col] = pd.to_datetime(treasury_table[date_col], errors="coerce")
        treasury_table[ten_y_col] = pd.to_numeric(treasury_table[ten_y_col], errors="coerce")

        treasury_table = treasury_table.dropna(subset=[date_col, ten_y_col])
        treasury_table = treasury_table.sort_values(date_col)

        if treasury_table.empty:
            return None, "Treasury table had no valid 10Y observations"

        latest_row = treasury_table.iloc[-1]
        latest_date = latest_row[date_col].date()
        latest_10y = float(latest_row[ten_y_col])

        return latest_10y, f"OK. Latest Treasury observation date: {latest_date}"

    except Exception as e:
        return None, f"Treasury parsing error: {e}"


# ============================================================
# FX FETCH
# ============================================================

def calculate_pct_change(latest: float, previous: float) -> Optional[float]:
    if previous is None or previous == 0:
        return None

    return ((latest / previous) - 1) * 100


def fetch_fx_data(ticker: str) -> Dict[str, Any]:
    try:
        fx = yf.Ticker(ticker)
        hist = fx.history(period="1y", auto_adjust=False)

        if hist.empty or "Close" not in hist.columns:
            return {"status": "FAIL", "error": "No FX history returned from yfinance"}

        closes = hist["Close"].dropna()

        if closes.empty:
            return {"status": "FAIL", "error": "FX close series is empty"}

        latest = float(closes.iloc[-1])

        fx_5d = calculate_pct_change(latest, float(closes.iloc[-6])) if len(closes) >= 6 else None
        fx_1m = calculate_pct_change(latest, float(closes.iloc[-22])) if len(closes) >= 22 else None

        current_year = closes.index[-1].year
        ytd_closes = closes[closes.index.year == current_year]

        fx_ytd = None
        if not ytd_closes.empty:
            fx_ytd = calculate_pct_change(latest, float(ytd_closes.iloc[0]))

        return {
            "status": "OK",
            "spot": latest,
            "5d": fx_5d,
            "1m": fx_1m,
            "ytd": fx_ytd,
        }

    except Exception as e:
        return {"status": "FAIL", "error": f"FX fetch/parsing error: {e}"}


# ============================================================
# ROW BUILDING
# ============================================================

def build_daily_row(
    country_cfg: Dict[str, str],
    ust_yield: Optional[float],
    ust_note: str,
) -> Dict[str, Any]:
    now = get_now_et()
    notes = []
    status = "OK"

    country = country_cfg["Country"]

    bond_yield, bond_note = fetch_bond_yield(country, country_cfg["Bond_URL"])

    if bond_yield is None:
        status = "PARTIAL_FAIL"
        notes.append(f"Bond: {bond_note}")

    if ust_yield is None:
        status = "PARTIAL_FAIL"
        notes.append(f"UST: {ust_note}")

    fx_data = fetch_fx_data(country_cfg["FX_Ticker"])

    if fx_data.get("status") != "OK":
        status = "PARTIAL_FAIL"
        notes.append(f"FX: {fx_data.get('error', 'Unknown FX error')}")

    spread_bps = None

    if bond_yield is not None and ust_yield is not None:
        spread_bps = round((bond_yield - ust_yield) * 100, 2)

    return {
        "Date": now.date().isoformat(),
        "Fetch_Timestamp": now.isoformat(),
        "Country": country,
        "Bond_Name": country_cfg["Bond_Name"],
        "Bond_URL": country_cfg["Bond_URL"],
        "Bond_Yield": bond_yield,
        "UST_10Y_Yield": ust_yield,
        "Spread_bps": spread_bps,
        "FX_Ticker": country_cfg["FX_Ticker"],
        "FX_URL": country_cfg["FX_URL"],
        "FX_Spot": fx_data.get("spot"),
        "FX_5D_Change_%": fx_data.get("5d"),
        "FX_1M_Change_%": fx_data.get("1m"),
        "FX_YTD_Change_%": fx_data.get("ytd"),
        "Bond_Source": "BondBloX",
        "UST_Source": "US Treasury",
        "FX_Source": "Yahoo Finance via yfinance",
        "Status": status,
        "Notes": "; ".join(notes),
    }


def build_daily_rows() -> list[Dict[str, Any]]:
    ust_yield, ust_note = fetch_ust_10y()

    if ust_yield is None:
        logger.error("UST fetch failed: %s", ust_note)
    else:
        logger.info("Fetched UST 10Y yield: %.2f | %s", ust_yield, ust_note)

    rows = []

    for country_cfg in COUNTRIES:
        country = country_cfg["Country"]

        try:
            logger.info("Fetching data for %s", country)
            row = build_daily_row(country_cfg, ust_yield, ust_note)
            rows.append(row)

            logger.info(
                "%s complete | Bond Yield: %s | UST: %s | Spread: %s | Status: %s",
                country,
                row["Bond_Yield"],
                row["UST_10Y_Yield"],
                row["Spread_bps"],
                row["Status"],
            )

        except Exception as e:
            logger.exception("Unexpected country-level failure for %s", country)

            now = get_now_et()

            rows.append({
                "Date": now.date().isoformat(),
                "Fetch_Timestamp": now.isoformat(),
                "Country": country,
                "Bond_Name": country_cfg["Bond_Name"],
                "Bond_URL": country_cfg["Bond_URL"],
                "Bond_Yield": None,
                "UST_10Y_Yield": ust_yield,
                "Spread_bps": None,
                "FX_Ticker": country_cfg["FX_Ticker"],
                "FX_URL": country_cfg["FX_URL"],
                "FX_Spot": None,
                "FX_5D_Change_%": None,
                "FX_1M_Change_%": None,
                "FX_YTD_Change_%": None,
                "Bond_Source": "BondBloX",
                "UST_Source": "US Treasury",
                "FX_Source": "Yahoo Finance via yfinance",
                "Status": "FAIL",
                "Notes": f"Unexpected country-level failure: {e}",
            })

    return rows


# ============================================================
# CSV APPEND
# ============================================================

def append_to_csv(rows: list[Dict[str, Any]], output_file: str = OUTPUT_FILE) -> None:
    if not rows:
        logger.warning("No rows to append")
        return

    df = pd.DataFrame(rows)

    expected_columns = [
        "Date",
        "Fetch_Timestamp",
        "Country",
        "Bond_Name",
        "Bond_URL",
        "Bond_Yield",
        "UST_10Y_Yield",
        "Spread_bps",
        "FX_Ticker",
        "FX_URL",
        "FX_Spot",
        "FX_5D_Change_%",
        "FX_1M_Change_%",
        "FX_YTD_Change_%",
        "Bond_Source",
        "UST_Source",
        "FX_Source",
        "Status",
        "Notes",
    ]

    df = df[expected_columns]

    file_exists = os.path.exists(output_file)

    df.to_csv(
        output_file,
        mode="a",
        header=not file_exists,
        index=False,
    )

    logger.info("Appended %d rows to %s", len(df), output_file)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    logger.info("Starting daily sovereign risk pipeline")
    rows = build_daily_rows()
    append_to_csv(rows)
    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()