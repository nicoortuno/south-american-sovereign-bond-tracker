#!/usr/bin/env bash

# Collect the latest BondTerminal analytics observation for five
# South American USD sovereign bonds and append/update a CSV.
#
# API requests are made exclusively with curl.
# Python is used only to parse the downloaded local JSON and write CSV.
#
# Exactly five API calls are made per run:
#   GET /api/v1/bonds/{ISIN}/analytics
#
# No history calls and no automatic retries.

set -u -o pipefail

API_BASE="https://bondterminal.com/api/v1"
OUTPUT_CSV="${OUTPUT_CSV:-data/bondterminal_daily.csv}"

BONDS=(
  "Brazil|brazil|US105756CL22"
  "Colombia|colombia|US195325EG61"
  "Chile|chile|US168863DV76"
  "Peru|peru|US715638EB48"
  "Argentina|argentina|US040114HT09"
)

if [[ -z "${BONDTERMINAL_API_KEY:-}" ]]; then
  echo "ERROR: BONDTERMINAL_API_KEY is not loaded." >&2
  exit 1
fi

RUN_TIMESTAMP_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

TMP_DIR="$(mktemp -d)"
ROWS_FILE="${TMP_DIR}/new_rows.jsonl"

trap 'rm -rf "$TMP_DIR"' EXIT

: > "$ROWS_FILE"

successful_bonds=0
incomplete_run=0

echo "Collecting BondTerminal data at ${RUN_TIMESTAMP_UTC}"
echo

for entry in "${BONDS[@]}"; do
  IFS="|" read -r country slug isin <<< "$entry"

  response_file="${TMP_DIR}/${slug}.json"
  request_url="${API_BASE}/bonds/${isin}/analytics"

  echo "Requesting ${country}: ${isin}"

  # This is the API call. There is no retry option, so it runs once.
  http_code="$(
    curl \
      --fail-with-body \
      --silent \
      --show-error \
      --max-time 60 \
      --output "$response_file" \
      --write-out "%{http_code}" \
      "$request_url" \
      -H "Authorization: Bearer ${BONDTERMINAL_API_KEY}" \
      -H "Accept: application/json"
  )"

  curl_exit_code=$?

  if [[ "$curl_exit_code" -ne 0 || "$http_code" != "200" ]]; then
    echo "  ERROR: curl exit ${curl_exit_code}; HTTP ${http_code}" >&2

    if [[ -s "$response_file" ]]; then
      echo "  Response:" >&2
      cat "$response_file" >&2
      echo >&2
    fi

    incomplete_run=1
    continue
  fi

  # Parse the local JSON file. This does not make any API request.
  python3 - \
    "$response_file" \
    "$country" \
    "$isin" \
    "$RUN_TIMESTAMP_UTC" \
    >> "$ROWS_FILE" <<'PY'
import json
import sys
from pathlib import Path


response_path = Path(sys.argv[1])
country = sys.argv[2]
expected_isin = sys.argv[3]
collection_timestamp_utc = sys.argv[4]

try:
    with response_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
except (OSError, json.JSONDecodeError) as error:
    print(f"Invalid JSON for {country}: {error}", file=sys.stderr)
    raise SystemExit(2)

if not isinstance(payload, dict):
    print(
        f"Unexpected API response type for {country}.",
        file=sys.stderr,
    )
    raise SystemExit(2)

if payload.get("error"):
    print(
        f"API error for {country}: "
        f"{payload.get('error')} — {payload.get('message', '')}",
        file=sys.stderr,
    )
    raise SystemExit(2)

returned_isin = payload.get("isin")

if returned_isin != expected_isin:
    print(
        f"ISIN mismatch for {country}: expected {expected_isin}, "
        f"received {returned_isin}",
        file=sys.stderr,
    )
    raise SystemExit(2)

market = payload.get("market") or {}
yields = payload.get("yields") or {}
risk = payload.get("risk") or {}
spreads = payload.get("spreads") or {}
pricing = payload.get("pricing") or {}
schedule = payload.get("schedule") or {}

market_timestamp = market.get("timestamp")

observation_date = (
    str(market_timestamp)[:10]
    if market_timestamp is not None
    else None
)

row = {
    "observation_date": observation_date,
    "market_timestamp": market_timestamp,
    "collection_timestamp_utc": collection_timestamp_utc,
    "country": country,
    "isin": returned_isin,
    "ticker": payload.get("ticker"),
    "settlement_date": payload.get("settlement"),
    "clean_price": pricing.get("cleanPrice"),
    "ytm_pct": yields.get("ytm"),
    "ytw_pct": yields.get("ytw"),
    "current_yield_pct": yields.get("currentYield"),
    "g_spread_bps": spreads.get("gSpread"),
    "z_spread_bps": spreads.get("zSpread"),
    "treasury_yield_pct": spreads.get("treasuryYield"),
    "treasury_tenor": spreads.get("treasuryTenor"),
    "modified_duration": risk.get("modifiedDuration"),
    "effective_duration": risk.get("effectiveDuration"),
    "convexity": risk.get("convexity"),
    "dv01": risk.get("dv01"),
    "average_life_years": schedule.get("averageLife"),
    "source": "BondTerminal API",
    "collection_status": None,
}

core_fields = [
    "observation_date",
    "market_timestamp",
    "isin",
    "ticker",
    "clean_price",
    "ytm_pct",
]

missing_core = [
    field
    for field in core_fields
    if row.get(field) is None
]

if missing_core:
    print(
        f"Cannot save {country}; missing core fields: "
        f"{', '.join(missing_core)}",
        file=sys.stderr,
    )
    raise SystemExit(2)

tracked_fields = [
    "settlement_date",
    "clean_price",
    "ytm_pct",
    "ytw_pct",
    "current_yield_pct",
    "g_spread_bps",
    "z_spread_bps",
    "treasury_yield_pct",
    "treasury_tenor",
    "modified_duration",
    "effective_duration",
    "convexity",
    "dv01",
    "average_life_years",
]

missing_tracked = [
    field
    for field in tracked_fields
    if row.get(field) is None
]

if missing_tracked:
    row["collection_status"] = "PARTIAL"

    print(
        f"{country} returned usable data but is missing: "
        f"{', '.join(missing_tracked)}",
        file=sys.stderr,
    )
else:
    row["collection_status"] = "OK"

print(json.dumps(row, ensure_ascii=False))

# Exit 3 means the row was usable and printed, but incomplete.
raise SystemExit(3 if missing_tracked else 0)
PY

  parse_exit_code=$?

  if [[ "$parse_exit_code" -eq 0 ]]; then
    echo "  Saved: complete response"
    successful_bonds=$((successful_bonds + 1))

  elif [[ "$parse_exit_code" -eq 3 ]]; then
    echo "  Saved: partial response"
    successful_bonds=$((successful_bonds + 1))
    incomplete_run=1

  else
    echo "  ERROR: response could not be converted to a row" >&2
    incomplete_run=1
  fi
done

echo

if [[ ! -s "$ROWS_FILE" ]]; then
  echo "ERROR: No usable bond observations were returned." >&2
  exit 1
fi

# Merge the new local rows into the CSV.
# This makes no API calls.
python3 - "$ROWS_FILE" "$OUTPUT_CSV" <<'PY'
import csv
import json
import os
import sys
from pathlib import Path


rows_path = Path(sys.argv[1])
csv_path = Path(sys.argv[2])

fieldnames = [
    "observation_date",
    "market_timestamp",
    "collection_timestamp_utc",
    "country",
    "isin",
    "ticker",
    "settlement_date",
    "clean_price",
    "ytm_pct",
    "ytw_pct",
    "current_yield_pct",
    "g_spread_bps",
    "z_spread_bps",
    "treasury_yield_pct",
    "treasury_tenor",
    "modified_duration",
    "effective_duration",
    "convexity",
    "dv01",
    "average_life_years",
    "source",
    "collection_status",
]


def normalize_row(row):
    return {
        field: "" if row.get(field) is None else str(row.get(field))
        for field in fieldnames
    }


rows_by_key = {}

# Preserve existing observations.
if csv_path.exists():
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != fieldnames:
            raise SystemExit(
                "Existing CSV has an unexpected column structure. "
                "Back it up or correct its headers before continuing."
            )

        for row in reader:
            key = (
                row.get("isin", ""),
                row.get("observation_date", ""),
            )

            if all(key):
                rows_by_key[key] = normalize_row(row)

# Add or update this run's observations.
with rows_path.open("r", encoding="utf-8") as file:
    for line in file:
        if not line.strip():
            continue

        row = normalize_row(json.loads(line))

        key = (
            row["isin"],
            row["observation_date"],
        )

        existing = rows_by_key.get(key)

        # Do not replace an existing complete row with a partial row.
        if (
            existing
            and existing.get("collection_status") == "OK"
            and row.get("collection_status") != "OK"
        ):
            continue

        rows_by_key[key] = row

sorted_rows = sorted(
    rows_by_key.values(),
    key=lambda row: (
        row["observation_date"],
        row["country"],
    ),
)

csv_path.parent.mkdir(parents=True, exist_ok=True)

temporary_path = csv_path.with_suffix(
    csv_path.suffix + ".tmp"
)

with temporary_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )
    writer.writeheader()
    writer.writerows(sorted_rows)

os.replace(temporary_path, csv_path)

print(
    f"CSV now contains {len(sorted_rows)} total observation rows."
)
print(f"Saved to: {csv_path}")
PY

merge_exit_code=$?

if [[ "$merge_exit_code" -ne 0 ]]; then
  echo "ERROR: CSV update failed." >&2
  exit 1
fi

echo
echo "Usable bonds saved this run: ${successful_bonds}/5"

if [[ "$incomplete_run" -ne 0 || "$successful_bonds" -ne 5 ]]; then
  echo "Collection was incomplete. Available rows were still saved." >&2
  exit 1
fi

echo "All five bonds were collected successfully."
exit 0