# main.py
import asyncio
import json
import os
import time
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from logger_config import logger

load_dotenv()

app = FastAPI(title="BOM Space Weather Wrapper")

BOM_API_KEY = os.getenv("BOM_API_KEY")
BOM_BASE_URL = os.getenv("BOM_BASE_URL")
SEND_TELEGRAM_ALERTS = os.getenv("SEND_TELEGRAM_ALERTS", "false").lower() == "true"
TIMEZONE = os.getenv("TIMEZONE", "Australia/Sydney")


# --- Pydantic Models ---
class BaseOptions(BaseModel):
    # 'options' is a flexible dictionary in the BOM API,
    # but usually contains 'location', 'start', 'end'.
    options: dict[str, Any] = Field(default_factory=dict)


class KIndexData(BaseModel):
    value: int
    threshold_met: bool
    timestamp: Optional[str] = None
    # 'error' is optional because it only appears if the fetch fails
    error: Optional[str] = None


class AuroraEvent(BaseModel):
    k_aus: Optional[int] = None
    lat_band: Optional[str] = None
    description: Optional[str] = None
    valid_until: Optional[str] = None
    cause: Optional[str] = None
    comments: Optional[str] = None
    start_date: Optional[str] = None


class AuroraCheckResponse(BaseModel):
    summary: str
    active: bool
    k_index: KIndexData
    # These fields are optional because they are None when no event is active
    alert: Optional[AuroraEvent] = None
    watch: Optional[AuroraEvent] = None
    outlook: Optional[AuroraEvent] = None


# --- 1. Middleware to Log Incoming Requests to YOUR Wrapper ---
@app.middleware("http")
async def log_middleware(request: Request, call_next):
    start_time = time.time()

    # Process the request
    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        f"INCOMING: {request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {process_time:.4f}s"
    )
    return response


# --- Helper: Async Client with Logging for OUTGOING requests ---
async def fetch_from_bom(endpoint: str, payload: dict[str, Any]):
    """
    Sends a request to BOM.
    NOTE: BOM SWS API usually requires POST for data retrieval with the API key in the body.
    """
    url = f"{BOM_BASE_URL}/{endpoint}"

    # Ensure API Key is present in payload
    if "api_key" not in payload:
        payload["api_key"] = BOM_API_KEY

    async with httpx.AsyncClient() as client:
        logger.info(f"OUTGOING REQUEST: POST {url} | Payload: {json.dumps(payload)}")

        try:
            resp = await client.post(url, json=payload, timeout=10.0)

            # Log the full response text for debugging/history
            logger.info(
                f"INCOMING RESPONSE (BOM): {resp.status_code} | Body: {resp.text}"
            )  # Truncated log

            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"BOM API Error: {e.response.text}")
            raise HTTPException(
                status_code=e.response.status_code, detail="BOM API Error"
            )
        except Exception as e:
            logger.error(f"Connection Error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal Server Error")


def extract_bom_data(
    response_data: dict | BaseException, field_map: dict[str, str]
) -> Optional[dict]:
    """
    Extracts and maps fields from the first item of a BOM response list.

    Args:
        response_data: The JSON response from BOM (or an Exception object if the request failed).
        field_map: A dictionary mapping { "our_output_name": "bom_api_field_name" }

    Returns:
        A dictionary of mapped data if an event is active, otherwise None.
    """
    # 1. Handle API failures gracefully (if asyncio.gather returned an exception)
    if isinstance(response_data, Exception):
        logger.error(f"Skipping data extraction due to API error: {response_data}")
        return None

    # 2. Check structure
    data_list = response_data.get("data", [])
    if not data_list:
        return None

    # 3. Extract and Map
    # BOM returns arrays; we usually care about the most recent/current one (index 0)
    item = data_list[0]

    return {
        target_key: item.get(source_key) for target_key, source_key in field_map.items()
    }


# --- 2. The 'Aurora Check' Endpoint (Optimized for Cron Job) ---
@app.get("/check-aurora-storm", response_model=AuroraCheckResponse)
async def check_aurora():
    """
    Aggregates K-Index, Alerts, Watches, and Outlooks into a single status report.
    Executes 4 BOM API calls in parallel.
    If configured, sends Telegram alerts for significant auroral activity.
    1. K-Index >= 5 indicates high aurora probability.
    2. Active Alerts indicate current geomagnetic activity.
    3. Watches indicate likely activity in next 48 hours.
    4. Outlooks indicate likely activity in 3-7 days.
    5. Summary message indicates overall auroral activity status.
    6. Logs significant events for monitoring.
    7. Designed for periodic execution (e.g., via cron) to monitor auroral
    """
    # 1. Define the 4 independent tasks
    task_k_index = fetch_from_bom(
        "get-k-index", {"options": {"location": "Australian region"}}
    )
    task_alert = fetch_from_bom("get-aurora-alert", {})
    task_watch = fetch_from_bom("get-aurora-watch", {})
    task_outlook = fetch_from_bom("get-aurora-outlook", {})

    # 2. Run them in parallel (Concurrency)
    # return_exceptions=True prevents one failed API call from crashing the whole request
    results = await asyncio.gather(
        task_k_index, task_alert, task_watch, task_outlook, return_exceptions=True
    )

    res_k, res_alert, res_watch, res_outlook = results

    # 3. Process K-Index (Current Conditions)
    k_data = {}
    is_storm_k = False

    if not isinstance(res_k, BaseException) and res_k.get("data"):
        latest = res_k["data"][0]
        k_val = int(latest.get("index", 0))
        is_storm_k = k_val >= 5
        k_data = {
            "value": k_val,
            "threshold_met": is_storm_k,
            "timestamp": latest.get("valid_time"),
        }
    else:
        k_data = {"error": "Could not fetch K-Index"}

    # 4. Process Events (Alert, Watch, Outlook) using DRY Helper
    # We define what fields we want to keep and rename them to be consistent

    current_alert = extract_bom_data(
        res_alert,
        {
            "k_aus": "k_aus",
            "description": "description",
            "valid_until": "valid_until",
            "lat_band": "lat_band",
        },
    )

    upcoming_watch = extract_bom_data(
        res_watch,
        {
            "k_aus": "k_aus",
            "cause": "cause",
            "comments": "comments",
            "start_date": "start_date",
        },
    )

    future_outlook = extract_bom_data(
        res_outlook,
        {
            "k_aus": "k_aus",  # Note: field might be missing in outlooks sometimes
            "cause": "cause",
            "comments": "comments",
            "start_date": "start_date",
        },
    )

    # 5. Determine Global Status
    # We flag "Auroral Activity" if K-index is high OR there is an active Alert.
    aurora_active = is_storm_k or (current_alert is not None)

    summary_msg = "No significant activity."

    if aurora_active:
        summary_msg = f"Aurora Likely! (K-Index: {k_data.get('value')})."
    elif current_alert:
        summary_msg = "Aurora Alert in effect!"
    elif upcoming_watch:
        summary_msg = "Aurora Watch in effect (next 48h)."

    if SEND_TELEGRAM_ALERTS:
        telegram_msg = None
        if aurora_active:
            telegram_msg = f"Aurora Likely! (K-Index: {k_data.get('value')})"
        if current_alert:
            telegram_msg = f"""{telegram_msg}
            Aurora Alert in effect!
            {current_alert.get("description")}"""
        elif upcoming_watch:
            telegram_msg = f"""Aurora Watch in effect (next 48h).
            Expected level of activity: {upcoming_watch.get("k_aus")}

            {upcoming_watch.get("comments")}"""

        if telegram_msg:
            logger.info(f"TELEGRAM MESSAGE:\n{telegram_msg}")

    if aurora_active:
        logger.warning(f"AURORA ACTIVE: {json.dumps(k_data)}")

    return {
        "summary": summary_msg,
        "active": aurora_active,
        "k_index": k_data,
        "alert": current_alert,  # None if no alert
        "watch": upcoming_watch,  # None if no watch
        "outlook": future_outlook,  # None if no outlook
    }


# --- Proxy Endpoints Endpoints Disabled ---


# @app.post("/get-aurora-alert")
# async def get_aurora_notice():
#     """
#     Requests details of any aurora alert current for the Australian region.
#     Aurora alerts are used to report geomagnetic activity currently in progress and favourable for auroras.
#     """
#     return await fetch_from_bom("get-aurora-alert", {})


# @app.post("/get-aurora-watch")
# async def get_aurora_watch():
#     """
#     Requests details of any aurora watch current for the Australian region.
#     Aurora watches are used to warn of likely auroral activity in the next 48 hours.
#     """
#     return await fetch_from_bom("get-aurora-watch", {})


# @app.post("/get-aurora-outlook")
# async def get_aurora_outlook():
#     """
#     Requests details of any aurora outlook current for the Australian region.
#     Aurora outlooks are used to warn of likely auroral activity 3-7 days hence.
#     """
#     return await fetch_from_bom("get-aurora-outlook", {})


# @app.post("/get-k-index")
# async def get_k_index(body: Optional[BaseOptions] = Body(default=None)):
#     """
#     Get the K-Index (Geomagnetic activity).
#     High K-index (>= 5) indicates high aurora probability.
#     """
#     # Default location to 'Australian region' if not provided
#     opts = body.options if body is not None else {}
#     if not opts.get("location"):
#         opts["location"] = "Australian region"
#     return await fetch_from_bom("get-k-index", {"options": opts})


# @app.post("/get-a-index")
# async def get_a_index(body: Optional[BaseOptions] = Body(default=None)):
#     """Get the A-Index (Daily average geomagnetic activity)."""
#     opts = body.options if body is not None else {}
#     if not opts.get("location"):
#         opts["location"] = "Australian region"
#     return await fetch_from_bom("get-a-index", {"options": opts})


# @app.post("/get-dst-index")
# async def get_dst_index(body: Optional[BaseOptions] = Body(default=None)):
#     """Get the Disturbance Storm Time (Dst) index."""
#     opts = body.options if body is not None else {}
#     if not opts.get("location"):
#         opts["location"] = "Australian region"
#     return await fetch_from_bom("get-dst-index", {"options": opts})


# @app.post("/get-mag-alert")
# async def get_magnetic_alert():
#     """Get current magnetic alerts."""
#     return await fetch_from_bom("get-mag-alert", {})


# @app.post("/get-mag-warning")
# async def get_geophysical_warning():
#     """Get current magnetic warnings."""
#     return await fetch_from_bom("get-mag-warning", {})


# --- Health Check ---
@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
