# main.py
import json
import os
import time
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from logger_config import logger

load_dotenv()

app = FastAPI(title="BOM Space Weather Wrapper")

BOM_API_KEY = os.getenv("BOM_API_KEY")
BOM_BASE_URL = os.getenv("BOM_BASE_URL")


# --- Pydantic Models ---
class BaseOptions(BaseModel):
    # 'options' is a flexible dictionary in the BOM API,
    # but usually contains 'location', 'start', 'end'.
    options: dict[str, Any] = Field(default_factory=dict)


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


# --- 2. The 'Aurora Check' Endpoint (Optimized for Cron Job) ---
@app.get("/check-aurora-storm")
async def check_aurora():
    """
    Simplified endpoint for the Cron Job.
    Fetches the K-Index. If K-Index >= 5, it indicates storm conditions suitable for Aurora.
    """
    # Specific BOM endpoint for K-Index
    endpoint = "get-k-index"

    # "Australian region" is the standard location option
    payload = {"options": {"location": "Australian region"}}

    data = await fetch_from_bom(endpoint, payload)

    # Parse logic
    try:
        # BOM returns a list of data points. We want the latest.
        latest_record = data["data"][0]
        k_index = int(latest_record["index"])

        # Aurora Logic: K >= 5 is usually the threshold for TAS/VIC visibility
        alert_active = k_index >= 5

        result = {
            "alert": alert_active,
            "k_index": k_index,
            "timestamp": latest_record["valid_time"],
            "message": "Aurora likely visible!"
            if alert_active
            else "No significant activity.",
        }

        # Log specifically if an alert is found
        if alert_active:
            logger.warning(f"AURORA ALERT TRIGGERED: K-Index is {k_index}")

        return result

    except (KeyError, IndexError, ValueError) as e:
        logger.error(f"Failed to parse BOM data: {e}")
        return {"error": "Failed to parse K-Index data"}


# --- Endpoints ---


@app.post("/get-aurora-alert")
async def get_aurora_notice():
    """
    Requests details of any aurora alert current for the Australian region.
    Aurora alerts are used to report geomagnetic activity currently in progress and favourable for auroras.
    """
    return await fetch_from_bom("get-aurora-alert", {})


@app.post("/get-aurora-watch")
async def get_aurora_watch():
    """
    Requests details of any aurora watch current for the Australian region.
    Aurora watches are used to warn of likely auroral activity in the next 48 hours.
    """
    return await fetch_from_bom("get-aurora-watch", {})


@app.post("/get-aurora-outlook")
async def get_aurora_outlook():
    """
    Requests details of any aurora outlook current for the Australian region.
    Aurora outlooks are used to warn of likely auroral activity 3-7 days hence.
    """
    return await fetch_from_bom("get-aurora-outlook", {})


@app.post("/get-k-index")
async def get_k_index(body: Optional[BaseOptions] = Body(default=None)):
    """
    Get the K-Index (Geomagnetic activity).
    High K-index (>= 5) indicates high aurora probability.
    """
    # Default location to 'Australian region' if not provided
    opts = body.options if body is not None else {}
    if not opts.get("location"):
        opts["location"] = "Australian region"
    return await fetch_from_bom("get-k-index", {"options": opts})


@app.post("/get-a-index")
async def get_a_index(body: Optional[BaseOptions] = Body(default=None)):
    """Get the A-Index (Daily average geomagnetic activity)."""
    opts = body.options if body is not None else {}
    if not opts.get("location"):
        opts["location"] = "Australian region"
    return await fetch_from_bom("get-a-index", {"options": opts})


@app.post("/get-dst-index")
async def get_dst_index(body: Optional[BaseOptions] = Body(default=None)):
    """Get the Disturbance Storm Time (Dst) index."""
    opts = body.options if body is not None else {}
    if not opts.get("location"):
        opts["location"] = "Australian region"
    return await fetch_from_bom("get-dst-index", {"options": opts})


@app.post("/get-mag-alert")
async def get_magnetic_alert():
    """Get current magnetic alerts."""
    return await fetch_from_bom("get-mag-alert", {})


@app.post("/get-mag-warning")
async def get_geophysical_warning():
    """Get current magnetic warnings."""
    return await fetch_from_bom("get-mag-warning", {})


# --- Health Check ---
@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
