import asyncio
import os
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from logger_config import logger


# --- 1. Configuration (Type-Safe) ---
class Settings(BaseSettings):
    bom_api_key: str
    bom_base_url: str
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    log_file: str = "bom_wrapper.log"

    class Config:
        env_file = ".env"


settings = Settings()


# --- 2. Pydantic Models (Data Layer) ---
class KIndexData(BaseModel):
    value: int
    threshold_met: bool
    timestamp: Optional[str] = None
    analysis_time: Optional[str] = None
    error: Optional[str] = None


class AuroraEvent(BaseModel):
    k_aus: Optional[int] = None
    lat_band: Optional[str] = None
    description: Optional[str] = None
    valid_until: Optional[str] = None
    cause: Optional[str] = None
    start_date: Optional[str] = None
    start_time: Optional[str] = None
    end_date: Optional[str] = None
    issue_time: Optional[str] = None


class AuroraCheckResponse(BaseModel):
    summary: str
    active: bool
    k_index: KIndexData
    alert: Optional[AuroraEvent] = None
    watch: Optional[AuroraEvent] = None
    outlook: Optional[AuroraEvent] = None


# --- 3. Lifespan & State Management ---
# This dictionary will hold our shared resources (like the HTTP client)
resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create a single client with a timeout
    logger.info("Starting up: Initializing HTTP Client...")
    resources["client"] = httpx.AsyncClient(
        base_url=settings.bom_base_url,
        timeout=10.0,
        headers={"Content-Type": "application/json"},  # Default headers
    )
    yield
    # Shutdown: Clean up resources
    logger.info("Shutting down: Closing HTTP Client...")
    await resources["client"].aclose()


app = FastAPI(title="BOM Space Weather Wrapper", lifespan=lifespan)

# --- 4. Services (Business Logic Layer) ---


class NotificationService:
    """
    Handles logic for sending alerts externally.
    Namespace class and holder of state as every method is a classmethod.
    """

    last_sent_path = os.path.join(
        os.path.dirname(__file__), "logs", "last_sent_content.txt"
    )

    @classmethod
    def _get_last_sent_content(cls) -> str:
        try:
            with open(cls.last_sent_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    @classmethod
    def _set_last_sent_content(cls, content: str):
        with open(cls.last_sent_path, "w") as f:
            f.write(content)

    @classmethod
    async def send_telegram_if_changed(cls, message: str):
        if not settings.telegram_bot_token:
            return

        last_content = cls._get_last_sent_content()
        if message == last_content:
            logger.info("Telegram alert not sent: content unchanged.")
            return

        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
        payload = {"chat_id": settings.telegram_chat_id, "text": message}
        
        try:
            logger.info("Sending Telegram alert...")
            async with httpx.AsyncClient() as tg_client:
                await tg_client.post(url, json=payload)
                logger.info("Telegram alert sent successfully.")
                cls._set_last_sent_content(message)
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")


class AuroraService:
    """Encapsulates BOM interaction logic."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def _fetch(self, endpoint: str, payload: dict) -> dict | Exception:
        """Internal helper to fetch data safely."""
        # Inject API Key automatically
        final_payload = {**payload, "api_key": settings.bom_api_key}

        try:
            logger.info(f"FETCH: {self.client.base_url}{endpoint}")
            resp = await self.client.post(endpoint, json=final_payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Error fetching {endpoint}: {str(e)}")
            return e

    def _extract_data(
        self, response: dict | BaseException, field_map: dict
    ) -> Optional[dict]:
        """Maps BOM response to our clean dict structure."""
        if isinstance(response, BaseException) or not response.get("data"):
            return None

        item = response["data"][0]
        return {target: item.get(source) for target, source in field_map.items()}

    async def get_check_status(self, send_telegram_alert: bool) -> AuroraCheckResponse:
        # 1. Parallel Fetching
        results = await asyncio.gather(
            self._fetch("get-k-index", {"options": {"location": "Australian region"}}),
            self._fetch("get-aurora-alert", {}),
            self._fetch("get-aurora-watch", {}),
            self._fetch("get-aurora-outlook", {}),
            return_exceptions=True,
        )

        # Unpack results (gather ensures order matches input)
        raw_k, raw_alert, raw_watch, raw_outlook = results

        # 2. Process K-Index
        k_data = {"value": 0, "threshold_met": False, "error": None}
        if not isinstance(raw_k, BaseException) and raw_k.get("data"):
            latest = raw_k["data"][0]
            val = int(latest.get("index", 0))
            k_data = {
                "value": val,
                "threshold_met": val >= 5,
                "timestamp": latest.get("valid_time"),
                "analysis_time": latest.get("analysis_time"),
            }
        else:
            k_data["error"] = "Failed to fetch K-Index"

        # 3. Process Events
        alert = self._extract_data(
            raw_alert,
            {
                "k_aus": "k_aus",
                "description": "description",
                "valid_until": "valid_until",
                "lat_band": "lat_band",
                "start_time": "start_time",
            },
        )
        watch = self._extract_data(
            raw_watch,
            {
                "k_aus": "k_aus",
                "cause": "cause",
                "description": "comments",
                "start_date": "start_date",
                "end_date": "end_date",
                "issue_time": "issue_time",
                "lat_band": "lat_band",
            },
        )
        outlook = self._extract_data(
            raw_outlook,
            {
                "k_aus": "k_aus",
                "cause": "cause",
                "description": "comments",
                "start_date": "start_date",
                "end_data": "end_date",
                "issue_time": "issue_time",
                "lat_band": "lat_band",
            },
        )

        # 4. Logic & Alerting
        is_active = k_data["threshold_met"] or (alert is not None)

        summary = "No significant activity."
        if is_active:
            summary = f"Aurora Likely! (K-Index: {k_data['value']})"
        elif watch:
            summary = "Aurora Watch in effect (next 48h)"

        # Construct Response Object
        response_model = AuroraCheckResponse(
            summary=summary,
            active=is_active,
            k_index=KIndexData(**k_data),
            alert=AuroraEvent(**alert) if alert else None,
            watch=AuroraEvent(**watch) if watch else None,
            outlook=AuroraEvent(**outlook) if outlook else None,
        )

        # 5. Handle Notifications (Fire and Forget task)
        if send_telegram_alert and (is_active or watch):
            source = alert or watch or {}
            api_comments = source.get("description", "")
            issue_time = source.get("issue_time", "")
            valid_until = source.get("valid_until", "")

            msg_parts = ["AURORA ALERT"]

            if issue_time:
                msg_parts.append(f"Issued at: {issue_time} UTC")

            if valid_until:
                msg_parts.append(f"Valid until: {valid_until} UTC")

            k_value = int(k_data["value"])
            if k_value >= 7:
                msg_parts.append(f"\n⚠️ HIGH ACTIVITY DETECTED! ⚠️\nk-index: {k_value}")

            msg_parts.extend(
                [summary, api_comments, "https://www.sws.bom.gov.au/Aurora", "https://aurora.dotdoing.com"]
            )

            msg = "\n\n".join(filter(None, msg_parts))
            asyncio.create_task(NotificationService.send_telegram_if_changed(msg))

        return response_model


# --- 5. Dependencies ---
def get_aurora_service() -> AuroraService:
    """Dependency injection for the service."""
    if "client" not in resources:
        raise HTTPException(status_code=503, detail="Client not initialized")
    return AuroraService(resources["client"])


# --- 6. Endpoints ---


@app.middleware("http")
async def performance_logger(request: Request, call_next):
    start = asyncio.get_event_loop().time()
    response = await call_next(request)
    duration = asyncio.get_event_loop().time() - start
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {duration:.4f}s"
    )
    return response


@app.get("/check-aurora-storm", response_model=AuroraCheckResponse)
async def check_aurora(
    send_telegram_alert: bool = False,
    service: AuroraService = Depends(get_aurora_service),
):
    """
    Optimized endpoint that delegates logic to AuroraService.
    Checks for aurora storm conditions and returns structured data.
    1. Fetches K-Index, Aurora Alerts, Watches, and Outlooks in parallel.
    2. Processes and determines if aurora conditions are met.
    3. Sends Telegram alerts if conditions are met and configured.
    4. Returns a comprehensive response with all relevant data.
    """
    return await service.get_check_status(send_telegram_alert=send_telegram_alert)


@app.get("/health")
def health_check():
    return {"status": "ok", "client_ready": "client" in resources}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8200)
