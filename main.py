import asyncio
import hashlib
import aiofiles
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any

import httpx
import redis.asyncio as redis
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, AliasChoices
from pydantic_settings import BaseSettings

from logger_config import logger

# --- 1. Configuration ---
class Settings(BaseSettings):
    bom_api_key: str
    bom_base_url: str
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    log_file: str = "bom_wrapper.log"
    log_to_file: str = "1"
    hash_storage_file: Path = Path("last_sent_hash.txt")
    REDIS_URL: Optional[str] = None

    class Config:
        env_file = ".env"

settings = Settings()

# --- 2. Data Models ---
class KIndexData(BaseModel):
    value: int
    threshold_met: bool
    timestamp: Optional[str] = None
    error: Optional[str] = None

class AuroraEvent(BaseModel):
    k_aus: Optional[int] = None
    lat_band: Optional[str] = None
    # Map 'comments' to 'description' automatically
    description: Optional[str] = Field(None, validation_alias=AliasChoices("description", "comments")) 
    valid_until: Optional[str] = None
    cause: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    issue_time: Optional[str] = None

class AuroraCheckResponse(BaseModel):
    summary: str
    active: bool
    k_index: KIndexData
    alert: Optional[AuroraEvent] = None
    watch: Optional[AuroraEvent] = None
    outlook: Optional[AuroraEvent] = None

# --- 3. Storage Strategies (Redis vs File) ---
class AsyncStorage(ABC):
    """Abstract Strategy for storing the message hash."""
    @abstractmethod
    async def get(self, key: str) -> Optional[str]: ...
    @abstractmethod
    async def set(self, key: str, value: str): ...

class RedisStorage(AsyncStorage):
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def get(self, key: str) -> Optional[str]:
        val = await self.redis.get(key)
        return val.decode("utf-8") if val else None

    async def set(self, key: str, value: str):
        await self.redis.set(key, value)

class FileStorage(AsyncStorage):
    def __init__(self, file_path: Path):
        self.file_path = file_path

    async def get(self, key: str) -> Optional[str]:
        if not self.file_path.exists():
            return None
        try:
            async with aiofiles.open(self.file_path, mode='r') as f:
                return await f.read()
        except Exception as e:
            logger.error(f"File read error: {e}")
            return None

    async def set(self, key: str, value: str):
        try:
            async with aiofiles.open(self.file_path, mode='w') as f:
                await f.write(value)
        except Exception as e:
            logger.error(f"File write error: {e}")

# --- 4. Services ---
class NotificationService:
    _HASH_KEY = "last_telegram_message_hash"

    def __init__(self, storage: AsyncStorage):
        self.storage = storage

    async def send_telegram_if_changed(self, message: str):
        if not settings.telegram_bot_token or not settings.telegram_chat_id:
            logger.warning("Telegram not configured.")
            return

        current_hash = hashlib.sha256(message.encode("utf-8")).hexdigest()
        last_hash = await self.storage.get(self._HASH_KEY)

        if current_hash == last_hash:
            logger.info("Telegram Alert suppressed: Content unchanged.")
            return

        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
        payload = {"chat_id": settings.telegram_chat_id, "text": message}

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                logger.info("Telegram alert sent.")
                await self.storage.set(self._HASH_KEY, current_hash)
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")

class AuroraService:
    def __init__(self, client: httpx.AsyncClient, notifier: NotificationService):
        self.client = client
        self.notifier = notifier

    async def _fetch(self, endpoint: str, payload: dict = None) -> Dict | None:
        """Generic safe fetcher."""
        final_payload = {**(payload or {}), "api_key": settings.bom_api_key}
        try:
            resp = await self.client.post(endpoint, json=final_payload)
            resp.raise_for_status()
            data = resp.json().get("data")
            return data[0] if data and isinstance(data, list) else None
        except Exception as e:
            logger.error(f"Fetch failed for {endpoint}: {e}")
            return None

    def _map_event(self, raw_data: Optional[Dict], model: Any) -> Any:
        """Generic mapper using Pydantic aliases."""
        return model(**raw_data) if raw_data else None

    async def get_check_status(self, send_alert: bool) -> AuroraCheckResponse:
        # 1. Parallel Fetch
        results = await asyncio.gather(
            self._fetch("get-k-index", {"options": {"location": "Australian region"}}),
            self._fetch("get-aurora-alert"),
            self._fetch("get-aurora-watch"),
            self._fetch("get-aurora-outlook"),
            return_exceptions=True
        )
        
        # Handle exceptions in gather results safely
        clean_results = [r if not isinstance(r, BaseException) else None for r in results]
        raw_k, raw_alert, raw_watch, raw_outlook = clean_results

        # 2. Process Data
        k_val = int(raw_k.get("index", 0)) if raw_k else 0
        k_data = KIndexData(
            value=k_val,
            threshold_met=(k_val >= 5),
            timestamp=raw_k.get("valid_time") if raw_k else None,
            error="Failed to fetch" if raw_k is None else None
        )

        alert = self._map_event(raw_alert, AuroraEvent)
        watch = self._map_event(raw_watch, AuroraEvent)
        outlook = self._map_event(raw_outlook, AuroraEvent)

        # 3. Logic
        is_active = k_data.threshold_met or (alert is not None)
        
        summary = "No significant activity."
        if is_active:
            summary = f"Aurora Likely! (K-Index: {k_data.value})"
        elif watch:
            summary = "Aurora Watch in effect (next 48h)"

        # 4. Notify (Background Task)
        if send_alert and (is_active or watch):
            msg = self._build_alert_message(summary, k_data.value, alert or watch)
            asyncio.create_task(self.notifier.send_telegram_if_changed(msg))

        return AuroraCheckResponse(
            summary=summary, active=is_active, k_index=k_data,
            alert=alert, watch=watch, outlook=outlook
        )

    def _build_alert_message(self, summary: str, k_val: int, event: AuroraEvent) -> str:
        lines = ["AURORA ALERT"]
        if event:
            if event.issue_time: lines.append(f"Issued: {event.issue_time} UTC")
            if event.valid_until: lines.append(f"Valid until: {event.valid_until} UTC")
        
        if k_val >= 7:
            lines.append(f"\n⚠️ HIGH ACTIVITY (K-{k_val}) ⚠️")
        
        lines.append(summary)
        if event and event.description: lines.append(event.description)
        lines.extend(["https://www.sws.bom.gov.au/Aurora", "https://aurora.dotdoing.com"])
        
        return "\n\n".join(lines)

# --- 5. Application Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: Initializing resources...")
    
    # 1. HTTP Client
    client = httpx.AsyncClient(
        base_url=settings.bom_base_url, 
        timeout=10.0, 
        headers={"Content-Type": "application/json"}
    )
    
    # 2. Storage Backend Selection
    redis_client = None
    storage: AsyncStorage
    
    if settings.REDIS_URL:
        try:
            redis_client = redis.from_url(settings.REDIS_URL)
            await redis_client.ping()
            storage = RedisStorage(redis_client)
            logger.info("Storage: Redis connected.")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}. Falling back to File.")
            storage = FileStorage(settings.hash_storage_file)
    else:
        logger.info("Storage: using local file system.")
        storage = FileStorage(settings.hash_storage_file)

    # 3. Dependency Injection Container
    notifier = NotificationService(storage)
    app.state.aurora_service = AuroraService(client, notifier)

    yield

    logger.info("Shutdown: cleaning up...")
    await client.aclose()
    if redis_client:
        await redis_client.close()

app = FastAPI(title="BOM Space Weather Wrapper", lifespan=lifespan)

# --- 6. Endpoints ---
@app.middleware("http")
async def performance_logger(request: Request, call_next):
    start = asyncio.get_event_loop().time()
    response = await call_next(request)
    duration = asyncio.get_event_loop().time() - start
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.4f}s")
    return response

async def get_service(request: Request) -> AuroraService:
    """Dependency Provider"""
    if not hasattr(request.app.state, "aurora_service"):
         raise HTTPException(status_code=503, detail="Service not initialized")
    return request.app.state.aurora_service

@app.get("/check-aurora-storm", response_model=AuroraCheckResponse)
async def check_aurora(
    send_telegram_alert: bool = False,
    service: AuroraService = Depends(get_service),
):
    return await service.get_check_status(send_alert=send_telegram_alert)

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200)
