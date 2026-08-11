import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.database.session import engine
from app.database import models
from app.api import endpoints_scans, endpoints_batches, endpoints_dashboard
from app.core.sync_worker import persistent_sync_loop, sync_event

# Create SQLite tables on startup
models.Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the background sync task
    sync_task = asyncio.create_task(persistent_sync_loop())
    
    # Wait for the app to finish
    yield
    
    # Shutdown background task
    sync_task.cancel()
    try:
        await sync_task
    except asyncio.CancelledError:
        pass

app = FastAPI(title="Smart Anti-Counterfeit Medicine API", lifespan=lifespan)

# Monkey-patch trigger_sync in endpoints_scans to set our asyncio Event
def trigger_sync():
    sync_event.set()
endpoints_scans.trigger_sync = trigger_sync

# Include API Routers
app.include_router(endpoints_scans.router, prefix="/api/scans", tags=["Scans"])
app.include_router(endpoints_batches.router, prefix="/api/batches", tags=["Batches"])
app.include_router(endpoints_dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])

@app.get("/health")
def health_check():
    return {"status": "ok"}
