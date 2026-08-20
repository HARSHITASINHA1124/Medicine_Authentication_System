import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text
from app.database.session import engine
from app.database import models
from app.api import endpoints_scans, endpoints_batches, endpoints_dashboard
from app.core.sync_worker import persistent_sync_loop, sync_event

# Create SQLite tables on startup and upgrade older scan databases in place.
models.Base.metadata.create_all(bind=engine)

with engine.begin() as connection:
    existing_columns = {column["name"] for column in inspect(engine).get_columns("scans")}
    if "id" not in existing_columns:
        connection.exec_driver_sql("ALTER TABLE scans RENAME TO scans_legacy")
        connection.exec_driver_sql("""
            CREATE TABLE scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id INTEGER UNIQUE,
                device_id TEXT NOT NULL,
                batch_id TEXT,
                medicine_name TEXT,
                timestamp DATETIME NOT NULL,
                channel_1 FLOAT,
                channel_2 FLOAT,
                channel_3 FLOAT,
                channel_4 FLOAT,
                channel_5 FLOAT,
                channel_6 FLOAT,
                classification TEXT,
                confidence_score FLOAT,
                anomaly_score FLOAT,
                classification_status TEXT,
                anomaly_status TEXT,
                final_status TEXT,
                number_of_readings INTEGER,
                stability_cv FLOAT,
                ml_result JSON,
                sync_status TEXT NOT NULL
            )
        """)
        connection.exec_driver_sql("""
            INSERT INTO scans (
                scan_id, device_id, batch_id, medicine_name, timestamp,
                classification, confidence_score, anomaly_score, classification_status,
                anomaly_status, final_status, number_of_readings, stability_cv, ml_result,
                sync_status
            )
            SELECT scan_id, device_id, batch_id, medicine_name, timestamp,
                classification, confidence_score, anomaly_score, classification_status,
                anomaly_status, final_status, number_of_readings, stability_cv, ml_result,
                sync_status
            FROM scans_legacy
        """)
        connection.exec_driver_sql("DROP TABLE scans_legacy")
    else:
        scan_columns = {
            "channel_1": "FLOAT", "channel_2": "FLOAT", "channel_3": "FLOAT",
            "channel_4": "FLOAT", "channel_5": "FLOAT", "channel_6": "FLOAT",
            "classification_status": "TEXT", "anomaly_status": "TEXT",
            "final_status": "TEXT", "number_of_readings": "INTEGER",
            "stability_cv": "FLOAT", "ml_result": "JSON",
        }
        for column_name, column_type in scan_columns.items():
            if column_name not in existing_columns:
                connection.execute(text(f"ALTER TABLE scans ADD COLUMN {column_name} {column_type}"))

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "http://localhost:5177",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "http://127.0.0.1:5176",
        "http://127.0.0.1:5177",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
