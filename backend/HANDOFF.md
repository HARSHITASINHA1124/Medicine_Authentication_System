# ML + Frontend Handoff

This document reflects the current backend contract for the Raspberry Pi + ML + cloud sync flow.

## 1) ML -> Backend Integration

### Endpoint
POST /api/scans/

### Exact request JSON
```json
{
  "scan_id": 987,
  "batch_id": "B12345",
  "medicine_name": "Paracetamol",
  "channel_1": 1.2,
  "channel_2": 2.3,
  "channel_3": 3.4,
  "channel_4": 4.5,
  "channel_5": 5.6,
  "channel_6": 6.7,
  "classification": "Suspicious",
  "confidence_score": 0.61,
  "anomaly_score": 0.47
}
```

### Field definitions
| Field | Type | Nullable | Notes |
|---|---|---:|---|
| `scan_id` | integer | Yes | Application-level scan identifier. The backend keeps an internal auto-increment key and sets `scan_id` as the business identifier when provided. |
| `batch_id` | string | Yes | Batch reference if available |
| `medicine_name` | string | Yes | Medicine name if available |
| `channel_1` | number | Yes | 6-channel input reading |
| `channel_2` | number | Yes | 6-channel input reading |
| `channel_3` | number | Yes | 6-channel input reading |
| `channel_4` | number | Yes | 6-channel input reading |
| `channel_5` | number | Yes | 6-channel input reading |
| `channel_6` | number | Yes | 6-channel input reading |
| `classification` | string | Yes | Allowed values: `Genuine`, `Counterfeit`, `Suspicious` |
| `confidence_score` | number | Yes | Range: 0.0 to 1.0 |
| `anomaly_score` | number | Yes | Range: 0.0 to 1.0 |

### Validation rules
- `classification` must be one of `"Genuine"`, `"Counterfeit"`, or `"Suspicious"` when provided.
- `confidence_score` must be between `0.0` and `1.0` when provided.
- `anomaly_score` must be between `0.0` and `1.0` when provided.
- All channel values are nullable and may be omitted when unavailable.
- Nulls are valid for incomplete or unavailable ML results.

### Raspberry Pi / ML data pipeline
- The Raspberry Pi produces 6 input channels for one reading.
- A single sample may include 10 rows from the sensor pipeline before ML evaluation.
- The backend expects the final ML result payload after aggregation/processing, not the raw 10-row sensor stream.
- The ML service returns the final classification plus confidence/anomaly values and the 6 aggregate channel readings.

### Backend-controlled fields
These are created by the backend and should not be provided by ML clients:
- `id` — internal auto-increment database primary key
- `device_id` — from `DEVICE_ID` in `.env`
- `timestamp` — server-generated UTC timestamp
- `sync_status` — backend-managed local status (`pending` initially)

### Example valid request
```json
{
  "scan_id": 101,
  "batch_id": "B12345",
  "medicine_name": "Paracetamol",
  "channel_1": 0.92,
  "channel_2": 0.88,
  "channel_3": 0.76,
  "channel_4": 0.81,
  "channel_5": 0.90,
  "channel_6": 0.87,
  "classification": "Genuine",
  "confidence_score": 0.96,
  "anomaly_score": 0.04
}
```

### Example nullable request
```json
{
  "scan_id": null,
  "batch_id": null,
  "medicine_name": null,
  "channel_1": null,
  "channel_2": null,
  "channel_3": null,
  "channel_4": null,
  "channel_5": null,
  "channel_6": null,
  "classification": null,
  "confidence_score": null,
  "anomaly_score": null
}
```

---

## 2) Backend -> Frontend Integration

### GET /api/scans/
List scans with pagination.

Query parameters:
- `skip` (int, default `0`)
- `limit` (int, default `50`)

Example response:
```json
[
  {
    "id": 1,
    "scan_id": 1,
    "device_id": "PI_001",
    "batch_id": "B12345",
    "medicine_name": "Paracetamol",
    "channel_1": 0.92,
    "channel_2": 0.88,
    "channel_3": 0.76,
    "channel_4": 0.81,
    "channel_5": 0.90,
    "channel_6": 0.87,
    "timestamp": "2026-08-14T12:00:00Z",
    "classification": "Genuine",
    "confidence_score": 0.96,
    "anomaly_score": 0.04,
    "sync_status": "pending"
  }
]
```

Pagination behavior:
- Response is a JSON array.
- Use `skip` and `limit` to page results.
- `limit` caps the number of returned rows.
- No pagination metadata wrapper is included.

### GET /api/scans/{scan_id}
Example response:
```json
{
  "id": 3,
  "scan_id": 3,
  "device_id": "PI_001",
  "batch_id": "B12345",
  "medicine_name": "Paracetamol",
  "channel_1": 0.92,
  "channel_2": 0.88,
  "channel_3": 0.76,
  "channel_4": 0.81,
  "channel_5": 0.90,
  "channel_6": 0.87,
  "timestamp": "2026-08-14T12:00:00Z",
  "classification": "Genuine",
  "confidence_score": 0.96,
  "anomaly_score": 0.04,
  "sync_status": "pending"
}
```

### GET /api/batches/
Query parameters:
- `skip` (int, default `0`)
- `limit` (int, default `50`)

Example response:
```json
[
  {
    "batch_id": "B12345",
    "medicine_name": "Paracetamol",
    "manufacturer": "Alpha Pharma",
    "batch_number": "ALPHA-001",
    "manufacturing_date": "2025-01-01",
    "expiry_date": "2028-01-01",
    "created_at": "2026-08-14T12:00:00Z"
  }
]
```

### GET /api/batches/{batch_id}
Example response:
```json
{
  "batch_id": "B12345",
  "medicine_name": "Paracetamol",
  "manufacturer": "Alpha Pharma",
  "batch_number": "ALPHA-001",
  "manufacturing_date": "2025-01-01",
  "expiry_date": "2028-01-01",
  "created_at": "2026-08-14T12:00:00Z"
}
```

### GET /api/dashboard/stats
Example response:
```json
{
  "total_scans": 42,
  "genuine": 30,
  "counterfeit": 12,
  "suspicious": 5,
  "pending_sync": 3
}
```

### GET /api/dashboard/trends
Query parameters:
- `days` (int, default `7`)

Example response:
```json
{
  "trends": [
    {"date": "2026-08-08", "count": 12},
    {"date": "2026-08-09", "count": 7},
    {"date": "2026-08-10", "count": 9}
  ]
}
```

### GET /api/dashboard/risk
Example response:
```json
{
  "average_anomaly_score": 0.2143
}
```

### Error responses
- `404 Not Found` — scan or batch does not exist
- `422 Unprocessable Entity` — invalid request body / validation failure
- `400 Bad Request` — duplicate batch ID on creation

Example 404 response:
```json
{
  "detail": "Scan not found"
}
```

Example 422 response:
```json
{
  "detail": [
    {
      "loc": ["body", "confidence_score"],
      "msg": "Input should be less than or equal to 1",
      "type": "less_than_equal"
    }
  ]
}
```

---

## 3) Backend-controlled fields

The following values are backend-controlled and should not be set or modified by the ML team or frontend clients unless explicitly required by the backend contract.

- `id` — internal database primary key used by SQLAlchemy
- `device_id` — device identity, loaded from `DEVICE_ID` in `.env`
- `scan_id` — application-level identifier, usually provided by the ML/PI pipeline or auto-filled when missing
- `timestamp` — UTC timestamp assigned by backend at creation time
- `sync_status` — backend synchronization state (`pending` or `synced`)

Frontend/ML clients should treat these values as read-only unless the backend explicitly exposes an update path.

---

## 4) Database behavior

### SQLite local/offline storage
- SQLite is the local source of truth.
- Scans and batches are saved locally before cloud sync.
- The app works fully offline.
- No cloud database is required for scan creation or local result retrieval.

### Neon PostgreSQL cloud storage
- Neon is used for centralized cloud storage and backup.
- It is not required for local operation.
- The cloud database stores the same core data model for central aggregation and analysis.

### Synchronization flow
1. Scan is created locally in SQLite.
2. Local record is marked `pending`.
3. Backend immediately returns the result to the caller.
4. Background sync worker uploads pending scans to Neon.
5. After successful confirmation, local row is set to `synced`.
6. The SQLite record is not deleted until cloud confirmation succeeds.

### Idempotency
The cloud sync uses the uniqueness of `(device_id, scan_id)`.

This prevents duplicate Neon rows on repeated sync attempts and app restarts.
The internal `id` column is the database primary key, while `scan_id` remains the application identifier used for idempotent cloud writes.

---

## 5) Running the backend locally

1. Open a terminal in the backend folder.
2. Create and activate a virtual environment.
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Copy `.env.example` to `.env` and set the values.
5. Start the server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```
6. Open the docs at:
```text
http://127.0.0.1:8000/docs
```

### Example `.env`
```env
DEVICE_ID=PI_001
DATABASE_URL=sqlite:///./local_scans.db
POSTGRES_URL=postgresql://user:password@host:5432/dbname?sslmode=require
SYNC_INTERVAL_SECONDS=300
```

---

## 6) Example curl / Postman requests

### Create a scan
```bash
curl -X POST "http://127.0.0.1:8000/api/scans/" \
  -H "Content-Type: application/json" \
  -d '{
    "batch_id": "B12345",
    "medicine_name": "Paracetamol",
    "classification": "Genuine",
    "confidence_score": 0.96,
    "anomaly_score": 0.04
  }'
```

### Get scans
```bash
curl "http://127.0.0.1:8000/api/scans/?skip=0&limit=10"
```

### Get one scan
```bash
curl "http://127.0.0.1:8000/api/scans/1"
```

### Create a batch
```bash
curl -X POST "http://127.0.0.1:8000/api/batches/" \
  -H "Content-Type: application/json" \
  -d '{
    "batch_id": "B12345",
    "medicine_name": "Paracetamol",
    "manufacturer": "Alpha Pharma",
    "batch_number": "ALPHA-001",
    "manufacturing_date": "2025-01-01",
    "expiry_date": "2028-01-01"
  }'
```

### Get batches
```bash
curl "http://127.0.0.1:8000/api/batches/"
```

### Dashboard stats
```bash
curl "http://127.0.0.1:8000/api/dashboard/stats"
```

### Dashboard trends
```bash
curl "http://127.0.0.1:8000/api/dashboard/trends?days=7"
```

### Dashboard risk
```bash
curl "http://127.0.0.1:8000/api/dashboard/risk"
```

---

## 7) Assumptions to confirm with ML / Frontend team

- Whether `classification` is always one of `Genuine` / `Counterfeit` or if a third value may be introduced later.
- Whether `batch_id` is always provided by the ML pipeline or sometimes absent.
- Whether the frontend wants to display `sync_status` directly.
- Whether the frontend needs raw scan ID ordering or a separate display-friendly field.
- Whether batch metadata is created separately before or after scan processing.
- Whether additional cloud-only metadata is expected in future backend versions.

## Contact / Handoff summary
- The backend is already offline-first and sync-safe.
- ML should send scan results via `POST /api/scans/` using the contract above.
- Frontend should consume the GET endpoints above and treat backend-created fields as read-only.
- The SQLite → Neon sync uses idempotent cloud inserts and preserves local pending records until confirmation succeeds.
