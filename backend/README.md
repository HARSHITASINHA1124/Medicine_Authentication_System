# Backend Module: Smart Anti-Counterfeit Medicine Authentication System

This is the FastAPI backend module for the capstone project. It runs locally on a Raspberry Pi, uses SQLite as the primary source of truth, and periodically synchronizes scans to a cloud PostgreSQL database.

## Architecture

- **FastAPI**: Provides REST APIs.
- **SQLite**: Stores local batch and scan data.
- **SQLAlchemy**: ORM for database queries.
- **Sync Worker**: A persistent asynchronous loop that checks for pending local scans and pushes them to PostgreSQL when online.

## Installation (Raspberry Pi or Local PC)

1. **Clone the repository / copy files.**
2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Or .\venv\Scripts\activate on Windows
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure environment:**
   Create a `.env` file in the root directory (copy `.env.example`):
   ```env
   DEVICE_ID=PI_001
   DATABASE_URL=sqlite:///./local_scans.db
   POSTGRES_URL=postgresql://user:password@cloud-host:5432/db_name
   SYNC_INTERVAL_SECONDS=300
   ```
   *Note: Set `POSTGRES_URL` to empty if testing completely offline without a cloud DB.*

## Running the Application

Run the server with Uvicorn (restricted to 1 worker for the Pi):
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

The database (`local_scans.db`) will be automatically created on startup.
You can view the API documentation by navigating to `http://127.0.0.1:8000/docs`.

## Testing the Pipeline

A mock script is provided to simulate the ML/hardware sending scan results to the API:
```bash
python scripts/mock_ml_client.py
```

## Production Deployment (systemd on Raspberry Pi)

To ensure the backend starts automatically on boot:

1. Create a service file: `sudo nano /etc/systemd/system/med-backend.service`
2. Add the following content (adjust paths accordingly):
   ```ini
   [Unit]
   Description=Anti-Counterfeit Medicine Backend API
   After=network.target

   [Service]
   User=pi
   Group=pi
   WorkingDirectory=/home/pi/backend
   Environment="PATH=/home/pi/backend/venv/bin"
   ExecStart=/home/pi/backend/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```
3. Enable and start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable med-backend
   sudo systemctl start med-backend
   ```
