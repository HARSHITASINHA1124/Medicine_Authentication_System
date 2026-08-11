import requests
import time
import random

BASE_URL = "http://127.0.0.1:8000/api"

def create_mock_batch():
    batch_data = {
        "batch_id": "B1001",
        "medicine_name": "Paracetamol 500mg",
        "manufacturer": "HealthCorp",
        "batch_number": "HC-0992",
        "manufacturing_date": "2025-01-01",
        "expiry_date": "2028-01-01"
    }
    
    print(f"Creating batch {batch_data['batch_id']}...")
    try:
        response = requests.post(f"{BASE_URL}/batches/", json=batch_data)
        if response.status_code in (201, 400):
            print("Batch ready.")
        else:
            print("Error creating batch:", response.text)
    except requests.exceptions.ConnectionError:
        print("Could not connect to the API. Is it running?")
        exit(1)

def simulate_ml_scans():
    print("Simulating ML hardware pipeline...")
    
    for i in range(5):
        classification = random.choices(["Genuine", "Counterfeit"], weights=[0.8, 0.2])[0]
        confidence = round(random.uniform(0.85, 0.99), 2)
        anomaly = round(random.uniform(0.01, 0.15) if classification == "Genuine" else random.uniform(0.6, 0.95), 2)
        
        scan_data = {
            "batch_id": "B1001",
            "medicine_name": "Paracetamol 500mg",
            "classification": classification,
            "confidence_score": confidence,
            "anomaly_score": anomaly
        }
        
        print(f"Sending scan {i+1}: {classification} (Conf: {confidence}, Anomaly: {anomaly})")
        response = requests.post(f"{BASE_URL}/scans/", json=scan_data)
        print("Response:", response.json())
        time.sleep(1)

if __name__ == "__main__":
    create_mock_batch()
    simulate_ml_scans()
