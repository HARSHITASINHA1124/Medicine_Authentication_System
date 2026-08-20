import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const SCAN_ENDPOINT = import.meta.env.VITE_SCAN_ENDPOINT || "/api/scans";
const ML_ENDPOINT = import.meta.env.VITE_ML_ENDPOINT || "/api/scans/analyze";
const RESULT_ENDPOINT = import.meta.env.VITE_RESULT_ENDPOINT || "/YOUR_EXISTING_RESULT_ENDPOINT";
const HEALTH_ENDPOINT = import.meta.env.VITE_HEALTH_ENDPOINT || "/health";

const CHANNELS = ["ch450", "ch500", "ch550", "ch570", "ch600", "ch650"];

function parseCsvLine(line) {
  const values = [];
  let value = "";
  let quoted = false;

  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (character === '"' && line[index + 1] === '"' && quoted) {
      value += '"';
      index += 1;
    } else if (character === '"') {
      quoted = !quoted;
    } else if (character === "," && !quoted) {
      values.push(value.trim());
      value = "";
    } else {
      value += character;
    }
  }

  values.push(value.trim());
  return values;
}

function parseCsv(text) {
  const lines = text.replace(/^\uFEFF/, "").split(/\r?\n/).filter((line) => line.trim());
  if (lines.length !== 10) throw new Error(`CSV must contain exactly 10 readings; found ${lines.length}.`);

  return lines.map((line, rowIndex) => {
    const values = parseCsvLine(line);
    if (values.length !== CHANNELS.length) {
      throw new Error(`Row ${rowIndex + 1} must contain exactly 6 columns.`);
    }
    const reading = {};
    CHANNELS.forEach((channel, channelIndex) => {
      const rawValue = values[channelIndex];
      const numericValue = Number(rawValue);
      if (rawValue === "" || !Number.isFinite(numericValue)) {
        throw new Error(`Row ${rowIndex + 1} has an invalid value in column ${channelIndex + 1}.`);
      }
      reading[channel] = numericValue;
    });
    return reading;
  });
}

function App() {
  const [screen, setScreen] = useState("scan");
  const [status, setStatus] = useState("ready");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [backendOnline, setBackendOnline] = useState(false);
  const [csvFile, setCsvFile] = useState(null);
  const [readings, setReadings] = useState([]);

  useEffect(() => {
    checkHealth();
  }, []);

  async function checkHealth() {
    try {
      const response = await fetch(`${API_BASE}${HEALTH_ENDPOINT}`, { signal: AbortSignal.timeout(2500) });
      setBackendOnline(response.ok);
    } catch {
      setBackendOnline(false);
    }
  }

  async function handleCsvChange(event) {
    const file = event.target.files?.[0];
    setError("");
    setCsvFile(file || null);
    setReadings([]);
    if (!file) return;

    try {
      setReadings(parseCsv(await file.text()));
    } catch (err) {
      setCsvFile(null);
      setError(err.message || "Unable to read the CSV file.");
      event.target.value = "";
    }
  }

  async function startScan() {
    setError("");
    setResult(null);
    setStatus("scanning");
    setScreen("scan");

    try {
      const response = await fetch(`${API_BASE}${ML_ENDPOINT}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ readings })
      });

      if (!response.ok) {
        let detail = `Backend returned ${response.status}`;
        try {
          const errorBody = await response.json();
          detail = errorBody.detail || detail;
        } catch {
          // Keep the HTTP status when the server did not return JSON.
        }
        throw new Error(detail);
      }

      const data = await response.json();
      const normalized = normalizeResult(data);
      setResult(normalized);
      setStatus("complete");
      setScreen("result");
    } catch (err) {
      setStatus("error");
      setError(err.message || "Unable to communicate with the authentication backend.");
    }
  }

  async function fetchResult() {
    try {
      const response = await fetch(`${API_BASE}${RESULT_ENDPOINT}`);
      if (!response.ok) throw new Error(`Backend returned ${response.status}`);
      const data = await response.json();
      setResult(normalizeResult(data));
      setStatus("complete");
      setScreen("result");
      setError("");
    } catch (err) {
      setError(err.message || "Unable to fetch the authentication result.");
    }
  }

  function normalizeResult(data) {
    const raw = JSON.stringify(data).toLowerCase();
    let verdict = "suspicious";
    if (raw.includes("genuine") || raw.includes("authentic")) verdict = "genuine";
    if (raw.includes("counterfeit") || raw.includes("fake")) verdict = "counterfeit";
    if (raw.includes("suspicious") || raw.includes("anomaly")) verdict = "suspicious";

    const modelResult = data.result || data;
    const finalStatus = String(modelResult.final_status || "").toLowerCase();
    if (["genuine", "counterfeit", "suspicious"].includes(finalStatus)) {
      verdict = finalStatus;
    }

    return {
      verdict,
      medicine: modelResult.medicine || modelResult.medicine_name || "Unknown",
      confidence: modelResult.classification_confidence,
      anomalyScore: modelResult.anomaly_score,
      status: modelResult.final_status || verdict.toUpperCase(),
      raw: data
    };
  }

  return (
    <main className="app">
      <header className="topbar">
        <div>
          <div className="brand">MEDICINE AUTHENTICATION</div>
          <div className="subtitle">AI assisted preliminary screening</div>
        </div>
        <div className={`connection ${backendOnline ? "online" : "offline"}`}>
          <span className="dot" />
          {backendOnline ? "SYSTEM READY" : "BACKEND OFFLINE"}
        </div>
      </header>

      {screen === "scan" && (
        <section className="screen scan-screen">
          <div className="hero">
            <span className="eyebrow">AUTHENTICATION TERMINAL</span>
            <h1>{status === "scanning" ? "Scanning medicine sample" : "Ready for authentication"}</h1>
            <p>
              Place the medicine sample in the sensing unit and start the authentication process.
            </p>
          </div>

          <div className="scan-card">
            <div className={`scanner ${status === "scanning" ? "active" : ""}`}>
              <div className="scanner-ring" />
              <div className="scanner-core">SCAN</div>
            </div>

            <div className="scan-copy">
              <h2>{status === "scanning" ? "Capturing spectral data..." : "Medicine sample"}</h2>
              <p>
                Upload the sensor CSV containing six unnamed columns and exactly 10 readings. The readings will be
                sent to the configured ML service for authentication.
              </p>

              {status === "scanning" ? (
                <div className="progress-wrap">
                  <div className="progress-label">
                    <span>PROCESSING</span><span>Please wait</span>
                  </div>
                  <div className="progress"><div className="progress-fill" /></div>
                </div>
              ) : (
                <div className="scan-controls">
                  <label className="file-picker">
                    <span>CHOOSE CSV FILE</span>
                    <input type="file" accept=".csv,text/csv" onChange={handleCsvChange} />
                  </label>
                  <div className="file-status">
                    {csvFile ? `${csvFile.name} · ${readings.length}/10 readings loaded` : "No CSV selected"}
                  </div>
                  <button className="primary" onClick={startScan} disabled={readings.length !== 10}>
                  START SCAN
                  </button>
                </div>
              )}

              {error && <div className="error">{error}</div>}
            </div>
          </div>

          <div className="bottom-note">
            <span>Raspberry Pi display</span>
            <span>•</span>
            <span>AS7262 sensing workflow</span>
            <span>•</span>
            <span>Local interface</span>
          </div>
        </section>
      )}

      {screen === "result" && result && (
        <section className="screen result-screen">
          <div className="hero center">
            <span className="eyebrow">AUTHENTICATION RESULT</span>
            <h1>Sample analysis complete</h1>
            <p>The authentication backend has returned a result.</p>
          </div>

          <div className={`result-card ${result.verdict}`}>
            <div className="result-icon">
              {result.verdict === "genuine" ? "✓" : result.verdict === "counterfeit" ? "×" : "!"}
            </div>
            <div className="result-word">{result.verdict.toUpperCase()}</div>
            <div className="result-medicine">{result.medicine}</div>
            <div className="result-metrics">
              <span>Classification confidence: {result.confidence == null ? "N/A" : Number(result.confidence).toFixed(4)}</span>
              <span>Anomaly score: {result.anomalyScore == null ? "N/A" : Number(result.anomalyScore).toLocaleString()}</span>
            </div>
            <div className="result-description">
              {result.verdict === "genuine" && "The sample is classified as genuine by the current authentication model."}
              {result.verdict === "counterfeit" && "The sample is classified as counterfeit by the current authentication model."}
              {result.verdict === "suspicious" && "The sample requires further verification based on the current authentication result."}
            </div>
          </div>

          <div className="result-actions">
            <button className="secondary" onClick={() => { setResult(null); setStatus("ready"); setScreen("scan"); }}>
              NEW SCAN
            </button>
            <button className="secondary" onClick={fetchResult}>
              REFRESH RESULT
            </button>
          </div>

          <details className="raw-toggle">
            <summary>VIEW ML RESPONSE</summary>
            <pre>{JSON.stringify(result.raw, null, 2)}</pre>
          </details>
        </section>
      )}
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
