import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const SCAN_ENDPOINT = import.meta.env.VITE_SCAN_ENDPOINT || "/YOUR_EXISTING_SCAN_ENDPOINT";
const RESULT_ENDPOINT = import.meta.env.VITE_RESULT_ENDPOINT || "/YOUR_EXISTING_RESULT_ENDPOINT";
const HEALTH_ENDPOINT = import.meta.env.VITE_HEALTH_ENDPOINT || "/health";

function App() {
  const [screen, setScreen] = useState("scan");
  const [status, setStatus] = useState("ready");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [backendOnline, setBackendOnline] = useState(false);

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

  async function startScan() {
    setError("");
    setResult(null);
    setStatus("scanning");
    setScreen("scan");

    try {
      // This adapter deliberately does not invent the backend request schema.
      // Update the body below after inspecting the existing backend endpoint.
      const response = await fetch(`${API_BASE}${SCAN_ENDPOINT}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({})
      });

      if (!response.ok) {
        throw new Error(`Backend returned ${response.status}`);
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

    return {
      verdict,
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
                The connected sensing hardware will acquire the sample data and send it to the existing
                authentication backend.
              </p>

              {status === "scanning" ? (
                <div className="progress-wrap">
                  <div className="progress-label">
                    <span>PROCESSING</span><span>Please wait</span>
                  </div>
                  <div className="progress"><div className="progress-fill" /></div>
                </div>
              ) : (
                <button className="primary" onClick={startScan} disabled={!backendOnline}>
                  START SCAN
                </button>
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

          <div className="raw-toggle">
            Backend response received successfully. Detailed result remains available through the existing backend/application.
          </div>
        </section>
      )}
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
