import { useState } from "react";
import "./App.css";

// Point this at the FastAPI backend (bems-rag serving app).
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// A few real BDG2 building ids to try.
const SAMPLE_BUILDINGS = [
  "Panther_lodging_Dean",
  "Panther_office_Hannah",
  "Robin_education_Estella",
];

export default function App() {
  const [buildingId, setBuildingId] = useState(SAMPLE_BUILDINGS[0]);
  const [question, setQuestion] = useState("What is the floor area?");
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function ask() {
    setLoading(true);
    setError(null);
    setAnswer(null);
    try {
      const res = await fetch(`${API_URL}/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ building_id: buildingId, text: question }),
      });
      if (!res.ok) throw new Error(`API returned ${res.status}`);
      const data = await res.json();
      setAnswer(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>RAG Query Playground</h1>
        <p className="sub">
          Tenant-scoped, grounded answers over real ASHRAE building data ·{" "}
          <span className="mono">bems-rag-mlops</span>
        </p>
      </header>

      <section className="panel">
        <label>
          Building
          <select value={buildingId} onChange={(e) => setBuildingId(e.target.value)}>
            {SAMPLE_BUILDINGS.map((b) => (
              <option key={b} value={b}>{b}</option>
            ))}
          </select>
        </label>

        <label>
          Question
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask about this building…"
          />
        </label>

        <button onClick={ask} disabled={loading || !question.trim()}>
          {loading ? "Asking…" : "Ask"}
        </button>
      </section>

      {error && <div className="error">Error: {error}</div>}

      {answer && (
        <section className="result">
          <div className="answer-head">
            <span className={`badge ${answer.grounded ? "ok" : "warn"}`}>
              {answer.grounded ? "grounded" : "ungrounded"}
            </span>
            <span className="served">served by {answer.served_by}</span>
          </div>
          <p className="answer-text">{answer.text}</p>
          <div className="meta">
            <span className="mono">
              {answer.contexts?.length || 0} context chunk(s): {answer.contexts?.join(", ") || "none"}
            </span>
          </div>
        </section>
      )}

      <footer>
        <span>
          Backend: <span className="mono">{API_URL}</span> · start it with{" "}
          <span className="mono">python scripts/serve.py</span>
        </span>
      </footer>
    </div>
  );
}
