import { useEffect, useState } from "react";
import { getOptions, predictRisk } from "./api.js";

// How each number input behaves (step size). Labels and ranges come from the API.
const STEPS = {
  study_hours_per_week: "0.5",
  attendance_pct: "0.1",
  previous_sem_gpa: "0.01",
  assignments_submitted_pct: "1",
  internal_test_score: "1",
  commute_minutes: "1",
  sleep_hours: "0.1",
};

// Starting answers for the Yes/No and Low/Medium/High questions.
const DEFAULT_CHOICES = {
  part_time_job: "No",
  family_support: "Medium",
  internet_access: "Yes",
  extracurricular: "No",
};

function ResultCard({ result }) {
  const percent = Math.round(result.probability * 100);
  const thresholdPercent = Math.round(result.threshold * 100);
  const level = result.risk_level.toLowerCase(); // "low" | "medium" | "high" -> CSS class

  return (
    <section className={`card result ${level}`}>
      <p className="muted">Chance of failing the final exam</p>
      <p className="big">
        {percent}% <span className={`badge ${level}`}>{result.risk_level} risk</span>
      </p>

      {/* Probability bar with a marker at the alert threshold */}
      <div className="bar" aria-label={`Probability ${percent} percent`}>
        <div className={`bar-fill ${level}`} style={{ width: `${percent}%` }} />
        <div className="bar-marker" style={{ left: `${thresholdPercent}%` }} title="alert threshold" />
      </div>
      <p className="small muted">The line marks the alert threshold ({thresholdPercent}%).</p>

      <p className="verdict">
        {result.at_risk ? "Flagged for a mentor check-in." : "Not flagged."}
      </p>

      {result.reasons.length > 0 ? (
        <>
          <p className="muted">Main reasons:</p>
          <ul>
            {result.reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        </>
      ) : (
        <p className="muted">No input stands out as clearly worse than a typical student.</p>
      )}
      <p className="small muted">{result.note} Model: {result.model}.</p>
    </section>
  );
}

function App() {
  const [options, setOptions] = useState(null); // labels, ranges and allowed values from the API
  const [form, setForm] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [fieldErrors, setFieldErrors] = useState({});
  const [loading, setLoading] = useState(false);

  // Load the form definition once when the page opens; start from a typical student.
  useEffect(() => {
    getOptions()
      .then((data) => {
        setOptions(data);
        const start = { ...DEFAULT_CHOICES };
        for (const name of data.numeric_features) {
          start[name] = String(data.typical_values[name]);
        }
        setForm(start);
      })
      .catch((err) => setError(err.message));
  }, []);

  function handleChange(event) {
    const { name, value } = event.target;
    setForm({ ...form, [name]: value });
    setFieldErrors({ ...fieldErrors, [name]: undefined }); // clear the message for this field
  }

  // Simple checks in the browser; the API checks everything again.
  function checkForm() {
    const problems = {};
    for (const name of options.numeric_features) {
      if (form[name] === "") problems[name] = "Required";
      else if (Number.isNaN(Number(form[name]))) problems[name] = "Must be a number";
    }
    return problems;
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setResult(null);
    setError("");
    const problems = checkForm();
    if (Object.keys(problems).length > 0) {
      setFieldErrors(problems);
      return;
    }
    setLoading(true);
    try {
      const data = await predictRisk(form);
      setResult(data);
    } catch (err) {
      setError(err.message);
      setFieldErrors(err.fields || {});
    } finally {
      setLoading(false);
    }
  }

  if (!options && !error) {
    return <main className="page"><p className="muted">Loading form…</p></main>;
  }

  return (
    <main className="page">
      <header>
        <h1>Student At-Risk Check</h1>
        <p className="muted">
          Enter a student's week-5 details. A flag means "offer support", never "punish".
        </p>
      </header>

      {error && <div className="alert">{error}</div>}

      {options && form && (
        <form onSubmit={handleSubmit} className="card" noValidate>
          <div className="grid">
            {options.numeric_features.map((name) => {
              const [min, max] = options.numeric_ranges[name];
              return (
                <label key={name}>
                  {options.feature_labels[name]}
                  <input
                    type="number"
                    name={name}
                    value={form[name]}
                    min={min}
                    max={max}
                    step={STEPS[name] || "any"}
                    onChange={handleChange}
                    className={fieldErrors[name] ? "invalid" : ""}
                  />
                  <span className="hint">{fieldErrors[name] || `Allowed: ${min} – ${max}`}</span>
                </label>
              );
            })}

            {options.categorical_features.map((name) => (
              <label key={name}>
                {options.feature_labels[name]}
                <select
                  name={name}
                  value={form[name]}
                  onChange={handleChange}
                  className={fieldErrors[name] ? "invalid" : ""}
                >
                  {options.categories[name].map((value) => (
                    <option key={value} value={value}>{value}</option>
                  ))}
                </select>
                {fieldErrors[name] && <span className="hint error">{fieldErrors[name]}</span>}
              </label>
            ))}
          </div>

          <button type="submit" disabled={loading}>
            {loading ? "Checking…" : "Check risk"}
          </button>
        </form>
      )}

      {result && <ResultCard result={result} />}
    </main>
  );
}

export default App;
