const themeToggle = document.getElementById("themeToggle");
const form = document.getElementById("analyzeForm");
const resultCard = document.getElementById("resultCard");
const statusNode = document.getElementById("status");
const resultNode = document.getElementById("result");
const submitBtn = document.getElementById("submitBtn");

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("smarthire_theme", theme);
}

function initTheme() {
  const saved = localStorage.getItem("smarthire_theme");
  if (saved === "light" || saved === "dark") {
    applyTheme(saved);
    return;
  }
  applyTheme("dark");
}

function safeList(items) {
  if (!Array.isArray(items) || items.length === 0) {
    return "<p class=\"note\">None</p>";
  }
  return `<ul>${items.map((x) => `<li>${x}</li>`).join("")}</ul>`;
}

function renderResult(data) {
  const scores = data.score_card || {};
  const candidate = data.candidate || {};
  const deep = data.deep_analysis || {};
  const quick = data.quick_suggestions || [];

  const targeted = Array.isArray(deep.targeted_improvements)
    ? deep.targeted_improvements
    : [];

  resultNode.innerHTML = `
    <p><b>Candidate:</b> ${candidate.file_name || "Unknown"}</p>
    <p><b>Experience:</b> ${candidate.years_experience || 0} years</p>
    <div class="score-grid">
      <div class="score-item"><b>${scores.overall_score ?? "-"}</b><span>Overall</span></div>
      <div class="score-item"><b>${scores.skills_match ?? "-"}</b><span>Skills</span></div>
      <div class="score-item"><b>${scores.keywords_match ?? "-"}</b><span>Keywords</span></div>
      <div class="score-item"><b>${scores.experience_match ?? "-"}</b><span>Experience</span></div>
      <div class="score-item"><b>${scores.impact_match ?? "-"}</b><span>Impact</span></div>
      <div class="score-item"><b>${scores.confidence ?? "-"}</b><span>Confidence</span></div>
    </div>

    <h3>Missing Keywords</h3>
    ${safeList(scores.missing_keywords)}

    <h3>Evidence Gaps</h3>
    ${safeList(scores.evidence_gaps)}

    <h3>Quick Suggestions</h3>
    ${safeList(quick)}

    <h3>Executive Summary</h3>
    <p>${deep.executive_summary || "No summary returned."}</p>

    <h3>ATS Risks</h3>
    ${safeList(deep.ats_risks)}

    <h3>Targeted Improvements</h3>
    ${
      targeted.length
        ? `<ul>${targeted
            .map(
              (x) =>
                `<li><b>${x.section || "Section"}</b> - ${x.issue || ""}<br/>Rewrite: ${x.rewrite || ""}</li>`
            )
            .join("")}</ul>`
        : "<p class=\"note\">No targeted improvements returned.</p>"
    }

    <h3>Optimized Summary</h3>
    <p>${deep.optimized_professional_summary || ""}</p>
  `;

  if (data.ai_warning) {
    statusNode.innerHTML = `<p class="note">AI warning: ${data.ai_warning}</p>`;
  } else {
    statusNode.innerHTML = `<p class="note">Analysis source: ${deep.model_used || "heuristic-engine"}</p>`;
  }
}

initTheme();

themeToggle.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme");
  applyTheme(current === "dark" ? "light" : "dark");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = document.getElementById("resume").files[0];
  const jobDescription = document.getElementById("jobDescription").value.trim();
  const enableAi = document.getElementById("enableAi").checked;
  const temperature = document.getElementById("temperature").value || "0.2";

  if (!file || !jobDescription) {
    statusNode.innerHTML = `<p class="error">Resume and job description are required.</p>`;
    resultCard.classList.remove("hidden");
    return;
  }

  resultCard.classList.remove("hidden");
  statusNode.innerHTML = `<p class="loading">Running ATS analysis...</p>`;
  resultNode.innerHTML = "";
  submitBtn.disabled = true;
  submitBtn.textContent = "Analyzing...";

  const payload = new FormData();
  payload.append("resume", file);
  payload.append("job_description", jobDescription);
  payload.append("enable_ai", String(enableAi));
  payload.append("temperature", String(temperature));

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: payload,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Analysis failed");
    }

    renderResult(data);
  } catch (error) {
    statusNode.innerHTML = `<p class="error">${error.message}</p>`;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Analyze Resume";
  }
});
