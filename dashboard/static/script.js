const dropzone = document.getElementById("dropzone");
const dropzoneInner = document.getElementById("dropzoneInner");
const dropzoneSweep = document.getElementById("dropzoneSweep");
const fileInput = document.getElementById("fileInput");
const resultsBody = document.getElementById("resultsBody");

let currentModel = "b";
let lastFile = null;

const CLASS_COLOR_FALLBACK = {
  sea: "#0A1B26", oil_spill: "#4FD8E8", look_alike: "#E8A23D", ship: "#C83C3C", land: "#5A8C5A",
};

// ---------- dropzone interactions ----------
dropzone.addEventListener("click", () => fileInput.click());

["dragenter", "dragover"].forEach(evt =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); })
);
["dragleave", "drop"].forEach(evt =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener("change", (e) => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

function showPreview(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    let preview = dropzone.querySelector(".dropzone__preview");
    if (!preview) {
      preview = document.createElement("div");
      preview.className = "dropzone__preview";
      const img = document.createElement("img");
      preview.appendChild(img);
      dropzone.appendChild(preview);
    }
    preview.querySelector("img").src = e.target.result;
    dropzone.classList.add("has-image");
  };
  reader.readAsDataURL(file);
}

function handleFile(file) {
  lastFile = file;
  showPreview(file);
  runScan(file);
}

// ---------- inference ----------
async function runScan(file) {
  dropzoneSweep.classList.add("active");
  resultsBody.innerHTML = `<p class="results__placeholder">Scanning…</p>`;

  const form = new FormData();
  form.append("file", file);
  form.append("model", currentModel);

  try {
    const res = await fetch("/api/predict", { method: "POST", body: form });
    const data = await res.json();
    renderResults(data);
  } catch (err) {
    resultsBody.innerHTML = `<p class="results__placeholder">Couldn't reach the server. Is it running? (${err})</p>`;
  } finally {
    dropzoneSweep.classList.remove("active");
  }
}

function renderResults(data) {
  const colors = data.class_colors && Object.keys(data.class_colors).length
    ? Object.fromEntries(Object.entries(data.class_colors).map(([k, v]) => [k, `rgb(${v.join(",")})`]))
    : CLASS_COLOR_FALLBACK;

  const rows = Object.entries(data.class_breakdown)
    .sort((a, b) => b[1] - a[1])
    .map(([name, pct]) => `
      <div class="breakdown__row">
        <span class="breakdown__label">${name.replace("_", " ")}</span>
        <div class="breakdown__bar">
          <div class="breakdown__fill" style="width:${pct}%; background:${colors[name] || "#4FD8E8"}"></div>
        </div>
        <span class="breakdown__pct">${pct.toFixed(1)}%</span>
      </div>
    `).join("");

  resultsBody.innerHTML = `
    <div class="result-overlay">
      <img src="data:image/png;base64,${data.overlay_png_base64}" alt="Segmentation overlay">
    </div>
    <div class="result-meta">
      <span>ResNet34 U-Net</span>
      <span>${data.inference_ms} ms</span>
    </div>
    <div class="breakdown">${rows}</div>
  `;
}

// ---------- comparison chart ----------
async function loadComparison() {
  const chart = document.getElementById("compareChart");
  const verdictEl = document.getElementById("compareVerdict");
  if (!chart || !verdictEl) return;

  try {
    const res = await fetch("/api/verdict");
    if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
    const data = await res.json();

    const metricsRows = [
      ["Pixel Accuracy", "pixel_accuracy"],
      ["Mean IoU (Jaccard)", "mean_iou"],
      ["Dice Coefficient (F1)", "macro_f1"],
    ];

    chart.innerHTML = metricsRows.map(([label, key]) => {
      const val = (data.model_b[key] ?? 0) * 100;
      return `
        <div class="compare-row">
          <span class="compare-row__label">${label}</span>
          <div class="compare-row__bars">
            <div class="compare-bar">
              <div class="compare-bar__track"><div class="compare-bar__fill compare-bar__fill--b" style="width:${val}%"></div></div>
              <span class="compare-bar__value">${val.toFixed(1)}%</span>
            </div>
          </div>
        </div>
      `;
    }).join("");

    verdictEl.textContent = (data.demo ? "[Demo numbers] " : "") + data.verdict;
  } catch (err) {
    console.error("Error loading metrics:", err);
    verdictEl.textContent = "Error loading model metrics. Please ensure the backend server is running and refresh the page.";
  }
}

loadComparison();
