const loadingOverlay = document.getElementById("loading-overlay");
const messageHost = document.getElementById("message-host");

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function showLoading(isVisible) {
  if (!loadingOverlay) return;
  loadingOverlay.classList.toggle("d-none", !isVisible);
}

function showMessage(message, tone = "error") {
  if (!messageHost) return;
  const node = document.createElement("div");
  node.className = `app-message ${tone === "success" ? "is-success" : "is-error"}`;
  node.textContent = message;
  messageHost.appendChild(node);
  window.setTimeout(() => {
    node.remove();
  }, 4200);
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || "The request could not be completed.");
  }
  return payload;
}

function setWeatherFields(form, autofill) {
  ["temperature", "humidity", "rainfall"].forEach((name) => {
    const input = form.querySelector(`[name="${name}"]`);
    if (input && autofill[name] !== undefined) {
      input.value = autofill[name];
    }
  });
}

function updateWeatherPanel(panel, weather) {
  if (!panel) return;
  const rainfallLabel = weather.rainfall_window === "none"
    ? "No recent rainfall reported"
    : `${weather.rainfall.toFixed(1)} mm in the last ${weather.rainfall_window}`;

  panel.innerHTML = `
    <div class="result-stat-grid">
      <article class="stat-card">
        <span>Location</span>
        <strong>${escapeHtml(weather.location_name)} ${weather.country_code ? `(${escapeHtml(weather.country_code)})` : ""}</strong>
      </article>
      <article class="stat-card">
        <span>Conditions</span>
        <strong>${escapeHtml(weather.conditions)}</strong>
      </article>
      <article class="stat-card">
        <span>Temperature</span>
        <strong>${Number(weather.temperature).toFixed(1)} °C</strong>
      </article>
      <article class="stat-card">
        <span>Humidity</span>
        <strong>${Number(weather.humidity).toFixed(1)}%</strong>
      </article>
    </div>
    <div class="inline-note mt-3">${rainfallLabel}. You can still edit the values below before prediction.</div>
  `;
}

function geolocationErrorMessage(error) {
  if (!error) return "We could not access your location.";
  if (error.code === 1) return "Location permission was denied. You can still enter weather values manually.";
  if (error.code === 2) return "Your location could not be determined right now. Please try again.";
  if (error.code === 3) return "Location request timed out. Please try again or enter values manually.";
  return "We could not access your location.";
}

function initWeatherButtons() {
  document.querySelectorAll("[data-weather-button]").forEach((button) => {
    button.addEventListener("click", async () => {
      const targetFormId = button.dataset.targetForm;
      const form = document.getElementById(targetFormId);
      if (!form) return;
      const panel = form.querySelector("[data-weather-panel]");

      if (!navigator.geolocation) {
        showMessage("Geolocation is not supported in this browser.");
        return;
      }

      if (!window.isSecureContext && !["localhost", "127.0.0.1"].includes(window.location.hostname)) {
        showMessage("Browser geolocation needs HTTPS in production. Use HTTPS or localhost.");
        return;
      }

      showLoading(true);
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          try {
            const payload = await fetchJson("/get_weather", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                latitude: position.coords.latitude,
                longitude: position.coords.longitude,
              }),
            });
            setWeatherFields(form, payload.autofill);
            updateWeatherPanel(panel, payload.weather);
            showMessage("Weather data loaded successfully.", "success");
          } catch (error) {
            showMessage(error.message || "Weather data could not be loaded.");
          } finally {
            showLoading(false);
          }
        },
        (error) => {
          showLoading(false);
          showMessage(geolocationErrorMessage(error));
        },
        {
          enableHighAccuracy: true,
          timeout: 12000,
          maximumAge: 300000,
        },
      );
    });
  });
}

function createConfidenceRows(items, labelGetter) {
  if (!items || items.length === 0) {
    return `<p class="mb-0 text-muted">No ranked predictions available.</p>`;
  }

  return `
    <div class="confidence-stack">
      ${items
        .map((item) => {
          const label = labelGetter(item);
          const confidencePct = Number(item.confidence_pct ?? item.confidence ?? 0);
          const normalized = item.confidence_pct !== undefined ? confidencePct : confidencePct * 100;
          return `
            <div class="confidence-row">
              <div class="confidence-meta">
                <span>${escapeHtml(label)}</span>
                <strong>${normalized.toFixed(2)}%</strong>
              </div>
              <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${Math.max(0, Math.min(normalized, 100))}%;"></div>
              </div>
            </div>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderCropResult(panel, data) {
  const recommendations = data.predictions || [];
  panel.innerHTML = `
    <section class="glass-panel result-panel h-100 animate-in" style="border-left: 6px solid var(--primary); padding: 30px;">
       <div class="panel-heading d-flex justify-content-between align-items-center mb-4">
            <div>
              <span class="section-kicker">Optimal Choice</span>
              <h2 class="mb-0">Recommended: ${escapeHtml(data.predicted_crop)}</h2>
            </div>
            ${data.rule_used ? `<div class="badge bg-warning-subtle text-warning border border-warning-subtle p-2" style="font-size: 0.7rem;"><i class="bi bi-shield-check me-1"></i> Model + Rule Engine</div>` : ''}
       </div>

       <div class="recommendation-list">
         ${recommendations.map((rec, idx) => `
            <div class="recommendation-item mb-3 p-3 border rounded-4 ${idx === 0 ? 'border-primary bg-primary-subtle' : 'bg-white shadow-sm'}">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <h4 class="mb-0 fw-bold">
                        ${idx + 1}. ${escapeHtml(rec.crop)}
                        ${rec.duration_match === false ? `<span class="badge bg-danger text-white ms-2" style="font-size: 0.6rem;">Duration Mismatch</span>` : ''}
                    </h4>
                    <span class="fw-bold text-primary">${(rec.confidence * 100).toFixed(1)}% Match</span>
                </div>
                <div class="progress mb-2" style="height: 6px;">
                    <div class="progress-bar bg-primary" style="width: ${rec.confidence * 100}%"></div>
                </div>
                <p class="small text-muted mb-0">Maturation: ${escapeHtml(rec.duration_range || 'Unknown')}</p>
            </div>
         `).join('')}
       </div>

       <div class="fertilizer-card mt-4 p-4 rounded-4 bg-dark text-white shadow-lg">
            <h5 class="fw-bold mb-2"><i class="bi bi-droplet-half text-info me-2"></i>Prescribed Fertilizer</h5>
            <p class="mb-0 text-white-50">${escapeHtml(data.fertilizer_recommendation || 'No recommendation available.')}</p>
       </div>

       <div class="mt-4">
            <a href="/advisor" class="btn btn-outline-primary rounded-pill w-100">View Detailed Growth Timeline</a>
       </div>
    </section>
  `;
}

function renderDiseaseResult(panel, data) {
  const topItems = (data.top_predictions || []).map((item) => ({
    confidence: item.confidence,
    label: `${item.crop_name} - ${item.disease_name}`,
  }));

  panel.innerHTML = `
    <div class="panel-heading">
      <span class="section-kicker">Diagnosis output</span>
      <h2>${escapeHtml(data.disease_name)}</h2>
    </div>
    <div class="result-layout">
      <div class="result-chip"><i class="bi bi-clipboard2-pulse"></i> ${data.is_healthy ? "Healthy leaf detected" : "Disease symptoms detected"}</div>
      <div class="result-stat-grid">
        <article class="stat-card">
          <span>Crop</span>
          <strong>${escapeHtml(data.crop_name)}</strong>
        </article>
        <article class="stat-card">
          <span>Disease confidence</span>
          <strong>${(Number(data.disease_confidence) * 100).toFixed(2)}%</strong>
        </article>
        <article class="stat-card">
          <span>Treatment</span>
          <strong>${escapeHtml(data.suggested_treatment)}</strong>
        </article>
      </div>
      <div class="result-media-grid">
        <article class="result-media-card">
          <h3>Uploaded image</h3>
          <img class="result-image" src="${escapeHtml(data.preview_url)}" alt="Uploaded leaf">
        </article>
        <article class="result-media-card">
          <h3>Grad-CAM heatmap</h3>
          <img class="result-image" src="${escapeHtml(data.heatmap_url)}" alt="Grad-CAM heatmap">
        </article>
      </div>
      <div class="result-card-lite">
        <h3 class="mb-3">Top disease candidates</h3>
        ${createConfidenceRows(topItems, (item) => item.label)}
      </div>
      <div class="result-card-lite">
        <h3 class="mb-3">Additional tips</h3>
        <ul class="tip-list">${(data.tips || []).map((tip) => `<li>${escapeHtml(tip)}</li>`).join("")}</ul>
      </div>
    </div>
  `;
}

function renderHybridResult(panel, data) {
  const cropPrediction = data.crop || data.crop_prediction || {};
  const hybrid = data.disease || data.hybrid_prediction || {};
  const cropItems = cropPrediction.predictions || [];
  const diseaseItems = (hybrid.top_predictions || []).map((item) => ({
    confidence: item.confidence,
    label: `${item.crop_name} - ${item.disease_name}`,
  }));

  panel.innerHTML = `
    <div class="panel-heading">
      <span class="section-kicker">Main output page</span>
      <h2>${escapeHtml(hybrid.crop_name || cropPrediction.predicted_crop || "Hybrid result")}</h2>
    </div>
    <div class="result-layout">
      <div class="result-chip"><i class="bi bi-diagram-3"></i> Hybrid decision confidence ${(Number(hybrid.disease_confidence || 0) * 100).toFixed(2)}%</div>
      <div class="result-stat-grid">
        <article class="stat-card">
          <span>Crop from structured model</span>
          <strong>${escapeHtml(cropPrediction.predicted_crop || "Unknown")}</strong>
        </article>
        <article class="stat-card">
          <span>Final disease</span>
          <strong>${escapeHtml(hybrid.disease_name || "Unknown")}</strong>
        </article>
        <article class="stat-card">
          <span>Final crop</span>
          <strong>${escapeHtml(hybrid.crop_name || cropPrediction.predicted_crop || "Unknown")}</strong>
        </article>
        <article class="stat-card">
          <span>Fertilizer hint</span>
          <strong>${escapeHtml(hybrid.fertilizer_recommendation || cropPrediction.fertilizer_recommendation || "Not available")}</strong>
        </article>
      </div>
      <div class="result-media-grid">
        <article class="result-media-card">
          <h3>Leaf preview</h3>
          <img class="result-image" src="${escapeHtml(hybrid.preview_url)}" alt="Uploaded leaf">
        </article>
        <article class="result-media-card">
          <h3>Grad-CAM heatmap</h3>
          <img class="result-image" src="${escapeHtml(hybrid.heatmap_url)}" alt="Grad-CAM heatmap">
        </article>
      </div>
      <div class="result-card-lite">
        <h3 class="mb-3">Ranked crop predictions</h3>
        ${createConfidenceRows(cropItems, (item) => item.label)}
      </div>
      <div class="result-card-lite">
        <h3 class="mb-3">Ranked disease candidates</h3>
        ${createConfidenceRows(diseaseItems, (item) => item.label)}
      </div>
      <div class="result-card-lite">
        <h3 class="mb-3">Suggested treatment</h3>
        <p class="mb-0">${escapeHtml(hybrid.suggested_treatment || "No treatment suggestion available.")}</p>
      </div>
      <div class="result-card-lite">
        <h3 class="mb-3">Additional field tips</h3>
        <ul class="tip-list">${(hybrid.tips || []).map((tip) => `<li>${escapeHtml(tip)}</li>`).join("")}</ul>
      </div>
    </div>
  `;
}

function renderErrorPanel(panel, title, message) {
  panel.innerHTML = `
    <div class="panel-heading">
      <span class="section-kicker">Request failed</span>
      <h2>${escapeHtml(title)}</h2>
    </div>
    <div class="inline-note">${escapeHtml(message)}</div>
  `;
}

function initFormSubmission(formId, endpoint, panelId, renderer, imageRequired = false) {
  const form = document.getElementById(formId);
  const panel = document.getElementById(panelId);
  if (!form || !panel) return;

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const captureInput = form.querySelector("[data-capture-input]");
    const fileInput = form.querySelector("[data-image-input]");
    if (imageRequired && !captureInput?.value && !(fileInput && fileInput.files && fileInput.files.length > 0)) {
      showMessage("Please upload a leaf image or capture one with the webcam.");
      return;
    }

    showLoading(true);
    try {
      const payload = await fetchJson(endpoint, {
        method: "POST",
        body: new FormData(form),
      });
      renderer(panel, payload);
      showMessage("Prediction completed successfully.", "success");
    } catch (error) {
      renderErrorPanel(panel, "Unable to complete prediction", error.message || "Please try again.");
      showMessage(error.message || "Please try again.");
    } finally {
      showLoading(false);
    }
  });
}

function initMediaForm(formId) {
  const form = document.getElementById(formId);
  if (!form) return;

  const dropzone = form.querySelector("[data-dropzone]");
  const input = form.querySelector("[data-image-input]");
  const previewImage = form.querySelector("[data-preview-image]");
  const previewEmpty = form.querySelector("[data-preview-empty]");
  const captureInput = form.querySelector("[data-capture-input]");
  const webcamShell = form.querySelector("[data-webcam-shell]");
  const video = form.querySelector("[data-webcam-video]");
  const canvas = form.querySelector("[data-webcam-canvas]");
  const startButton = form.querySelector("[data-webcam-start]");
  const captureButton = form.querySelector("[data-webcam-capture]");
  const stopButton = form.querySelector("[data-webcam-stop]");

  let mediaStream = null;

  const setPreview = (src) => {
    if (!previewImage || !previewEmpty) return;
    previewImage.src = src;
    previewImage.classList.remove("d-none");
    previewEmpty.classList.add("d-none");
  };

  const clearCamera = () => {
    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
      mediaStream = null;
    }
    webcamShell?.classList.add("d-none");
    if (startButton) startButton.disabled = false;
    if (captureButton) captureButton.disabled = true;
    if (stopButton) stopButton.disabled = true;
  };

  if (dropzone && input) {
    dropzone.addEventListener("click", () => input.click());
    dropzone.addEventListener("dragover", (event) => {
      event.preventDefault();
      dropzone.classList.add("is-active");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("is-active"));
    dropzone.addEventListener("drop", (event) => {
      event.preventDefault();
      dropzone.classList.remove("is-active");
      const [file] = event.dataTransfer.files || [];
      if (file) {
        try {
          const dataTransfer = new DataTransfer();
          dataTransfer.items.add(file);
          input.files = dataTransfer.files;
          captureInput.value = "";
          setPreview(URL.createObjectURL(file));
        } catch (_error) {
          const reader = new FileReader();
          reader.onload = () => {
            captureInput.value = reader.result;
            input.value = "";
            setPreview(reader.result);
          };
          reader.readAsDataURL(file);
        }
      }
    });

    input.addEventListener("change", () => {
      const [file] = input.files || [];
      if (!file) return;
      captureInput.value = "";
      setPreview(URL.createObjectURL(file));
    });
  }

  startButton?.addEventListener("click", async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      showMessage("Webcam capture is not supported in this browser.");
      return;
    }

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false,
      });
      video.srcObject = mediaStream;
      webcamShell?.classList.remove("d-none");
      startButton.disabled = true;
      captureButton.disabled = false;
      stopButton.disabled = false;
    } catch (_error) {
      showMessage("The webcam could not be started. Check browser permissions and camera access.");
    }
  });

  captureButton?.addEventListener("click", () => {
    if (!video || !canvas) return;
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
    captureInput.value = dataUrl;
    if (input) input.value = "";
    setPreview(dataUrl);
    clearCamera();
    showMessage("Webcam frame captured.", "success");
  });

  stopButton?.addEventListener("click", clearCamera);
  window.addEventListener("beforeunload", clearCamera);
}

document.addEventListener("DOMContentLoaded", () => {
  initWeatherButtons();
  initMediaForm("disease-detection-form");
  initMediaForm("hybrid-prediction-form");
  initFormSubmission("crop-prediction-form", "/predict_crop", "crop-result-panel", renderCropResult, false);
  initFormSubmission("disease-detection-form", "/predict_disease", "disease-result-panel", renderDiseaseResult, true);
  initFormSubmission("hybrid-prediction-form", "/hybrid_predict", "hybrid-result-panel", renderHybridResult, true);
});
