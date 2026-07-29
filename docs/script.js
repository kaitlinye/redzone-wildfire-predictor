"use strict";

const PREDICTIONS_URL = "data/predictions.json";
const riskColors = {
  Low: "#5ca53d",
  Medium: "#f2be2e",
  High: "#f28b24",
  Extreme: "#e3483b"
};

let locations = [];
let predictionMetadata = null;
let selectedId = null;
let activePopupMarker = null;
let heatLayer = null;

const californiaBounds = L.latLngBounds([31.8, -125.1], [42.7, -113.7]);
const naturalView = { center: [38.35, -120.5], zoom: 6 };
const map = L.map("map", {
  zoomControl: false,
  attributionControl: false,
  minZoom: 5,
  maxZoom: 12,
  maxBounds: californiaBounds,
  maxBoundsViscosity: 0.82
}).setView(naturalView.center, naturalView.zoom);

const terrainLayers = {
  soft: L.tileLayer("https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png", {
    maxZoom: 19,
    subdomains: "abcd",
    attribution: "© OpenStreetMap contributors © CARTO"
  }),
  balanced: L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}", {
    maxZoom: 19,
    attribution: "Tiles © Esri"
  }),
  detailed: L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    maxZoom: 19,
    attribution: "Tiles © Esri"
  })
};

let activeTerrainLayer = terrainLayers.balanced.addTo(map);

L.rectangle(californiaBounds, {
  color: "transparent",
  fillColor: "#707579",
  fillOpacity: 0.22,
  interactive: false
}).addTo(map);

const analysisOutline = L.polygon([
  [42.0, -124.35], [42.0, -120.0], [39.0, -120.0], [35.0, -114.64],
  [32.53, -117.12], [34.0, -120.45], [38.0, -123.0], [41.0, -124.25]
], {
  color: "#f3eee5",
  weight: 2,
  fillColor: "#ffffff",
  fillOpacity: 0.03,
  interactive: false
}).addTo(map);

const markerLayer = L.layerGroup().addTo(map);

function escapeHTML(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatDate(dateString) {
  const date = new Date(`${dateString}T12:00:00`);
  if (Number.isNaN(date.getTime())) return dateString;
  return new Intl.DateTimeFormat("en-US", {
    weekday: "short",
    year: "numeric",
    month: "short",
    day: "numeric"
  }).format(date);
}

function normalizeLocation(item) {
  const conditions = item.conditions || {};
  return {
    id: String(item.id),
    name: String(item.name || item.id),
    area: String(item.area || "California analysis grid"),
    lat: Number(item.lat),
    lng: Number(item.lng),
    forestType: String(item.forest_type || "Not available"),
    score: Number(item.risk_score),
    risk: String(item.risk_level),
    modelScore: Number(item.model_score),
    temperature: Number(conditions.temperature_max_c),
    humidity: Number(conditions.humidity_min_percent),
    wind: Number(conditions.wind_speed_max_kmh),
    rainfall: Number(conditions.precipitation_mm)
  };
}

function isValidLocation(location) {
  return (
    Number.isFinite(location.lat) &&
    Number.isFinite(location.lng) &&
    Number.isFinite(location.score) &&
    Object.prototype.hasOwnProperty.call(
      riskColors,
      location.risk
    )
  );
}

function markerIcon(risk) {
  return L.divIcon({
    className: "risk-marker",
    html: `<span class="pin-shape" style="--pin:${riskColors[risk]}"></span>`,
    iconSize: [24, 31],
    iconAnchor: [12, 30],
    popupAnchor: [0, -29]
  });
}

function renderHeatLayer() {
  if (heatLayer) map.removeLayer(heatLayer);
  if (!locations.length) return;

  const points = locations.map(location => [
    location.lat,
    location.lng,
    Math.max(0.08, location.score / 100)
  ]);

  heatLayer = L.heatLayer(points, {
    radius: 34,
    blur: 28,
    maxZoom: 10,
    minOpacity: 0.2,
    max: 1,
    gradient: {
      0.18: "#5ca53d",
      0.42: "#b6c83e",
      0.58: "#f2be2e",
      0.76: "#f28b24",
      1.0: "#e3483b"
    }
  }).addTo(map);
}

function popupHTML(location) {
  const node = document.getElementById("popup-template").content.cloneNode(true);
  node.querySelector(".popup-risk").textContent =
    `${location.risk.toUpperCase()} — ${location.score.toFixed(1)} percentile`;
  node.querySelector(".popup-risk").style.color = riskColors[location.risk];
  node.querySelector(".popup-pin").style.color = riskColors[location.risk];
  node.querySelector(".popup-title").textContent = location.name;

  const rows = [
    ["Maximum temperature", `${location.temperature.toFixed(1)} °C`],
    ["Minimum humidity", `${location.humidity.toFixed(0)}%`],
    ["Maximum wind", `${location.wind.toFixed(1)} km/h`],
    ["Precipitation", `${location.rainfall.toFixed(2)} mm`],
    ["Fuel model", location.forestType]
  ];
  const list = node.querySelector(".popup-stats");
  rows.forEach(([label, value]) => {
    const row = document.createElement("div");
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = label;
    description.textContent = value;
    row.append(term, description);
    list.appendChild(row);
  });

  const button = node.querySelector(".learn-more");
  button.dataset.id = location.id;
  const wrapper = document.createElement("div");
  wrapper.appendChild(node);
  return wrapper.innerHTML;
}

function renderMapData() {
  markerLayer.clearLayers();
  activePopupMarker = null;
  renderHeatLayer();

  // Keep the map responsive by placing exact pins only on the top 10%.
  locations
    .filter(location => location.score >= 90)
    .forEach(location => {
      const marker = L.marker([location.lat, location.lng], {
        icon: markerIcon(location.risk),
        title: location.name
      });
      marker.bindPopup(popupHTML(location), {
        closeButton: true,
        offset: [0, -1]
      });
      marker.on("popupopen", event => {
        activePopupMarker = marker;
        event.popup
          .getElement()
          ?.querySelector(".learn-more")
          ?.addEventListener("click", () => showDetails(location.id));
      });
      marker.addTo(markerLayer);
      if (selectedId === location.id) activePopupMarker = marker;
    });

  if (selectedId) showDetails(selectedId, false);
}

function conciseExplanation(location) {
  const factors = [];
  if (location.temperature >= 30) factors.push("high maximum temperature");
  if (location.humidity <= 30) factors.push("low minimum humidity");
  if (location.wind >= 30) factors.push("strong maximum wind");
  if (location.rainfall <= 0.5) factors.push("little forecast precipitation");
  if (!factors.length) factors.push("the combined weather, vegetation, and historical-fire features");
  return `Conditions supplied to the model include ${factors.slice(0, 3).join(", ")}. This summary is descriptive, not a feature-attribution analysis.`;
}

function showDetails(id, openPanel = true) {
  const location = locations.find(item => item.id === id);
  if (!location) return;
  selectedId = id;

  document.getElementById("empty-state").hidden = true;
  const detail = document.getElementById("detail-content");
  detail.hidden = false;
  detail.innerHTML = `
    <h1>${escapeHTML(location.name)}</h1>
    <p class="detail-subtitle">${escapeHTML(location.area)}</p>
    <div class="risk-banner" style="color:${riskColors[location.risk]}">
      <div>
        <strong>${escapeHTML(location.risk.toUpperCase())} — ${location.score.toFixed(1)} PERCENTILE</strong>
        <p>Next-day FIRMS hotspot-detection risk for ${escapeHTML(formatDate(predictionMetadata.prediction_date))}</p>
      </div>
      <span class="risk-symbol">▲</span>
    </div>
    <section class="detail-section">
      <h2>Forecast conditions used by the model</h2>
      <dl class="conditions-list">
        <div><dt>Maximum temperature</dt><dd>${location.temperature.toFixed(1)} °C</dd></div>
        <div><dt>Minimum humidity</dt><dd>${location.humidity.toFixed(0)}%</dd></div>
        <div><dt>Maximum wind</dt><dd>${location.wind.toFixed(1)} km/h</dd></div>
        <div><dt>Precipitation</dt><dd>${location.rainfall.toFixed(2)} mm</dd></div>
        <div><dt>Fuel model</dt><dd>${escapeHTML(location.forestType)}</dd></div>
      </dl>
    </section>
    <section class="detail-section">
      <h2>Conditions summary</h2>
      <p class="ai-explanation">${escapeHTML(conciseExplanation(location))}</p>
    </section>
    <p class="detail-note">The score is a within-day relative-risk percentile, not the probability of a wildfire. The target is a satellite FIRMS hotspot detection in this grid cell on the next calendar day.</p>
  `;

  if (openPanel && window.innerWidth < 981) {
    document.getElementById("detail-panel").scrollIntoView({ behavior: "smooth" });
  }
}

function closeDetails() {
  selectedId = null;
  document.getElementById("detail-content").hidden = true;
  document.getElementById("detail-content").innerHTML = "";
  document.getElementById("empty-state").hidden = false;
  map.closePopup();
}

function showLoadFailure(message) {
  const status = document.getElementById("prediction-status");
  status.textContent = "Predictions unavailable";
  status.classList.add("error");
  document.getElementById("forecast-date").textContent = "Unavailable";
  document.getElementById("forecast-context").textContent = message;
  document.querySelector("#empty-state h1").textContent = "Predictions unavailable";
  document.querySelector("#empty-state p").textContent =
    "Generate docs/data/predictions.json with the project inference scripts, then reload this page.";
}

async function loadPredictions() {
  try {
    const response = await fetch(PREDICTIONS_URL, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`prediction request returned ${response.status}`);
    }

    const payload = await response.json();
    if (payload.status !== "ready" || !Array.isArray(payload.locations) || !payload.locations.length) {
      throw new Error("prediction file is not ready");
    }

    const normalized = payload.locations.map(normalizeLocation).filter(isValidLocation);
    if (!normalized.length) throw new Error("prediction file contains no valid grid cells");

    predictionMetadata = payload;
    locations = normalized;
    document.getElementById("forecast-date").textContent = formatDate(payload.prediction_date);
    document.getElementById("forecast-context").textContent =
      `Features dated ${formatDate(payload.feature_date)} · ${payload.score_semantics}`;

    const status = document.getElementById("prediction-status");
    status.textContent = `${locations.length.toLocaleString()} grids analyzed`;
    status.classList.add("ready");

    const predictionDay = new Date(`${payload.prediction_date}T23:59:59`);
    if (predictionDay < new Date()) {
      status.textContent += " · stale";
      status.classList.add("stale");
    }

    renderMapData();
  } catch (error) {
    console.error("Unable to load model predictions:", error);
    const reason = window.location.protocol === "file:"
      ? "Serve the docs directory over HTTP; browser fetch requests do not work reliably from file:// pages."
      : `Could not load ${PREDICTIONS_URL}: ${error.message}`;
    showLoadFailure(reason);
  }
}

document.getElementById("zoom-in").addEventListener("click", () => map.zoomIn());
document.getElementById("zoom-out").addEventListener("click", () => map.zoomOut());
document.getElementById("home-map").addEventListener("click", () => {
  map.closePopup();
  map.setView(naturalView.center, naturalView.zoom, { animate: true });
});
document.getElementById("close-details").addEventListener("click", closeDetails);

document.querySelectorAll("[data-terrain]").forEach(button => {
  button.addEventListener("click", () => {
    const nextLayer = terrainLayers[button.dataset.terrain];
    if (!nextLayer || nextLayer === activeTerrainLayer) return;
    document.querySelectorAll("[data-terrain]").forEach(item => item.classList.remove("active"));
    button.classList.add("active");
    map.removeLayer(activeTerrainLayer);
    activeTerrainLayer = nextLayer.addTo(map);
  });
});

loadPredictions();
requestAnimationFrame(() => map.invalidateSize(true));
window.addEventListener("load", () => map.invalidateSize(true));
