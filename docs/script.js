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
let riskSurfaceLayer = null;
let pinsVisible = true;
const visiblePinRisks = new Set([
  "Medium",
  "High",
  "Extreme"
]);

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
const riskSurfacePane = map.createPane("riskSurfacePane");
riskSurfacePane.classList.add("risk-surface-pane");
riskSurfacePane.style.zIndex = "350";
riskSurfacePane.style.pointerEvents = "none";
const riskSurfaceRenderer = L.canvas({
  pane: "riskSurfacePane",
  padding: 0.5
});

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

function riskFromScore(score) {
  if (score >= 99) return "Extreme";
  if (score >= 95) return "High";
  if (score >= 90) return "Medium";
  return "Low";
}

function clusterIcon(averageScore, count) {
  const risk = riskFromScore(averageScore);
  return L.divIcon({
    className: "risk-cluster-marker",
    html: `
      <span class="cluster-shape" style="--cluster-color:${riskColors[risk]}">
        <strong>${averageScore.toFixed(1)}</strong>
        <small>${count} grids</small>
      </span>
    `,
    iconSize: [64, 64],
    iconAnchor: [32, 32]
  });
}

function clusterPixelSize(zoom) {
  if (zoom <= 5) return 110;
  if (zoom === 6) return 90;
  if (zoom === 7) return 72;
  if (zoom === 8) return 52;
  return 0;
}

function groupLocationsForZoom(items) {
  const pixelSize = clusterPixelSize(map.getZoom());
  if (!pixelSize) return items.map(location => [location]);

  const groups = new Map();
  items.forEach(location => {
    const point = map.project(
      [location.lat, location.lng],
      map.getZoom()
    );
    const key = [
      Math.floor(point.x / pixelSize),
      Math.floor(point.y / pixelSize)
    ].join(":");
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(location);
  });
  return [...groups.values()];
}

function addIndividualMarker(location) {
  const marker = L.marker([location.lat, location.lng], {
    icon: markerIcon(location.risk),
    title: location.name
  });
  marker.bindPopup(popupContent(location), {
    closeButton: true,
    offset: [0, -1]
  });
  marker.on("popupopen", () => {
    activePopupMarker = marker;
  });
  marker.addTo(markerLayer);
  if (selectedId === location.id) activePopupMarker = marker;
}

function addClusterMarker(group) {
  const averageScore = (
    group.reduce((total, location) => total + location.score, 0)
    / group.length
  );
  const center = [
    group.reduce((total, location) => total + location.lat, 0)
      / group.length,
    group.reduce((total, location) => total + location.lng, 0)
      / group.length
  ];
  const marker = L.marker(center, {
    icon: clusterIcon(averageScore, group.length),
    title: `${group.length} grids · average ${averageScore.toFixed(1)} percentile`
  });
  marker.on("click", () => {
    const bounds = L.latLngBounds(
      group.map(location => [location.lat, location.lng])
    );
    map.fitBounds(bounds.pad(0.35), {
      animate: true,
      maxZoom: Math.min(9, map.getZoom() + 2)
    });
  });
  marker.addTo(markerLayer);
}

function renderPredictionMarkers() {
  markerLayer.clearLayers();
  activePopupMarker = null;
  if (!pinsVisible) return;

  let pinLocations = locations.filter(
    location => visiblePinRisks.has(location.risk)
  );

  // At close zooms, render only the visible neighborhood so thousands of
  // DOM markers do not slow down panning while every grid remains available.
  if (map.getZoom() >= 9) {
    const visibleBounds = map.getBounds().pad(0.25);
    pinLocations = pinLocations.filter(location => (
      visibleBounds.contains([location.lat, location.lng])
    ));
  }

  groupLocationsForZoom(pinLocations).forEach(group => {
    if (group.length === 1) {
      addIndividualMarker(group[0]);
    } else {
      addClusterMarker(group);
    }
  });
}

const riskGradientStops = [
  [0, "#2f8f3a"],
  [70, "#78b83f"],
  [90, "#f2be2e"],
  [95, "#f28b24"],
  [99, "#e3483b"],
  [100, "#b7212d"]
];

function hexToRGB(hex) {
  const value = Number.parseInt(hex.slice(1), 16);
  return {
    red: (value >> 16) & 255,
    green: (value >> 8) & 255,
    blue: value & 255
  };
}

function interpolateColor(start, end, amount) {
  const first = hexToRGB(start);
  const second = hexToRGB(end);
  const channel = name => Math.round(
    first[name] + (second[name] - first[name]) * amount
  );
  return `rgb(${channel("red")}, ${channel("green")}, ${channel("blue")})`;
}

function riskSurfaceColor(score) {
  const boundedScore = Math.max(0, Math.min(100, score));
  for (let index = 1; index < riskGradientStops.length; index += 1) {
    const [upperScore, upperColor] = riskGradientStops[index];
    if (boundedScore <= upperScore) {
      const [lowerScore, lowerColor] = riskGradientStops[index - 1];
      const position = (boundedScore - lowerScore) / (upperScore - lowerScore);
      return interpolateColor(lowerColor, upperColor, position);
    }
  }
  return riskGradientStops.at(-1)[1];
}

function renderRiskSurface() {
  if (riskSurfaceLayer) map.removeLayer(riskSurfaceLayer);
  if (!locations.length) return;

  riskSurfaceLayer = L.layerGroup();

  // A geographic radius keeps adjacent 10 km grids covered at every zoom.
  // Draw low scores first so higher-risk colors remain visible on top.
  [...locations]
    .sort((first, second) => first.score - second.score)
    .forEach(location => {
      L.circle([location.lat, location.lng], {
        renderer: riskSurfaceRenderer,
        radius: 7200,
        stroke: false,
        fill: true,
        fillColor: riskSurfaceColor(location.score),
        fillOpacity: 0.62,
        interactive: false
      }).addTo(riskSurfaceLayer);
    });

  riskSurfaceLayer.addTo(map);
}

function popupContent(location) {
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
  button.addEventListener("click", event => {
    L.DomEvent.stop(event);
    showDetails(location.id);
    map.closePopup();
  });
  const wrapper = document.createElement("div");
  wrapper.appendChild(node);
  return wrapper;
}

function renderMapData() {
  renderRiskSurface();
  renderPredictionMarkers();
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
  document.getElementById("close-details").hidden = false;
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
      <h2>Grid and relative ranking</h2>
      <dl class="conditions-list">
        <div><dt>Risk tier</dt><dd>${escapeHTML(location.risk)}</dd></div>
        <div><dt>Statewide percentile</dt><dd>${location.score.toFixed(1)}</dd></div>
        <div><dt>Grid center</dt><dd>${location.lat.toFixed(4)}, ${location.lng.toFixed(4)}</dd></div>
      </dl>
      <p class="ranking-explanation">For this forecast date, the model ranked this grid at the ${location.score.toFixed(1)} percentile compared with the other analyzed California grids. This is relative risk, not wildfire probability.</p>
    </section>
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
  document.getElementById("close-details").hidden = true;
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
map.on("moveend", renderPredictionMarkers);

document.getElementById("toggle-pins").addEventListener("click", event => {
  pinsVisible = !pinsVisible;
  const button = event.currentTarget;
  const label = pinsVisible ? "Hide pins" : "Show pins";
  button.querySelector("span").textContent = label;
  button.title = `${label} for prediction grids`;
  button.setAttribute("aria-pressed", String(pinsVisible));
  button.classList.toggle("pins-hidden", !pinsVisible);
  if (!pinsVisible) map.closePopup();
  renderPredictionMarkers();
});

document.querySelectorAll("[data-pin-tier]").forEach(checkbox => {
  checkbox.addEventListener("change", event => {
    const tier = event.currentTarget.dataset.pinTier;
    if (event.currentTarget.checked) {
      visiblePinRisks.add(tier);
    } else {
      visiblePinRisks.delete(tier);
    }
    map.closePopup();
    renderPredictionMarkers();
  });
});

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
