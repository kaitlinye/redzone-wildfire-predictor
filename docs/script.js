"use strict";

const timeLabels = ["Now", "6 hours", "12 hours", "24 hours", "48 hours", "72 hours"];
const timeShort = ["Now", "6h", "12h", "24h", "48h", "72h"];
const riskColors = { Low: "#5ca53d", Medium: "#f2be2e", High: "#f28b24", Extreme: "#e3483b" };

const locations = [
  {
    id: "shasta",
    name: "Shasta–Trinity National Forest",
    area: "Northern California",
    lat: 40.72, lng: -122.63,
    forestType: "Mixed conifer forest",
    base: { temperature: 82, humidity: 37, wind: 8, rainfall: 0.05, dryness: 62 },
    trend: [56, 62, 70, 73, 66, 58]
  },
  {
    id: "klamath",
    name: "Klamath National Forest",
    area: "Far Northern California",
    lat: 41.61, lng: -123.08,
    forestType: "Mixed evergreen forest",
    base: { temperature: 77, humidity: 46, wind: 5, rainfall: 0.12, dryness: 48 },
    trend: [38, 42, 48, 54, 49, 45]
  },
  {
    id: "modoc",
    name: "Modoc National Forest",
    area: "Northeastern California",
    lat: 41.53, lng: -120.85,
    forestType: "Pine and juniper woodland",
    base: { temperature: 84, humidity: 28, wind: 12, rainfall: 0, dryness: 72 },
    trend: [68, 74, 81, 85, 78, 72]
  },
  {
    id: "mendocino",
    name: "Mendocino National Forest",
    area: "Northern Coast Range",
    lat: 39.56, lng: -122.95,
    forestType: "Mixed oak and conifer forest",
    base: { temperature: 80, humidity: 42, wind: 7, rainfall: 0.08, dryness: 55 },
    trend: [46, 51, 59, 63, 57, 50]
  },
  {
    id: "lassen",
    name: "Lassen National Forest",
    area: "Northern Sierra and Cascades",
    lat: 40.18, lng: -121.15,
    forestType: "Pine and fir forest",
    base: { temperature: 79, humidity: 34, wind: 10, rainfall: 0.02, dryness: 66 },
    trend: [59, 65, 72, 77, 69, 63]
  },
  {
    id: "plumas",
    name: "Plumas National Forest",
    area: "Northern Sierra Nevada",
    lat: 39.92, lng: -120.88,
    forestType: "Mixed conifer forest",
    base: { temperature: 78, humidity: 39, wind: 6, rainfall: 0.09, dryness: 52 },
    trend: [43, 48, 56, 61, 55, 49]
  },
  {
    id: "tahoe",
    name: "Tahoe National Forest",
    area: "Northern Sierra Nevada",
    lat: 39.36, lng: -120.62,
    forestType: "Pine and fir forest",
    base: { temperature: 75, humidity: 44, wind: 8, rainfall: 0.15, dryness: 44 },
    trend: [34, 39, 47, 53, 48, 43]
  },
  {
    id: "six-rivers",
    name: "Six Rivers National Forest",
    area: "North Coast",
    lat: 41.05, lng: -123.85,
    forestType: "Coastal conifer forest",
    base: { temperature: 71, humidity: 63, wind: 4, rainfall: 0.22, dryness: 31 },
    trend: [22, 26, 31, 36, 32, 29]
  }
];

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

// A broad gray context mask and a highlighted California analysis footprint.
L.rectangle(californiaBounds, {
  color: "transparent",
  fillColor: "#707579",
  fillOpacity: 0.22,
  interactive: false
}).addTo(map);

const analysisOutline = L.polygon([
  [42.0,-124.35],[42.0,-120.0],[39.0,-120.0],[35.0,-114.64],[32.53,-117.12],[34.0,-120.45],[38.0,-123.0],[41.0,-124.25]
], {
  color: "#f3eee5",
  weight: 2,
  fillColor: "#ffffff",
  fillOpacity: 0.03,
  interactive: false
}).addTo(map);

const markerLayer = L.layerGroup().addTo(map);
let heatLayer = null;
let selectedId = null;
let currentTimeIndex = 0;
let activePopupMarker = null;

function getRisk(score) {
  if (score < 35) return "Low";
  if (score < 60) return "Medium";
  if (score < 80) return "High";
  return "Extreme";
}

function adjustedConditions(location, timeIndex) {
  const score = location.trend[timeIndex];
  return {
    score,
    risk: getRisk(score),
    temperature: Math.round(location.base.temperature + [0,2,4,5,1,-2][timeIndex]),
    humidity: Math.max(15, Math.round(location.base.humidity + [0,-3,-7,-9,-4,1][timeIndex])),
    wind: Math.max(0, Math.round(location.base.wind + [0,2,4,6,3,1][timeIndex])),
    rainfall: Math.max(0, +(location.base.rainfall + [0,0,0,.02,.08,.15][timeIndex]).toFixed(2)),
    dryness: Math.min(100, Math.round(location.base.dryness + [0,3,7,10,5,1][timeIndex]))
  };
}

function markerIcon(risk) {
  return L.divIcon({
    className: "risk-marker",
    html: `<span class="pin-shape" style="--pin:${riskColors[risk]}"></span>`,
    iconSize: [24,31],
    iconAnchor: [12,30],
    popupAnchor: [0,-29]
  });
}

function buildHeatPoints() {
  const points = [];

  locations.forEach(location => {
    const conditions = adjustedConditions(location, currentTimeIndex);
    const intensity = Math.max(0.18, conditions.score / 100);

    // A central reading plus nearby support points creates a continuous,
    // flowing surface instead of separate polygon or hexagon-shaped regions.
    points.push([location.lat, location.lng, intensity]);

    const spread = 0.18 + (conditions.score / 100) * 0.12;
    const support = [
      [ spread, 0], [-spread, 0], [0, spread], [0, -spread],
      [ spread * 0.7, spread * 0.7],
      [ spread * 0.7, -spread * 0.7],
      [-spread * 0.7, spread * 0.7],
      [-spread * 0.7, -spread * 0.7]
    ];

    support.forEach(([latOffset, lngOffset]) => {
      points.push([
        location.lat + latOffset,
        location.lng + lngOffset,
        intensity * 0.72
      ]);
    });
  });

  return points;
}

function renderHeatLayer() {
  if (heatLayer) map.removeLayer(heatLayer);

  heatLayer = L.heatLayer(buildHeatPoints(), {
    radius: 52,
    blur: 42,
    maxZoom: 10,
    minOpacity: 0.28,
    max: 1,
    gradient: {
      0.18: "#5ca53d",
      0.42: "#b6c83e",
      0.58: "#f2be2e",
      0.76: "#f28b24",
      1.0: "#e3483b"
    }
  }).addTo(map);

  // Keep exact clickable prediction pins above the blended heat surface.
  heatLayer.bringToBack();
}

function popupHTML(location, conditions) {
  const node = document.getElementById("popup-template").content.cloneNode(true);
  node.querySelector(".popup-risk").textContent = `${conditions.risk.toUpperCase()} RISK — ${conditions.score}/100`;
  node.querySelector(".popup-risk").style.color = riskColors[conditions.risk];
  node.querySelector(".popup-pin").style.color = riskColors[conditions.risk];
  node.querySelector(".popup-title").textContent = location.name;
  const rows = [
    ["Temperature", `${conditions.temperature}°F`],
    ["Humidity", `${conditions.humidity}%`],
    ["Wind", `${conditions.wind} mph`],
    ["Rainfall", `${conditions.rainfall.toFixed(2)} in`],
    ["Forest type", location.forestType]
  ];
  const dl = node.querySelector(".popup-stats");
  rows.forEach(([label,value]) => {
    const row = document.createElement("div");
    row.innerHTML = `<dt>${label}</dt><dd>${value}</dd>`;
    dl.appendChild(row);
  });
  node.querySelector(".learn-more").dataset.id = location.id;
  const wrapper = document.createElement("div");
  wrapper.appendChild(node);
  return wrapper.innerHTML;
}

function renderMapData() {
  markerLayer.clearLayers();
  activePopupMarker = null;
  renderHeatLayer();

  locations.forEach(location => {
    const conditions = adjustedConditions(location, currentTimeIndex);
    const marker = L.marker([location.lat, location.lng], { icon: markerIcon(conditions.risk), title: location.name });
    marker.bindPopup(popupHTML(location, conditions), { closeButton: true, offset: [0,-1] });
    marker.on("popupopen", event => {
      activePopupMarker = marker;
      const popupElement = event.popup.getElement();
      popupElement?.querySelector(".learn-more")?.addEventListener("click", () => showDetails(location.id));
    });
    marker.addTo(markerLayer);
    if (selectedId === location.id) activePopupMarker = marker;
  });
  if (selectedId) showDetails(selectedId, false);
}

function conciseExplanation(location, conditions) {
  const factors = [];
  if (conditions.temperature >= 82) factors.push("high temperature");
  if (conditions.humidity <= 35) factors.push("low humidity");
  if (conditions.wind >= 12) factors.push("strong wind");
  if (conditions.rainfall <= .05) factors.push("little recent rainfall");
  if (conditions.dryness >= 65) factors.push("dry vegetation");
  if (!factors.length) factors.push("milder weather and less-dry vegetation");
  const lead = conditions.risk === "Low" ? "Risk stays low because of" : `${conditions.risk} risk is mainly driven by`;
  return `${lead} ${factors.slice(0,3).join(", ")}.`;
}

function showDetails(id, openPanel = true) {
  const location = locations.find(item => item.id === id);
  if (!location) return;
  selectedId = id;
  const c = adjustedConditions(location, currentTimeIndex);
  document.getElementById("empty-state").hidden = true;
  const detail = document.getElementById("detail-content");
  detail.hidden = false;
  const horizon = currentTimeIndex === 0 ? "Current prediction" : `Prediction for ${timeLabels[currentTimeIndex]} ahead`;
  detail.innerHTML = `
    <h1>${location.name}</h1>
    <p class="detail-subtitle">${location.area}</p>
    <div class="risk-banner" style="color:${riskColors[c.risk]}">
      <div><strong>${c.risk.toUpperCase()} RISK — ${c.score}/100</strong><p>${horizon}</p></div>
      <span class="risk-symbol">▲</span>
    </div>
    <section class="detail-section">
      <h2>Conditions</h2>
      <dl class="conditions-list">
        <div><dt>Temperature</dt><dd>${c.temperature}°F</dd></div>
        <div><dt>Humidity</dt><dd>${c.humidity}%</dd></div>
        <div><dt>Wind</dt><dd>${c.wind} mph</dd></div>
        <div><dt>Rainfall</dt><dd>${c.rainfall.toFixed(2)} in</dd></div>
        <div><dt>Forest type</dt><dd>${location.forestType}</dd></div>
      </dl>
    </section>
    <section class="detail-section">
      <h2>Prediction explanation</h2>
      <p class="ai-explanation">${conciseExplanation(location, c)}</p>
    </section>
    <p class="detail-note">Demo interface using sample predictions. Connect this view to the trained model API when it is ready.</p>
  `;
  if (openPanel && window.innerWidth < 981) document.getElementById("detail-panel").scrollIntoView({ behavior: "smooth" });
}

function closeDetails() {
  selectedId = null;
  document.getElementById("detail-content").hidden = true;
  document.getElementById("detail-content").innerHTML = "";
  document.getElementById("empty-state").hidden = false;
  map.closePopup();
}

document.getElementById("time-slider").addEventListener("input", event => {
  currentTimeIndex = Number(event.target.value);
  document.getElementById("time-value").textContent = timeShort[currentTimeIndex];
  renderMapData();
});

document.getElementById("zoom-in").addEventListener("click", () => map.zoomIn());
document.getElementById("zoom-out").addEventListener("click", () => map.zoomOut());
document.getElementById("home-map").addEventListener("click", () => {
  map.closePopup();
  map.setView(naturalView.center, naturalView.zoom, { animate: true });
});
document.getElementById("close-details").addEventListener("click", closeDetails);

document.querySelectorAll("[data-terrain]").forEach(button => {
  button.addEventListener("click", () => {
    const terrainName = button.dataset.terrain;
    const nextLayer = terrainLayers[terrainName];
    if (!nextLayer || nextLayer === activeTerrainLayer) return;

    document.querySelectorAll("[data-terrain]").forEach(item => item.classList.remove("active"));
    button.classList.add("active");
    map.removeLayer(activeTerrainLayer);
    activeTerrainLayer = nextLayer.addTo(map);
  });
});

renderMapData();
requestAnimationFrame(() => {
  map.invalidateSize(true);
  map.setView(naturalView.center, naturalView.zoom, { animate: false });
});
window.addEventListener("load", () => {
  map.invalidateSize(true);
  map.setView(naturalView.center, naturalView.zoom, { animate: false });
});
