# RedZone interface demo

A standalone front-end mockup for a California wildfire-risk prediction project.

## Run it

Because the map uses external browser resources, serve the folder locally rather than opening the HTML file directly:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Current behavior

- California-centered map with a limited surrounding buffer
- Sample Northern California forest prediction points
- Risk sections and map pins change from Now through 72 hours
- Point popup and right-side details panel
- Zoom, home, and terrain-detail controls
- Laptop-first responsive layout

## Connecting the real model

Replace the `locations` sample array in `script.js` with API data. Keep each item shaped like this:

```js
{
  id: "unique-id",
  name: "Forest name",
  area: "Region",
  lat: 40.7,
  lng: -122.6,
  forestType: "Mixed conifer forest",
  base: {
    temperature: 82,
    humidity: 37,
    wind: 8,
    rainfall: 0.05,
    dryness: 62
  },
  trend: [56, 62, 70, 73, 66, 58]
}
```

The six `trend` values correspond to Now, 6h, 12h, 24h, 48h, and 72h.
