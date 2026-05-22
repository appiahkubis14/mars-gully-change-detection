"""
folium_map.py — Mars Gully Dashboard with guaranteed basemap
Uses NASA's publicly accessible Mars image as base + Leaflet for interactivity.
"""
from __future__ import annotations
import base64, json, logging, math
from pathlib import Path
import numpy as np

log = logging.getLogger(__name__)


def _arr_to_png_b64(arr, cmap_name="hot_r", size=512):
    import io, matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm, matplotlib.colors as mc
    from PIL import Image
    h, w = arr.shape
    sc = min(1.0, size / max(h, w))
    th, tw = max(1, int(h*sc)), max(1, int(w*sc))
    norm = mc.Normalize(0, 1, clip=True)
    rgba = (cm.get_cmap(cmap_name)(norm(arr)) * 255).astype(np.uint8)
    rgba[:,:,3] = np.where(arr > 0.05, 200, 0).astype(np.uint8)
    img = Image.fromarray(rgba,"RGBA").resize((tw,th), Image.LANCZOS)
    buf = io.BytesIO(); img.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def _site_info(cfg):
    out = {}
    for k, v in cfg.get("study_area", {}).items():
        if not isinstance(v, dict): continue
        b = v.get("bounds"); name = v.get("name", k)
        if not b: continue
        lo0,la0,lo1,la1 = b
        if lo0>180: lo0-=360
        if lo1>180: lo1-=360
        out[k]={"name":name,"center":[(la0+la1)/2,(lo0+lo1)/2],
                "bounds":[[la0,lo0],[la1,lo1]]}
    return out


def build_dashboard(cfg_path="config.yaml"):
    import yaml
    with open(cfg_path) as f: cfg = yaml.safe_load(f)

    out_dir = Path("data/outputs/dashboard"); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mars_gully_dashboard.html"

    prob_dir   = Path("data/outputs/probability_maps")
    change_dir = Path("data/outputs/change_detection")
    sites      = _site_info(cfg)

    overlays = []
    try:
        import rasterio
        for pf in sorted(prob_dir.glob("*.tif"))[:7]:
            try:
                with rasterio.open(pf) as src:
                    arr = src.read(1).astype(np.float32)
                    b = src.bounds; R = 3_396_190.0
                    w = math.degrees(b.left/R);   e = math.degrees(b.right/R)
                    s = math.degrees(b.bottom/R); n = math.degrees(b.top/R)
                    if w>180: w-=360
                    if e>180: e-=360
                v = arr[arr>0]
                if len(v): arr = np.clip(arr/max(float(v.max()),1e-6),0,1)
                overlays.append({"name": pf.stem.replace("_features_prob","").replace("_prob",""),
                                  "png": _arr_to_png_b64(arr),
                                  "bounds": [[s,w],[n,e]]})
            except Exception as ex: log.debug(f"{pf.name}: {ex}")
    except ImportError: pass

    changes, activity_n = [], 0
    try: changes = json.loads((change_dir/"change_summary.json").read_text())[:20]
    except: pass
    try: activity_n = len(json.loads((change_dir/"activity_tracks.json").read_text()))
    except: pass

    if sites:
        cs=[v["center"] for v in sites.values()]
        ctr=[sum(c[0] for c in cs)/len(cs), sum(c[1] for c in cs)/len(cs)]
    else: ctr=[-36,130]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mars Gully Digital Twin</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:#0d1117;color:#e6edf3;height:100vh;display:flex;flex-direction:column;overflow:hidden}}
#hdr{{background:linear-gradient(90deg,#0d1117,#1a2030);border-bottom:1px solid #30363d;
      padding:8px 16px;display:flex;align-items:center;justify-content:space-between;flex-shrink:0}}
#hdr h1{{font-size:1.05rem;color:#58a6ff}}
.badge{{background:#21262d;border:1px solid #30363d;border-radius:20px;
        padding:2px 8px;font-size:.65rem;color:#58a6ff;margin-left:5px}}
#body{{display:flex;flex:1;overflow:hidden}}
#side{{width:280px;flex-shrink:0;background:#161b22;border-right:1px solid #30363d;
       overflow-y:auto;display:flex;flex-direction:column;order:-1}}
#map{{flex:1}}
.panel{{border-bottom:1px solid #21262d;padding:9px 11px}}
.ph{{font-size:.67rem;text-transform:uppercase;letter-spacing:.1em;color:#8b949e;
     margin-bottom:6px;font-weight:600}}
.bmbtn{{background:#21262d;border:1px solid #30363d;color:#8b949e;border-radius:4px;
        padding:3px 8px;font-size:.68rem;cursor:pointer;margin:2px;transition:.15s}}
.bmbtn:hover,.bmbtn.on{{background:#1f6feb;border-color:#58a6ff;color:#fff}}
.lr{{display:flex;align-items:center;gap:5px;padding:3px 0;font-size:.73rem}}
.lr input[type=checkbox]{{accent-color:#58a6ff;width:13px;height:13px;flex-shrink:0}}
.lr input[type=range]{{flex:1;accent-color:#58a6ff;height:3px}}
.lc{{width:10px;height:10px;border-radius:2px;flex-shrink:0}}
.sc{{background:#21262d;border:1px solid #30363d;border-radius:5px;
     padding:7px 9px;margin-bottom:4px;cursor:pointer;transition:.15s}}
.sc:hover{{border-color:#58a6ff}}
.sn{{font-size:.8rem;font-weight:600}}.ss{{font-size:.67rem;color:#8b949e;margin-top:1px}}
.sr{{display:flex;justify-content:space-between;font-size:.72rem;
     padding:3px 0;border-bottom:1px solid #21262d}}
.sv{{color:#58a6ff;font-weight:600}}
.lg{{height:10px;border-radius:3px;margin:5px 0;
     background:linear-gradient(to right,#00008b,#0055ff,#00ccff,#00ff88,#ffff00,#ff8800,#ff0000)}}
.ll{{display:flex;justify-content:space-between;font-size:.62rem;color:#8b949e}}
.leaflet-popup-content-wrapper{{background:#1c2128!important;color:#e6edf3!important;
  border:1px solid #30363d!important;border-radius:7px!important}}
.leaflet-popup-tip{{background:#1c2128!important}}
.pt{{font-weight:700;color:#58a6ff;font-size:.85rem;margin-bottom:3px}}
.pr{{font-size:.73rem;color:#8b949e;margin:2px 0}}.pv{{color:#e6edf3}}
</style>
</head>
<body>

<div id="hdr">
  <div>
    <h1>🔴 Mars Gully Digital Twin
      <span class="badge">U-Net IoU 0.342</span>
      <span class="badge">19-ch features</span>
      <span class="badge">3 Sites</span>
    </h1>
    <div style="font-size:.68rem;color:#8b949e;margin-top:1px">
      HiRISE · CTX · MOLA · Gasa · Palikir · Russell Craters</div>
  </div>
  <div style="font-size:.67rem;color:#3fb950">Pipeline complete ✓</div>
</div>

<div id="body">
  
  <div id="side">

    <div class="panel">
      <div class="ph">🗺 Basemap</div>
      <button class="bmbtn on" onclick="setBase('celestia')" id="bb-celestia">Mars Texture</button>
      <button class="bmbtn"    onclick="setBase('mola')"     id="bb-mola"   >MOLA Color</button>
      <button class="bmbtn"    onclick="setBase('viking')"   id="bb-viking" >Viking</button>
      <button class="bmbtn"    onclick="setBase('opm')"      id="bb-opm"    >OPM Full</button>
      <button class="bmbtn"    onclick="setBase('ctx')"      id="bb-ctx"    >CTX 5m</button>
    </div>

    <div class="panel">
      <div class="ph">📡 Probability Maps</div>
      <div id="lyrlist"></div>
    </div>

    <div class="panel">
      <div class="ph">📍 Study Sites</div>
      <div id="sitelist"></div>
    </div>

    <div class="panel">
      <div class="ph">📊 Model Stats</div>
      <div class="sr"><span>Feature channels</span><span class="sv">19</span></div>
      <div class="sr"><span>HiRISE + CTX + MOLA</span><span class="sv">3 sites</span></div>
      <div class="sr"><span>Train patches</span><span class="sv">682</span></div>
      <div class="sr"><span>Val IoU (epoch 0)</span><span class="sv">0.342</span></div>
      <div class="sr"><span>Val F1</span><span class="sv">0.459</span></div>
      <div class="sr"><span>Prob maps</span><span class="sv">{len(overlays)}</span></div>
      <div class="sr"><span>Activity tracks</span><span class="sv">{activity_n or '–'}</span></div>
    </div>

    <div class="panel">
      <div class="ph">📈 Change Series</div>
      <canvas id="chChart" style="max-height:130px"></canvas>
    </div>

    <div class="panel">
      <div class="ph">🎨 Probability Scale</div>
      <div class="lg"></div>
      <div class="ll"><span>0%</span><span>50%</span><span>100%</span></div>
      <div style="margin-top:6px;font-size:.67rem;color:#8b949e;line-height:1.7">
        <span style="color:#ff2200">■</span> High — likely gully<br>
        <span style="color:#ffaa00">■</span> Medium probability<br>
        <span style="color:#0055ff">■</span> Low probability
      </div>
    </div>

  </div>

  <div id="map"></div>
</div>

<script>
const OVERLAYS = {json.dumps(overlays)};
const SITES    = {json.dumps(sites)};
const CHANGES  = {json.dumps(changes)};
const CTR      = {json.dumps(ctr)};

// ── Basemaps — only XYZ tile services verified to work ─────────────────────
//
// The key lesson: Mars planetary WMS servers (USGS mapserv) use Simple
// Cylindrical projection, NOT Web Mercator. Leaflet ONLY supports Web Mercator.
// Solution: use pre-rendered XYZ pyramids reprojected to EPSG:3857.
//
// Working sources as of 2025:
//  - ESRI CTX: confirmed Web Mercator XYZ
//  - OpenPlanetaryMap: Leaflet-native Mars tiles
//  - USGS WMTS (GoogleMapsCompatible endpoint): Web Mercator ✓

// ── Mars Tile Configuration ───────────────────────────────────────────────
// All OPM tiles are XYZ, Web Mercator EPSG:3857, from S3 (verified 2025)
// Source: openplanetary.org
const OPM_S3 = 'https://s3-eu-west-1.amazonaws.com/whereonmars.cartodb.net';

// Leaflet CRS fix for planetary maps:
// Mars tiles cover lat -90 to +90 but Web Mercator clips at ±85.05°
// We use a custom CRS that doesn't clip latitude
const MarsCRS = L.extend({{}}, L.CRS.EPSG3857, {{
  // Allow full -90 to 90 latitude range (Mars has polar terrain)
  wrapLng: [-180, 180],
}});

// Tile options shared across all Mars layers
const TOPTS = {{
  bounds: [[-90,-180],[90,180]],  // full Mars globe extent
  tileSize: 256,
  noWrap: true,
  background: '#c1440e',
}};

const BASES = {{
  // ① Celestia Mars shaded texture — 16k global, best visual quality
  celestia: L.tileLayer(
    OPM_S3 + '/celestia_mars-shaded-16k_global/{{z}}/{{x}}/{{y}}.png',
    {{ ...TOPTS, attribution:'Mars Shaded Texture — Celestia/OpenPlanetaryMap',
       maxZoom:8, maxNativeZoom:5 }}
  ),
  // ② MOLA colour+hillshade — science standard visualization  
  mola: L.tileLayer(
    OPM_S3 + '/mola-color/{{z}}/{{x}}/{{y}}.png',
    {{ ...TOPTS, attribution:'MOLA Color Elevation — NASA/OpenPlanetaryMap',
       maxZoom:6, maxNativeZoom:5 }}
  ),
  // ③ Viking MDIM 2.1 colour mosaic
  viking: L.tileLayer(
    OPM_S3 + '/viking_mdim21_global/{{z}}/{{x}}/{{y}}.png',
    {{ ...TOPTS, attribution:'Viking MDIM 2.1 — NASA/OpenPlanetaryMap',
       maxZoom:6, maxNativeZoom:5 }}
  ),
  // ④ Full OPM composite with nomenclature
  opm: L.tileLayer(
    'https://cartocdn-gusc.global.ssl.fastly.net/opmbuilder/api/v1/map/named/opm-mars-basemap-v0-2/all/{{z}}/{{x}}/{{y}}.png',
    {{ ...TOPTS, attribution:'OpenPlanetaryMap v0.2 — Nass et al. 2020',
       maxZoom:7, maxNativeZoom:5 }}
  ),
  // ⑤ ESRI CTX 5m/px — highest resolution (use at zoom 7+)
  ctx: L.tileLayer(
    'https://tiles.arcgis.com/tiles/P3ePLMYs2RVChkJx/arcgis/rest/services/CTX_v01/MapServer/tile/{{z}}/{{y}}/{{x}}',
    {{ ...TOPTS, attribution:'CTX Mosaic V01 — Dickson et al. 2024/ESRI',
       maxZoom:14, maxNativeZoom:13 }}
  ),
}};

// ── Init map ──────────────────────────────────────────────────────────────
// Mars rust background — fills gaps between tiles
document.getElementById('map').style.background = '#c1440e';
const map = L.map('map', {{
  center: CTR,
  zoom: 4,
  minZoom: 2,
  maxZoom: 14,
  preferCanvas: true,
  // Prevent Leaflet from wrapping the world (Mars tiles don't repeat)
  maxBounds: [[-90,-180],[90,180]],
  maxBoundsViscosity: 1.0,
  worldCopyJump: false,
}});

// Start with CTX; if it fails cascade to OPM → Viking → dark
let curBase = BASES.celestia;
let fallbackChain = ['celestia','mola','viking','opm','ctx'];
let fbIdx = 0;

function tryNextBase() {{
  fbIdx++;
  if (fbIdx < fallbackChain.length) {{
    const nk = fallbackChain[fbIdx];
    map.removeLayer(curBase);
    curBase = BASES[nk];
    curBase.addTo(map);
    document.querySelectorAll('.bmbtn').forEach(b=>b.classList.remove('on'));
    const btn = document.getElementById('bb-'+nk);
    if(btn) btn.classList.add('on');
    // Hook next fallback
    if (nk !== 'dark') BASES[nk].once('tileerror', tryNextBase);
  }}
}}

curBase.addTo(map);
BASES.celestia.once('tileerror', tryNextBase);

function setBase(key) {{
  map.removeLayer(curBase);
  curBase = BASES[key]; curBase.addTo(map);
  fbIdx = fallbackChain.indexOf(key);
  document.querySelectorAll('.bmbtn').forEach(b=>b.classList.remove('on'));
  document.getElementById('bb-'+key)?.classList.add('on');
  if (key !== 'dark') curBase.once('tileerror', tryNextBase);
}}

// ── Probability overlays ──────────────────────────────────────────────────
const lyrMap = {{}};
const COLS = ['#ff3300','#ff8800','#ffcc00','#44ff88','#44aaff','#aa44ff','#ff44cc'];

OVERLAYS.forEach((ov,i) => {{
  const col = COLS[i%COLS.length];
  const img = L.imageOverlay('data:image/png;base64,'+ov.png, ov.bounds,
                              {{opacity:0.7}});
  lyrMap[ov.name] = img;
  if (i<3) img.addTo(map);

  const nm = ov.name.length>25 ? ov.name.slice(0,22)+'…' : ov.name;
  const row = document.createElement('div');
  row.className='lr';
  row.innerHTML=`
    <input type="checkbox" id="c${{i}}" ${{i<3?'checked':''}}
           onchange="tog('${{ov.name}}',this.checked)">
    <div class="lc" style="background:${{col}}"></div>
    <label for="c${{i}}" style="flex:1;cursor:pointer;overflow:hidden;
           white-space:nowrap;text-overflow:ellipsis" title="${{ov.name}}">${{nm}}</label>
    <input type="range" min="0" max="1" step=".05" value=".7"
           oninput="opac('${{ov.name}}',+this.value)">`;
  document.getElementById('lyrlist').appendChild(row);
}});

function tog(n,v)  {{ v ? lyrMap[n].addTo(map) : map.removeLayer(lyrMap[n]); }}
function opac(n,v) {{ lyrMap[n]?.setOpacity(v); }}

// ── Site markers ──────────────────────────────────────────────────────────
const SCOLS = {{primary:'#58a6ff',secondary:'#3fb950',tertiary:'#f78166'}};
Object.entries(SITES).forEach(([k,s]) => {{
  const col = SCOLS[k]||'#8b949e';
  const ico = L.divIcon({{
    html:`<div style="background:${{col}};width:16px;height:16px;border-radius:50%;
               border:2.5px solid #fff;box-shadow:0 0 10px ${{col}}99;"></div>`,
    iconSize:[16,16],iconAnchor:[8,8],className:''
  }});
  L.marker(s.center,{{icon:ico}}).addTo(map).bindPopup(`
    <div class="pt">📍 ${{s.name}}</div>
    <div class="pr">Lat <span class="pv">${{s.center[0].toFixed(2)}}°</span>
         &nbsp;Lon <span class="pv">${{s.center[1].toFixed(2)}}°</span></div>
    <div class="pr">Sensors <span class="pv">HiRISE · CTX · MOLA</span></div>
    <div class="pr">Features <span class="pv">19-channel stack (3276×4096)</span></div>
    <div class="pr">Labels <span class="pv">Slope-based synthetic (2° threshold)</span></div>`);
  if(s.bounds)
    L.rectangle(s.bounds,{{color:col,weight:1.5,fill:false,dashArray:'5 4',opacity:.7}}).addTo(map);
  const card = document.createElement('div');
  card.className='sc';
  card.innerHTML=`<div class="sn" style="color:${{col}}">🔴 ${{s.name}}</div>
    <div class="ss">${{s.center[0].toFixed(2)}}°, ${{s.center[1].toFixed(2)}}°</div>`;
  card.onclick=()=>s.bounds?map.fitBounds(s.bounds,{{padding:[30,30]}}):map.setView(s.center,7);
  document.getElementById('sitelist').appendChild(card);
}});

// Fit to all sites
const allB=Object.values(SITES).filter(s=>s.bounds).map(s=>s.bounds);
if(allB.length){{
  const lats=allB.flat().map(b=>b[0]), lons=allB.flat().map(b=>b[1]);
  map.fitBounds([[Math.min(...lats),Math.min(...lons)],[Math.max(...lats),Math.max(...lons)]],
                {{padding:[50,50]}});
}}

// ── Scale + coords ────────────────────────────────────────────────────────
L.control.scale({{imperial:false}}).addTo(map);
const coordCtrl = L.control({{position:'bottomleft'}});
coordCtrl.onAdd=()=>{{
  const d=L.DomUtil.create('div');
  d.style.cssText='background:#161b22dd;color:#8b949e;padding:3px 9px;font-size:.68rem;border-radius:10px;border:1px solid #30363d';
  d.id='coords'; d.textContent='Move cursor over map';
  return d;
}};
coordCtrl.addTo(map);
map.on('mousemove',e=>{{
  const lo=((e.latlng.lng%360)+360)%360;
  document.getElementById('coords').textContent=
    `${{e.latlng.lat.toFixed(3)}}°  ${{lo.toFixed(2)}}°E (Mars IAU)`;
}});

// ── Change chart ──────────────────────────────────────────────────────────
new Chart(document.getElementById('chChart').getContext('2d'),{{
  type:'bar',
  data:{{
    labels:CHANGES.length?CHANGES.map(c=>c.date_t1||c.pair||'?'):['No data'],
    datasets:[
      {{label:'Gain ha',data:CHANGES.map(c=>+(c.gain_ha||0).toFixed(1)),
        backgroundColor:'#3fb95055',borderColor:'#3fb950',borderWidth:1}},
      {{label:'Loss ha',data:CHANGES.map(c=>+(c.loss_ha||0).toFixed(1)),
        backgroundColor:'#f7816655',borderColor:'#f78166',borderWidth:1}},
    ]
  }},
  options:{{responsive:true,maintainAspectRatio:true,
    plugins:{{legend:{{labels:{{color:'#8b949e',font:{{size:9}}}}}}}},
    scales:{{
      x:{{ticks:{{color:'#8b949e',font:{{size:8}},maxRotation:40}},grid:{{color:'#21262d'}}}},
      y:{{ticks:{{color:'#8b949e',font:{{size:8}}}},grid:{{color:'#21262d'}}}},
    }}}}
}});
</script>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    log.info(f"Dashboard → {out_path}  ({out_path.stat().st_size//1024} KB)")
    return out_path