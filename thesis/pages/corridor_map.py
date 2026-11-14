"""Spec section 5, step 2: draw the interactive corridor map of one city.

Run from ~/thesis after corridors.py build:

    bus-embeddings-corridor/thesis/.venv/bin/python -I bus-embeddings-corridor/thesis/pages/corridor_map.py GM0503 delft
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely import transform, unary_union
from shapely.geometry import mapping, shape

ROOT = Path("/home/abdulbari/thesis")
TO_LONLAT = Transformer.from_crs(28992, 4326, always_xy=True).transform


def lonlat(geom):
    """Metres to lon/lat, simplified by 1 m to keep the page small."""
    return transform(geom.simplify(1), lambda xy: np.column_stack(TO_LONLAT(xy[:, 0], xy[:, 1])))


def rounded(geom):
    """GeoJSON geometry with coordinates rounded to about 1 m."""
    def walk(c):
        return [round(c[0], 5), round(c[1], 5)] if isinstance(c[0], float) else [walk(x) for x in c]
    g = mapping(geom)
    return {"type": g["type"], "coordinates": walk(g["coordinates"])}


def main(city, name):
    folder = ROOT / f"data/corridor/geometry/{city}"
    with open(folder / "links.csv", newline="") as f:
        info = {r["id"]: r for r in csv.DictReader(f)}
    paths = {f["properties"]["id"]: shape(f["geometry"])
             for f in json.loads((folder / "paths.geojson").read_text())["features"]}
    corridors = {f["properties"]["id"]: shape(f["geometry"])
                 for f in json.loads((folder / "corridors.geojson").read_text())["features"]}
    with open(ROOT / f"data/artifacts/{city}/nodes_stop.csv", newline="") as f:
        stops = [{"id": r["stop_id"], "ll": [round(float(r["lat"]), 6), round(float(r["lon"]), 6)]}
                 for r in csv.DictReader(f)]

    links = []
    for link_id, row in info.items():
        links.append({
            "id": link_id,
            "len": float(row["path_length_m"]),
            "straight": float(row["straight_m"]),
            "ratio": float(row["detour_ratio"]) if row["detour_ratio"] else None,
            "odd": row["odd_route"] == "1",
            "area": int(row["corridor_area_m2"]),
            "path": rounded(lonlat(paths[link_id])),
            "corridor": rounded(lonlat(corridors[link_id])),
        })
    merged = rounded(lonlat(unary_union(list(corridors.values()))))
    data = {"name": name.capitalize(), "city": city, "links": links, "stops": stops, "merged": merged}

    out = ROOT / f"notes/maps/{name}_corridors.html"
    out.write_text(PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), "
          f"{len(links)} links, {sum(l['odd'] for l in links)} odd-route, {len(stops)} stops")


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Delft corridors map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
html,body{margin:0;height:100%;font:13px system-ui,sans-serif}
#map{position:absolute;top:0;bottom:0;left:0;right:0}
.box{background:#fff;padding:8px 10px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);max-width:260px;line-height:1.5}
.box h1{font-size:15px;margin:0 0 6px}
.sw{display:inline-block;width:18px;height:4px;margin-right:6px;vertical-align:middle}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin:0 11px 0 5px;vertical-align:middle}
.leaflet-popup-content table td{padding:1px 6px 1px 0}
</style></head><body><div id="map"></div>
<script>
const DATA=__DATA__;
const NORMAL="#2166ac", ODD="#d6336c", CORRIDOR="#f08c00", STOP="#212529";
const map=L.map("map",{preferCanvas:true});
// Three key-free base maps to switch between, in case one provider stops working.
const ESRI="https://server.arcgisonline.com/ArcGIS/rest/services/";
const bases={
  "Light map":L.tileLayer(ESRI+"Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",{maxZoom:20,maxNativeZoom:16,attribution:"Tiles &copy; Esri"}),
  "Street map":L.tileLayer(ESRI+"World_Street_Map/MapServer/tile/{z}/{y}/{x}",{maxZoom:20,maxNativeZoom:19,attribution:"Tiles &copy; Esri"}),
  "OpenStreetMap":L.tileLayer("https://tile.openstreetmap.de/{z}/{x}/{y}.png",{maxZoom:20,maxNativeZoom:19,
    attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'})};
bases["Street map"].addTo(map);
const renderer=L.canvas({tolerance:6});
const fmt=n=>Math.round(n).toLocaleString("en-US");
let selected=null;
function popup(l){return `<b>${l.id.replace("link_","")}</b><table>
<tr><td>Road path length</td><td><b>${fmt(l.len)} m</b></td></tr>
<tr><td>Straight line</td><td>${fmt(l.straight)} m</td></tr>
<tr><td>Detour ratio</td><td><b>${l.ratio===null?"n/a":l.ratio.toFixed(2)}</b></td></tr>
<tr><td>Odd-route link</td><td>${l.odd?"yes (ratio above 3)":"no"}</td></tr>
<tr><td>Corridor area</td><td>${(l.area/1e6).toFixed(3)} km&sup2;</td></tr></table>`}
function pick(l,e){
  if(selected)map.removeLayer(selected);
  selected=L.geoJSON(l.corridor,{interactive:false,style:{color:l.odd?ODD:NORMAL,weight:2,fillOpacity:.2}}).addTo(map);
  L.popup().setLatLng(e.latlng).setContent(popup(l)).openOn(map);
}
map.on("popupclose",()=>{if(selected){map.removeLayer(selected);selected=null}});
const line=l=>L.geoJSON(l.path,{renderer,style:{color:l.odd?ODD:NORMAL,weight:l.odd?4:2.5,opacity:.9},
  pointToLayer:(f,ll)=>L.circleMarker(ll,{radius:4,renderer})}).on("click",e=>{L.DomEvent.stopPropagation(e);pick(l,e)});
const normal=L.layerGroup(DATA.links.filter(l=>!l.odd).map(line));
const odd=L.layerGroup(DATA.links.filter(l=>l.odd).map(line));
const merged=L.geoJSON(DATA.merged,{interactive:false,style:{color:CORRIDOR,weight:1,fillColor:CORRIDOR,fillOpacity:.22}});
const outlines=L.layerGroup(DATA.links.map(l=>L.geoJSON(l.corridor,{interactive:false,style:{color:l.odd?ODD:CORRIDOR,weight:1,fill:false,opacity:.6}})));
const stops=L.layerGroup(DATA.stops.map(s=>L.circleMarker(s.ll,{renderer,radius:3,color:STOP,weight:1,fillColor:"#fff",fillOpacity:1}).bindPopup(`<b>Stop</b> ${s.id}`)));
[merged,normal,odd,stops].forEach(g=>g.addTo(map));
L.control.layers(bases,{"100 m corridors (merged area)":merged,"Corridor outline of each link":outlines,"Road paths":normal,"Odd-route links":odd,"Stops":stops},{collapsed:false,position:"topright"}).addTo(map);
map.fitBounds(L.geoJSON(DATA.merged).getBounds());
const info=L.control({position:"bottomleft"});
info.onAdd=()=>{const d=L.DomUtil.create("div","box");const n=DATA.links.filter(l=>l.odd).length;
d.innerHTML=`<h1>${DATA.name} (${DATA.city})</h1>${DATA.links.length} links &middot; ${n} odd-route &middot; ${DATA.stops.length} stops<br>
<span class="sw" style="background:${NORMAL}"></span>Road path<br>
<span class="sw" style="background:${ODD}"></span>Odd-route link (detour ratio above 3)<br>
<span class="sw" style="background:${CORRIDOR};height:10px;opacity:.5"></span>100 m each side of the path<br>
<span class="dot" style="background:#fff;border:1px solid ${STOP}"></span>Stop<br>
Click a road path for its numbers.`;return d};
info.addTo(map);
</script></body></html>
"""

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
