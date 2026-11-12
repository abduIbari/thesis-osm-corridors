"""Spec section 5, step 2 level 1: interactive map of one city's level 1 values.

Run from ~/thesis after level1.py build:

    bus-embeddings-corridor/thesis/.venv/bin/python -I bus-embeddings-corridor/thesis/pages/level1_map.py GM0503 delft
"""

import csv
import json
import sys
from pathlib import Path

import shapely
from shapely.geometry import shape

sys.path.insert(0, str(Path(__file__).parent))
from corridor_map import ROOT, TO_LONLAT, lonlat, rounded  # noqa: E402


def main(city, name):
    geometry = ROOT / f"data/corridor/geometry/{city}"
    features = ROOT / f"data/corridor/features/{city}"
    with open(geometry / "links.csv", newline="") as f:
        lengths = {r["id"]: float(r["path_length_m"]) for r in csv.DictReader(f)}
    paths = {f["properties"]["id"]: shape(f["geometry"])
             for f in json.loads((geometry / "paths.geojson").read_text())["features"]}
    corridors = {f["properties"]["id"]: shape(f["geometry"])
                 for f in json.loads((geometry / "corridors.geojson").read_text())["features"]}
    with open(features / "level1.csv", newline="") as f:
        reader = csv.reader(f)
        columns = next(reader)[1:]
        values = {row[0]: [round(float(v), 4) for v in row[1:]] for row in reader}
    assert set(values) == set(paths), "level1.csv and the corridors must hold the same links"

    links = [{
        "id": link_id,
        "len": lengths[link_id],
        "v": values[link_id],
        "path": rounded(lonlat(paths[link_id])),
        "corridor": rounded(lonlat(corridors[link_id])),
    } for link_id in paths]

    # Only POIs inside some corridor are drawn: the others count for no link.
    merged = shapely.union_all(list(corridors.values()))
    shapely.prepare(merged)
    poi_columns = [c for c in columns if c.startswith("corridor_poi_")]
    category_index = {c[len("corridor_poi_"):-len("_per_km")]: i for i, c in enumerate(poi_columns)}
    pois = []
    with open(features / "pois.csv", newline="") as f:
        for r in csv.DictReader(f):
            x, y = float(r["x"]), float(r["y"])
            if merged.contains(shapely.Point(x, y)):
                lon, lat = TO_LONLAT(x, y)
                pois.append([round(lat, 5), round(lon, 5), category_index[r["category"]]])

    data = {"name": name.capitalize(), "city": city, "columns": columns, "links": links,
            "pois": pois, "merged": rounded(lonlat(merged))}
    out = ROOT / f"notes/maps/{name}_level1.html"
    out.write_text(PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), {len(links)} links, "
          f"{len(columns)} columns, {len(pois)} POIs inside corridors")


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Level 1 map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
html,body{margin:0;height:100%;font:13px system-ui,sans-serif;color:#212529}
#map{position:absolute;top:0;bottom:0;left:0;right:0}
.box{background:#fff;padding:8px 10px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);max-width:280px;line-height:1.5}
.box h1{font-size:15px;margin:0 0 6px}
.box select{width:100%;font:inherit;margin:2px 0 6px}
.sw{display:inline-block;width:22px;height:5px;margin-right:6px;vertical-align:middle;border-radius:2px}
.dot{display:inline-block;width:6px;height:6px;border-radius:50%;margin:0 14px 0 8px;vertical-align:middle;background:#212529}
.muted{color:#6c757d}
.leaflet-popup-content{margin:10px 12px}
.leaflet-popup-content table{border-collapse:collapse}
.leaflet-popup-content td{padding:1px 10px 1px 0}
.leaflet-popup-content td.n{text-align:right;font-variant-numeric:tabular-nums;padding-right:0}
.leaflet-popup-content tr.on td{font-weight:600}
.leaflet-popup-content th{text-align:left;padding-top:5px;color:#6c757d;font-weight:500}
</style></head><body><div id="map"></div>
<script>
const DATA=__DATA__;
// One blue ramp, light to dark, for "more of it". Grey means zero.
const RAMP=["#9ec5f4","#6da7ec","#3987e5","#1c5cab","#0d366b"], ZERO="#adb5bd", CORRIDOR="#f08c00";
const isPoi=c=>c.startsWith("corridor_poi_");
const label=c=>(isPoi(c)?c.slice(13,-7):c.slice(17)).replace(/_/g," ");
const fmt=(c,v)=>isPoi(c)?v.toFixed(1):(v*100).toFixed(1)+" %";
const unit=c=>isPoi(c)?"POIs per km of link":"share of corridor area";

const map=L.map("map",{preferCanvas:true});
const ESRI="https://server.arcgisonline.com/ArcGIS/rest/services/";
const bases={
  "Light map":L.tileLayer(ESRI+"Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",{maxZoom:20,maxNativeZoom:16,attribution:"Tiles &copy; Esri"}),
  "Street map":L.tileLayer(ESRI+"World_Street_Map/MapServer/tile/{z}/{y}/{x}",{maxZoom:20,maxNativeZoom:19,attribution:"Tiles &copy; Esri"}),
  "OpenStreetMap":L.tileLayer("https://tile.openstreetmap.de/{z}/{x}/{y}.png",{maxZoom:20,maxNativeZoom:19,
    attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'})};
bases["Light map"].addTo(map);
const renderer=L.canvas({tolerance:6});
let col=1, cuts=[], selected=null;

// Four cut points that split the links with a value above zero into five equal groups.
function cutPoints(i){
  const v=DATA.links.map(l=>l.v[i]).filter(x=>x>0).sort((a,b)=>a-b);
  if(!v.length)return [];
  return [...new Set([.2,.4,.6,.8].map(q=>v[Math.min(v.length-1,Math.floor(q*v.length))]))];
}
const colour=v=>v<=0?ZERO:RAMP[Math.round(cuts.filter(c=>v>=c).length*(RAMP.length-1)/Math.max(cuts.length,1))];

function popup(l){
  const row=(c,i)=>`<tr class="${i===col?"on":""}"><td>${label(c)}</td><td class="n">${fmt(c,l.v[i])}</td></tr>`;
  const part=(test,title)=>`<tr><th colspan="2">${title}</th></tr>`+DATA.columns.map((c,i)=>test(c)?row(c,i):"").join("");
  return `<b>${l.id.replace("link_","")}</b><br><span class="muted">${Math.round(l.len)} m long</span>
<table>${part(isPoi,"POIs per km of link")}${part(c=>!isPoi(c),"Land use, share of corridor area")}</table>`;
}
const lines=DATA.links.map(l=>{
  const layer=L.geoJSON(l.path,{renderer,style:{weight:4,opacity:.95}});
  layer.on("click",e=>{L.DomEvent.stopPropagation(e);
    if(selected)map.removeLayer(selected);
    selected=L.geoJSON(l.corridor,{interactive:false,style:{color:"#212529",weight:1.5,fillOpacity:.08}}).addTo(map);
    L.popup({maxHeight:420}).setLatLng(e.latlng).setContent(popup(l)).openOn(map)});
  layer.bindTooltip(()=>`${label(DATA.columns[col])}: <b>${fmt(DATA.columns[col],l.v[col])}</b>`,{sticky:true});
  return [l,layer];
});
map.on("popupclose",()=>{if(selected){map.removeLayer(selected);selected=null}});

const merged=L.geoJSON(DATA.merged,{interactive:false,style:{color:CORRIDOR,weight:1,fillColor:CORRIDOR,fillOpacity:.12}}).addTo(map);
const linkGroup=L.layerGroup(lines.map(x=>x[1])).addTo(map);
const poiGroup=L.layerGroup().addTo(map);
L.control.layers(bases,{"100 m corridors":merged,"Links, coloured by the chosen value":linkGroup,"POIs counted in the chosen value":poiGroup},{collapsed:false,position:"topright"}).addTo(map);
map.fitBounds(merged.getBounds());

const info=L.control({position:"bottomleft"});
info.onAdd=()=>{const d=L.DomUtil.create("div","box");L.DomEvent.disableClickPropagation(d);L.DomEvent.disableScrollPropagation(d);
  d.innerHTML=`<h1>${DATA.name} (${DATA.city}), level 1</h1>Colour the ${DATA.links.length} links by
<select id="col">${["POIs per km of link","Land use, share of corridor area"].map((g,k)=>`<optgroup label="${g}">`+
  DATA.columns.map((c,i)=>isPoi(c)===(k===0)?`<option value="${i}">${label(c)}</option>`:"").join("")+"</optgroup>").join("")}</select>
<div id="legend"></div>`;return d};
info.addTo(map);

function draw(){
  const c=DATA.columns[col]; cuts=cutPoints(col);
  lines.forEach(([l,layer])=>layer.setStyle({color:colour(l.v[col])}));
  poiGroup.clearLayers();
  let shown=0;
  if(isPoi(c)){const k=DATA.columns.filter(isPoi).indexOf(c);
    DATA.pois.forEach(p=>{if(p[2]===k){shown++;L.circleMarker([p[0],p[1]],{renderer,interactive:false,radius:2.5,stroke:false,fillColor:"#212529",fillOpacity:.75}).addTo(poiGroup)}})}
  const zero=DATA.links.filter(l=>l.v[col]<=0).length, edges=[0,...cuts];
  const used=[...new Set(edges.map((_,i)=>RAMP[Math.round(i*(RAMP.length-1)/Math.max(cuts.length,1))]))];
  let rows=`<span class="muted">${unit(c)}</span><br>`;
  if(zero)rows+=`<span class="sw" style="background:${ZERO}"></span>0 <span class="muted">(${zero} links)</span><br>`;
  if(cuts.length||zero<DATA.links.length)edges.forEach((lo,i)=>{const hi=edges[i+1];
    rows+=`<span class="sw" style="background:${used[i]}"></span>${i===0?"above 0":"from "+fmt(c,lo)}${hi===undefined?"":" to "+fmt(c,hi)}<br>`});
  rows+=isPoi(c)?`<span class="dot"></span>${shown} ${label(c)} POIs in the corridors<br>`:`<span class="muted">Land-use areas are not drawn.</span><br>`;
  document.getElementById("legend").innerHTML=rows+`<span class="muted">Hover a link for its value, click for all 20.</span>`;
}
const select=document.getElementById("col"); select.value=col;
select.addEventListener("change",()=>{col=+select.value;map.closePopup();draw()});
draw();
</script></body></html>
"""

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
