"""Compose native draw.io model diagrams with 13 embedded R SVG panels."""
from pathlib import Path
from urllib.parse import quote
import xml.etree.ElementTree as ET
import pandas as pd
import argparse

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description="Build the original v6 editable technical route.")
parser.add_argument("--route-dir", type=Path, default=ROOT / "paper/technical_route_v6")
parser.add_argument("--current-benchmarks", action="store_true", help="Update labels for the current benchmark panels while preserving all geometry.")
args = parser.parse_args()
OUT = args.route_dir
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1800, 1960
INK, GREY, LINE = "#202B33", "#667580", "#A6B1BA"
BLUE, GREEN, PURPLE, RED, ORANGE = "#2166AC", "#1B7837", "#762A83", "#B2182B", "#F1A340"
STATE = ["#2166AC", "#92C5DE", "#1B7837", "#A6DBA0", "#762A83",
         "#C2A5CF", "#B2182B", "#EF8A62", "#F1A340", "#999999"]
mxfile = ET.Element("mxfile", host="Electron", type="device")
diagram = ET.SubElement(mxfile, "diagram", id="iot-roadmap-v6", name="Technical route")
model = ET.SubElement(diagram, "mxGraphModel", dx=str(W), dy=str(H), grid="1", gridSize="10",
                      guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1",
                      pageScale="1", pageWidth=str(W), pageHeight=str(H), math="0", shadow="0",
                      background="#FFFFFF")
root = ET.SubElement(model, "root")
ET.SubElement(root, "mxCell", id="0")
ET.SubElement(root, "mxCell", id="1", parent="0")
origins = {"1": (0, 0)}


def vertex(ident, value, x, y, w, h, style, parent="1"):
    ox, oy = origins[parent]
    cell = ET.SubElement(root, "mxCell", id=ident, value=value, parent=parent, vertex="1",
                         style=style + ";fontFamily=Arial;html=0;whiteSpace=wrap;overflow=hidden;")
    ET.SubElement(cell, "mxGeometry", x=str(x-ox), y=str(y-oy), width=str(w), height=str(h),
                  **{"as": "geometry"})
    return ident


def group(ident, x, y, w, h):
    vertex(ident, "", x, y, w, h, "group;connectable=0;")
    origins[ident] = (x, y)
    return ident


def rect(ident, x, y, w, h, fill="white", stroke=LINE, rounded=False, dashed=False, parent="1"):
    return vertex(ident, "", x, y, w, h,
                  f"rounded={int(rounded)};arcSize=8;strokeColor={stroke};strokeWidth=1.5;"
                  f"fillColor={fill};dashed={int(dashed)};dashPattern=6 4;", parent)


def text(ident, value, x, y, w, h, size=21, color=INK, bold=False, align="left", parent="1"):
    return vertex(ident, value, x, y, w, h,
                  f"text;strokeColor=none;fillColor=none;align={align};verticalAlign=middle;"
                  f"fontSize={size};fontColor={color};fontStyle={int(bold)};spacing=0;", parent)


def node(ident, value, x, y, w, h, fill, color=INK, size=21, shape="rectangle", parent="1"):
    return vertex(ident, value, x, y, w, h,
                  f"shape={shape};rounded=1;arcSize=12;strokeColor={color};strokeWidth=1.5;"
                  f"fillColor={fill};fontSize={size};fontColor={color};align=center;verticalAlign=middle;spacing=5;", parent)


def edge(ident, source=None, target=None, points=(), color=GREY, dashed=False, arrow=True,
         width=1.7, start=(1, .5), end=(0, .5), straight=False, source_point=None, target_point=None):
    attrs = dict(id=ident, parent="1", edge="1")
    if source: attrs["source"] = source
    if target: attrs["target"] = target
    attrs["style"] = (f"edgeStyle={'none' if straight else 'orthogonalEdgeStyle'};rounded=0;"
                      f"strokeColor={color};strokeWidth={width};endArrow={'block' if arrow else 'none'};"
                      f"endSize=7;endFill=1;dashed={int(dashed)};dashPattern=4 3;"
                      f"exitX={start[0]};exitY={start[1]};entryX={end[0]};entryY={end[1]};"
                      "exitDx=0;exitDy=0;entryDx=0;entryDy=0;html=0;")
    cell = ET.SubElement(root, "mxCell", **attrs)
    geom = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
    for point, name in [(source_point, "sourcePoint"), (target_point, "targetPoint")]:
        if point:
            ET.SubElement(geom, "mxPoint", x=str(point[0]), y=str(point[1]), **{"as": name})
    if points:
        arr = ET.SubElement(geom, "Array", **{"as": "points"})
        for x, y in points: ET.SubElement(arr, "mxPoint", x=str(x), y=str(y))
    return ident


def section(ident, title, x, y, w, h, color, fill, tag=None):
    group(ident, x, y, w, h)
    rect(ident+"-border", x, y, w, h, "#FFFFFF", color, dashed=True, parent=ident)
    rect(ident+"-header", x+2, y+2, w-4, 62, fill, "none", parent=ident)
    text(ident+"-heading", title, x+18, y+8, w-(160 if tag else 36), 46, 29, color, True, parent=ident)
    if tag:
        text(ident+"-tag", tag, x+w-135, y+11, 115, 41, 21, color, align="right", parent=ident)
    return ident


def panel(ident, name, title, x, y, w, h, parent, size=23):
    text(ident+"-title", title, x, y, w, 35, size, INK, True, parent=parent)
    path = OUT / "R_panels" / (name+".svg")
    svg = path.read_text(encoding="utf-8")
    svg = svg[svg.index("<svg"):]
    uri = "data:image/svg+xml," + quote(svg, safe="")
    vertex(ident+"-plot", "", x, y+43, w, h,
           "shape=image;verticalLabelPosition=bottom;verticalAlign=top;imageAspect=0;"
           f"aspect=fixed;image={uri};", parent)


rect("page", 0, 0, W, H, "#FFFFFF", "none")
text("figure-title", "Figure 1 | Technical route for inverse optimal transport", 24, 20, 1750, 55,
     36, INK, True)

data = section("data", "Data preparation", 20, 100, 340, 1200, "#536D86", "#DFE8F0")
text("data-cohort", "GSE228154 · 3,208 cells", 35, 175, 310, 30, 21, INK, True, parent=data)
panel("pca", "01_expression_pca", "Expression PCA", 35, 213, 310, 240, data)
edge("data-divider-1", source_point=(35,510), target_point=(345,510), arrow=False, color="#CBD4DB", straight=True)
panel("composition", "02_state_composition", "Treatment-state composition", 35, 529, 310, 240, data, size=21)
edge("data-divider-2", source_point=(35,822), target_point=(345,822), arrow=False, color="#CBD4DB", straight=True)
panel("programmes", "03_state_programmes", "State × programme features", 35, 841, 310, 240, data, size=21)
node("representation", "QC → state labels → a, b, φ", 35, 1154, 310, 55, "#ECF1F5", "#536D86", 21, parent=data)
text("cohort-scope", "Context / lineage metadata\nIndependent prospective inputs", 39, 1220, 303, 63, 20, INK, parent=data)

s1 = section("analysis1", "1 | Identify transport costs and test transfer", 385, 100, 1395, 580,
             BLUE, "#DCEAF5", "Fig. 2")
rect("uot-method", 405, 181, 535, 477, "#F7FAFD", "#BECEDB", True, parent=s1)
text("transport-heading", "From cell states to transport costs", 423, 194, 499, 35, 24, BLUE, True, parent=s1)

# Native state bars use the actual D0 and D9 proportions; matrix is labelled schematic.
cells = pd.read_csv(OUT / "source_data/GSE228154_cells.csv")
for ident, tp, x in [("source", "D0", 464), ("target", "D9", 852)]:
    text(ident+"-label", f"{('Source a' if tp=='D0' else 'Target b')}\n{tp}", x-34, 241, 129, 52,
         20, INK, True, "center", s1)
    counts = cells.loc[cells.timepoint.eq(tp), "state"].value_counts(normalize=True)
    top = 304
    for state in range(10):
        height = float(counts.get(state,0))*92
        if height > 0:
            rect(f"{ident}-state-{state}", x, top, 54, height, STATE[state], "none", parent=s1)
        top += height
    rect(ident+"-outline", x, 304, 54, 92, "none", "#748493", parent=s1)
rect("pi-matrix", 618, 274, 120, 120, "#FFFFFF", "#90AAC1", parent=s1)
for i in range(4):
    for j in range(4):
        rect(f"pi-{i}-{j}", 621+j*29, 277+i*29, 27, 27,
             "#6E9AC3" if i==j else ("#C4D9EB" if abs(i-j)==1 else "#EBF2F8"), "none", parent=s1)
text("pi-label", "πθ", 649, 237, 60, 32, 27, BLUE, True, "center", s1)
text("pi-key", "Coupling schematic", 573, 397, 210, 26, 19, INK, False, "center", s1)
edge("source-to-plan", "source-outline", "pi-matrix", color=BLUE)
edge("plan-to-target", "pi-matrix", "target-outline", color=BLUE)
node("balanced", "Balanced OT\nπ1 = a;  πᵀ1 = b", 425, 440, 220, 75, "#F0F1F2", GREY, 21, parent=s1)
node("kl-anchor", "UOT-IOT\nπ1 = a;  μ DKL(πᵀ1 ∥ b)", 694, 440, 226, 75, "#E1EEF8", BLUE, 20, parent=s1)
edge("hard-to-soft", "balanced", "kl-anchor", color=BLUE)
text("cost-formula", "Cθ(i,j) = C₀(i,j) − θᵀφ(i,j)", 427, 535, 490, 44, 27, BLUE, True, "center", s1)
text("inverse-fit", "Inverse fit → θ and πθ", 430, 581, 485, 34, 23, INK, False, "center", s1)
for ident, label, x, w in [("curv","Curvature",425,150),("scan","μ / ε scan",590,150),("restart","Restarts",755,165)]:
    node("diag-"+ident, label, x, 620, w, 27, "#FFFFFF", BLUE, 19, parent=s1)
panel("curvature", "04_curvature", "Empirical curvature", 965, 176, 380, 200, s1)
panel("recovery", "05_synthetic_recovery", "Synthetic coefficient recovery", 1375, 176, 380, 200, s1, 21)
panel("sensitivity", "06_anchor_sensitivity", "KL-anchor sensitivity", 965, 423, 380, 200, s1)
panel("external", "07_external_transfer", "Frozen transfer · six sites", 1375, 423, 380, 200, s1, 22)
edge("data-to-uot", source_point=(360,347), target_point=(405,347), color=BLUE, straight=True, width=2.2)

s2 = section("analysis2", "2 | Resolve context and lineage-history effects", 385, 700, 1395, 600,
             GREEN, "#E1EDE3", "Figs. 3–4")
rect("gh-method", 405, 781, 535, 499, "#F6FAF6", "#C0D1C2", True, parent=s2)
text("gh-heading", "GH-IOT: shared and context effects", 423, 794, 499, 38, 24, GREEN, True, parent=s2)
node("shared", "Shared direction θshared", 550, 849, 244, 60, "#DEEDDF", GREEN, 23, parent=s2)
node("tena-context", "TENA context\nθc = θshared + δc", 425, 960, 216, 76, "#EAF1F8", BLUE, 21, parent=s2)
node("cdh-context", "Cdh context\nθc = θshared + δc", 704, 960, 216, 76, "#F8E9EA", RED, 21, parent=s2)
edge("shared-to-tena", "shared", "tena-context", points=((490,934),), color=GREEN, start=(.25,1), end=(.5,0))
edge("shared-to-cdh", "shared", "cdh-context", points=((855,934),), color=GREEN, start=(.75,1), end=(.5,0))
text("shrinkage", "Shrinkage", 608, 917, 130, 28, 19, GREEN, False, "center", s2)
node("training-prior", "Training-group IOT prior", 425, 1066, 216, 54, "#FFFFFF", GREEN, 20, parent=s2)
node("context-residual", "Context residual", 704, 1066, 216, 54, "#FFFFFF", GREEN, 21, parent=s2)
edge("tena-to-prior", "tena-context", "training-prior", color=GREEN, start=(.5,1), end=(.5,0))
edge("cdh-to-prior", "cdh-context", "training-prior", points=((812,1050),(597,1050)),
     color=GREEN, start=(.5,1), end=(.8,0))
node("gate", "Gate αc", 585, 1160, 176, 51, "#DEEDDF", GREEN, 23, parent=s2)
edge("prior-to-gate", "training-prior", "gate", points=((535,1140),(630,1140)), color=GREEN, start=(.5,1), end=(.25,0))
edge("residual-to-gate", "context-residual", "gate", points=((812,1140),(720,1140)), color=GREEN, start=(.5,1), end=(.75,0))
text("heldout-title", "Group-wise evaluation", 426, 1224, 246, 27, 20, GREEN, True, parent=s2)
for i, label in enumerate(["g₁", "g₂", "…", "gₖ", "gₕ"]):
    rect(f"fold-{i}", 429+i*44, 1252, 36, 25,
         "#C8DFCC" if i<4 else "#FFFFFF", GREEN, parent=s2)
    text(f"fold-number-{i}", label, 429+i*44, 1251, 36, 26, 18, GREEN, align="center", parent=s2)
text("train-heldout-key", "Train / held out", 684, 1238, 221, 31, 19, INK, False, "center", s2)
edge("gate-evaluation", "gate", "train-heldout-key", color=GREEN, start=(.8,1), end=(.5,0))
panel("lineage", "08_lineage_pairs", "Lineage history · GSE171940", 965, 777, 380, 255, s2, 21)
panel("response", "09_patient_response", "Treatment response · ARTEMIS", 1375, 777, 380, 255, s2, 21)
edge("patient-divider", source_point=(965,1090), target_point=(1755,1090), arrow=False, color="#CFDAD0", straight=True)
panel("patient-heatmap", "10_patient_programme_heatmap", "Patient × programme profiles · all 79 patients", 965, 1100, 790, 150, s2, 22)
edge("data-to-gh", source_point=(360,1020), target_point=(405,1020), color=GREEN, straight=True, width=2.2)

s3 = section("analysis3", "3 | Predict future composition and persistence", 20, 1320, 1760, 580,
             PURPLE, "#ECDFF0", "Fig. 5")
rect("persist-method", 40, 1401, 785, 475, "#FAF7FB", "#D6C4DB", True, parent=s3)
text("persist-heading", "PERSIST-IOT: source inputs and observation model", 59, 1410, 750, 38, 24, PURPLE, True, parent=s3)
node("forecast-source", "Source inputs", 60, 1470, 180, 160, "#FFFFFF", PURPLE, 22, parent=s3)
# Override the node label with separate title so the small source-population glyph is clear.
root.find("mxCell[@id='forecast-source']").set("value", "")
text("source-input-title", "Source inputs", 69, 1481, 162, 32, 22, PURPLE, True, "center", s3)
text("source-variables", "s₀, n₀, c, Δt", 67, 1518, 166, 37, 24, INK, False, "center", s3)
for i, (cx, cy, c) in enumerate([(90,1585,0),(132,1571,4),(170,1590,8),(207,1570,2)]):
    node("source-cell-"+str(i), "", cx-10, cy-10, 20, 20, STATE[c], STATE[c], shape="ellipse", parent=s3)
node("generator", "", 289, 1470, 211, 160, "#FFFFFF", PURPLE, parent=s3)
text("generator-title", "IOT generator Qθ", 300, 1480, 190, 34, 21, PURPLE, True, "center", s3)
q0=node("q0", "0", 321, 1565, 32, 32, "#F1E7F4", PURPLE, 19, "ellipse", s3)
q1=node("q1", "1", 378, 1523, 32, 32, "#F1E7F4", PURPLE, 19, "ellipse", s3)
q2=node("q2", "2", 443, 1565, 32, 32, "#F1E7F4", PURPLE, 19, "ellipse", s3)
edge("q01", q0, q1, color=PURPLE, straight=True, start=(.8,.1), end=(.2,.9))
edge("q12", q1, q2, color=PURPLE, straight=True, start=(.8,.9), end=(.2,.1))
edge("q20", q2, q0, color=PURPLE, straight=True, start=(0,.5), end=(1,.5))
node("hmm", "", 553, 1470, 253, 160, "#FFFFFF", PURPLE, parent=s3)
text("hmm-title", "Observation-corrected HMM", 563, 1479, 233, 40, 20, PURPLE, True, "center", s3)
z0=node("z0", "Zt", 583, 1530, 44, 35, "#DCC1E3", PURPLE, 19, "ellipse", s3)
z1=node("z1", "Zt+1", 731, 1530, 52, 35, "#DCC1E3", PURPLE, 18, "ellipse", s3)
y0=node("y0", "Yt", 583, 1587, 44, 30, "#FFFFFF", PURPLE, 19, "rectangle", s3)
y1=node("y1", "Yt+1", 731, 1587, 52, 30, "#FFFFFF", PURPLE, 18, "rectangle", s3)
edge("latent-evolution", z0, z1, color=PURPLE)
edge("observation0", z0, y0, color=PURPLE, start=(.5,1), end=(.5,0))
edge("observation1", z1, y1, color=PURPLE, start=(.5,1), end=(.5,0))
edge("source-generator", "forecast-source", "generator", color=PURPLE)
edge("generator-hmm", "generator", "hmm", color=PURPLE)
text("forecast-design", "GSE239651\nSource-only forecast", 64, 1657, 227, 64, 20, INK, False, "center", s3)
node("future-output", "Future state composition\nb̂(t)", 81, 1752, 314, 78, "#EEE2F2", PURPLE, 24, parent=s3)
node("persistence-output", "Persistence / detection\nP(Zt), P(Yt)", 470, 1752, 314, 78, "#EEE2F2", PURPLE, 24, parent=s3)
edge("generator-composition", "generator", "future-output", points=((395,1724),(238,1724)),
     color=PURPLE, start=(.5,1), end=(.5,0))
edge("hmm-detection", "hmm", "persistence-output", points=((680,1724),(627,1724)),
     color=PURPLE, start=(.5,1), end=(.5,0))
text("freeze-design", "Five outer folds → freeze → locked external E1", 68, 1841, 727, 29, 21, PURPLE, True, "center", s3)
panel("calibration", "11_external_calibration", "External composition calibration", 860, 1427, 430, 335, s3, 21)
text("calibration-unit", "2 conditions × 28 windows × 6 states", 861, 1833, 430, 33, 20, INK, False, "center", s3)
panel("performance", "12_state_performance", "Composition performance", 1324, 1427, 420, 170, s3, 23)
panel("detection", "13_detection_brier", "Persistence detection", 1324, 1669, 420, 170, s3, 23)
edge("persist-evaluate", source_point=(825,1440), target_point=(849,1440), color=PURPLE, straight=True, width=2)
text("footnote", "a, b: state distributions   •   φ: transport features   •   θ: cost directions   •   δc: context offsets   •   Zt / Yt: latent persistence / observed detection",
     28, 1919, 1744, 27, 20, INK)

if args.current_benchmarks:
    updates = {
        "figure-title": "Figure 1 | Inverse optimal transport and current benchmark design",
        "diag-scan": "Known truth",
        "tena-context": "Context 1\\nθc = θshared + δc",
        "cdh-context": "Context 2\\nθc = θshared + δc",
        "calibration-unit": "Blue: GSE140802 · Orange: GSE239651 expt2",
        "footnote": "B1: six scenarios × three sample sizes × five seeds   •   B2: four lineage panels   •   B3: clone and population forecasts",
    }
    # Panel and section title cell IDs are generated by their helper functions.
    text_replacements = {
        "Empirical curvature": "Known-truth curvature",
        "KL-anchor sensitivity": "Direction stability",
        "2 | Resolve context and lineage-history effects": "2 | Context model and common method comparisons",
        "Figs. 3–4": "Figs. 3, 5",
        "Lineage history · GSE171940": "Lineage transition recovery",
        "Treatment response · ARTEMIS": "Output-proxy stability",
        "Patient × programme profiles · all 79 patients": "Comparable signed feature parameters · adapter capability",
        "3 | Predict future composition and persistence": "3 | Forecast future composition and evaluate calibration",
        "Fig. 5": "Fig. 4",
        "External composition calibration": "Population calibration",
        "Composition performance": "Clone-level composition",
        "Persistence detection": "Population-level composition",
    }
    for cell in root.findall("mxCell"):
        value = cell.get("value", "")
        if cell.get("id") in updates:
            cell.set("value", updates[cell.get("id")].replace(chr(92) + "n", chr(10)))
        elif value in text_replacements:
            cell.set("value", text_replacements[value])

ET.indent(mxfile, space="  ")
path = OUT / "Fig1_technical_route_v6.drawio"
ET.ElementTree(mxfile).write(path, encoding="utf-8", xml_declaration=True)
print(f"Saved {path}; {len(root.findall('mxCell[@vertex=\"1\"]'))} vertices, "
      f"{len(root.findall('mxCell[@edge=\"1\"]'))} connectors, 13 embedded R SVGs")
