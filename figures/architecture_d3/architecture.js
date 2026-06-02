/* GENESIS architecture diagram (AIAYN-style), drawn with D3.
   Single source of truth: used by index.html (browser, D3 from CDN) and
   render.mjs (headless via jsdom) to emit a paper-ready SVG.
   drawArchitecture(svg, d3): svg is a D3 selection of an <svg> element. */
function drawArchitecture(svg, d3) {
  const W = 1060, H = 1020;
  svg.attr("viewBox", `0 0 ${W} ${H}`).attr("font-family", "Helvetica, Arial, sans-serif");

  const C = {
    text: "#283747", muted: "#5D6D7E", arrow: "#8A97A0",
    block: "#EEF2F4", blockStroke: "#B8C4CC",
    mha: "#F4A582", norm: "#FCE3A0", ffn: "#A6CEE3",
    cls: "#C7B8E0", mgm: "#E89B6C", clf: "#82C99A",
    chem: "#B8A6D9", aux: "#9CCFB0", clsTok: "#5D6D7E", wbt: "#D7B8A0",
    tok: "#D5DBDB", src: "#E5E8E8", srcStroke: "#AAB7B8",
  };

  // defs: arrowhead + soft drop shadow
  const defs = svg.append("defs");
  defs.append("marker").attr("id", "arr").attr("viewBox", "0 0 10 10")
    .attr("refX", 8).attr("refY", 5).attr("markerWidth", 7).attr("markerHeight", 7)
    .attr("orient", "auto-start-reverse")
    .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", C.arrow);
  const f = defs.append("filter").attr("id", "sh").attr("x", "-20%").attr("y", "-20%")
    .attr("width", "140%").attr("height", "140%");
  f.append("feDropShadow").attr("dx", 0).attr("dy", 1.2).attr("stdDeviation", 1.4)
    .attr("flood-color", "#000").attr("flood-opacity", 0.16);

  svg.append("rect").attr("width", W).attr("height", H).attr("fill", "#ffffff");

  const cx = W / 2;

  // ---- helpers ----
  function rrect(x, y, w, h, fill, r = 9, stroke = null, shadow = true) {
    const e = svg.append("rect").attr("x", x).attr("y", y).attr("width", w).attr("height", h)
      .attr("rx", r).attr("ry", r).attr("fill", fill);
    if (stroke) e.attr("stroke", stroke).attr("stroke-width", 1.1);
    if (shadow) e.attr("filter", "url(#sh)");
    return e;
  }
  function label(x, y, lines, { size = 14, weight = "normal", fill = C.text, anchor = "middle", italic = false } = {}) {
    const arr = Array.isArray(lines) ? lines : [lines];
    const t = svg.append("text").attr("x", x).attr("y", y).attr("text-anchor", anchor)
      .attr("font-size", size).attr("font-weight", weight).attr("fill", fill)
      .attr("dominant-baseline", "middle");
    if (italic) t.attr("font-style", "italic");
    arr.forEach((ln, i) => t.append("tspan").attr("x", x)
      .attr("dy", i === 0 ? -(arr.length - 1) * (size * 0.62) : size * 1.24).text(ln));
    return t;
  }
  function boxL(x, y, w, h, fill, lines, opt = {}) {
    rrect(x, y, w, h, fill, opt.r ?? 9, opt.stroke, opt.shadow);
    label(x + w / 2, y + h / 2, lines, { size: opt.size ?? 14, weight: opt.weight ?? "600",
      fill: opt.fill ?? C.text });
  }
  function vArrow(x, y1, y2) { // upward
    svg.append("line").attr("x1", x).attr("y1", y1).attr("x2", x).attr("y2", y2)
      .attr("stroke", C.arrow).attr("stroke-width", 1.6).attr("marker-end", "url(#arr)");
  }
  function stage(y, txt) {
    label(34, y, txt, { size: 12.5, weight: "700", fill: C.muted, anchor: "middle" })
      .attr("transform", `rotate(-90 34 ${y})`);
  }

  // ===================== DATA SOURCES (bottom) =====================
  stage(905, "Data");
  const srcs = [
    ["GEMStat", "943K"], ["USGS NWIS", "163K"], ["ADES", "694K"],
    ["EEA", "288K"], ["Bangladesh", "1,807"],
  ];
  const sW = 150, gap = 24, totalW = srcs.length * sW + (srcs.length - 1) * gap;
  let sx = cx - totalW / 2;
  srcs.forEach(([n, c]) => {
    boxL(sx, 868, sW, 60, C.src, [n, c], { stroke: C.srcStroke, weight: "600", size: 13.5 });
    vArrow(sx + sW / 2, 868, 838);
    sx += sW + gap;
  });

  // ===================== HARMONIZATION / TOKENIZATION =====================
  stage(808, "Tokenize");
  boxL(cx - 380, 792, 760, 46, C.tok,
    "Harmonization → 20 chemistry parameters + 18 GEE auxiliary features",
    { stroke: C.srcStroke, weight: "600", size: 14 });
  vArrow(cx, 792, 768);

  // ===================== INPUT TOKEN ROW (hybrid, chemistry-specific) =====================
  stage(710, "Input");
  rrect(cx - 400, 686, 800, 76, "#F7F9FA", 12, "#CBD5DB", false);
  // chips
  const chip = (x, w, fill, txt, tcol = C.text) => {
    rrect(x, 700, w, 30, fill, 6, null, false);
    label(x + w / 2, 715, txt, { size: 12.5, weight: "600", fill: tcol });
  };
  let chx = cx - 384;
  chip(chx, 52, C.clsTok, "[CLS]", "#fff"); chx += 60;
  ["As", "Fe", "Eh", "PO₄", "U", "⋯"].forEach(t => { chip(chx, 44, C.chem, t); chx += 50; });
  chx += 8;
  ["elev", "NDVI", "TWSA", "⋯"].forEach(t => { chip(chx, 50, C.aux, t); chx += 56; });
  chx += 8;
  chip(chx, 58, C.wbt, "WBT");
  // legend + projection note
  label(cx - 384, 748, "chemistry tokens", { size: 11.5, fill: C.chem, anchor: "start", weight: "700" });
  label(cx + 40, 748, "auxiliary (satellite) tokens", { size: 11.5, fill: "#5BA77C", anchor: "start", weight: "700" });
  label(cx, 770, "each token = type embedding + per-parameter value projection",
    { size: 11.5, fill: C.muted, italic: true });
  vArrow(cx, 686, 648);

  // ===================== ENCODER BLOCK (N×, AIAYN style) =====================
  stage(495, "Encoder (N×)");
  const bx = cx - 190, bw = 380, bTop = 350, bBot = 648, bh = bBot - bTop;
  rrect(bx, bTop, bw, bh, C.block, 14, C.blockStroke, false);
  // "N x" badge
  rrect(bx - 4, bTop + bh / 2 - 22, 38, 44, "#fff", 8, C.blockStroke, false);
  label(bx + 15, bTop + bh / 2, ["N", "×"], { size: 15, weight: "800", fill: C.muted });

  const slX = cx - 130, slW = 260;
  // bottom-up: MHA -> Add&Norm -> FFN -> Add&Norm
  boxL(slX, 566, slW, 56, C.mha, ["Multi-Head", "Self-Attention"], { size: 13.5, weight: "700" });
  boxL(slX, 514, slW, 34, C.norm, "Add & Norm", { size: 13, weight: "600" });
  boxL(slX, 442, slW, 50, C.ffn, "Feed Forward", { size: 13.5, weight: "700" });
  boxL(slX, 390, slW, 34, C.norm, "Add & Norm", { size: 13, weight: "600" });
  // internal flow arrows
  vArrow(cx, 566, 548);   // MHA -> Add&Norm
  vArrow(cx, 514, 492);   // Add&Norm -> FFN
  vArrow(cx, 442, 424);   // FFN -> Add&Norm
  // residual connections (right side, curving around each sublayer pair)
  const resid = (yIn, yBox) => {
    const rx = slX + slW + 26;
    svg.append("path").attr("fill", "none").attr("stroke", C.arrow).attr("stroke-width", 1.4)
      .attr("stroke-dasharray", "4 3")
      .attr("d", `M ${cx} ${yIn} L ${rx} ${yIn} L ${rx} ${yBox} L ${slX + slW} ${yBox}`)
      .attr("marker-end", "url(#arr)");
  };
  resid(560, 531);  // around MHA into Add&Norm
  resid(436, 407);  // around FFN into Add&Norm

  // ===================== [CLS] POOLING =====================
  stage(286, "Pool");
  vArrow(cx, 390, 318);
  boxL(cx - 165, 262, 330, 50, C.cls, "[CLS] pooling   →   latent  h ∈ ℝᵈ",
    { size: 14, weight: "700" });

  // ===================== HEADS (top, two branches) =====================
  stage(140, "Heads");
  // split arrows
  svg.append("path").attr("fill", "none").attr("stroke", C.arrow).attr("stroke-width", 1.6)
    .attr("d", `M ${cx} 262 L ${cx} 230 L ${cx - 170} 230 L ${cx - 170} 196`).attr("marker-end", "url(#arr)");
  svg.append("path").attr("fill", "none").attr("stroke", C.arrow).attr("stroke-width", 1.6)
    .attr("d", `M ${cx} 262 L ${cx} 230 L ${cx + 170} 230 L ${cx + 170} 196`).attr("marker-end", "url(#arr)");

  boxL(cx - 290, 112, 240, 82, C.mgm, ["MGM Reconstruction Head", "(masked value prediction)"],
    { size: 13.5, weight: "700", fill: "#3B2415" });
  boxL(cx + 50, 112, 240, 82, C.clf, ["Classifier Head (MLP)", "WHO-threshold exceedance"],
    { size: 13.5, weight: "700", fill: "#1E3A2A" });
  label(cx - 170, 92, "pretraining objective", { size: 12, italic: true, fill: C.muted });
  label(cx + 170, 92, "downstream task", { size: 12, italic: true, fill: C.muted });
}

if (typeof globalThis !== "undefined") globalThis.drawArchitecture = drawArchitecture;
