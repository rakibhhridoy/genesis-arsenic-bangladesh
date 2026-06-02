/* GENESIS architecture diagram (AIAYN-style), drawn with D3.
   Flow is top -> bottom: data input at top, prediction heads at bottom.
   Single source of truth: used by index.html (browser, D3 from CDN) and
   render.mjs (headless via jsdom) to emit a paper-ready SVG.
   drawArchitecture(svg, d3): svg is a D3 selection of an <svg> element. */
function drawArchitecture(svg, d3) {
  const W = 1060, H = 930;
  svg.attr("viewBox", `0 0 ${W} ${H}`).attr("font-family", "Helvetica, Arial, sans-serif");

  const C = {
    text: "#283747", muted: "#5D6D7E", arrow: "#8A97A0",
    block: "#EEF2F4", blockStroke: "#B8C4CC",
    mha: "#F4A582", norm: "#FCE3A0", ffn: "#A6CEE3",
    cls: "#C7B8E0", mgm: "#E89B6C", clf: "#82C99A",
    chem: "#B8A6D9", aux: "#9CCFB0", clsTok: "#5D6D7E", wbt: "#D7B8A0",
    chemTxt: "#7E57C2", auxTxt: "#3E9C6B",
    tok: "#D5DBDB", src: "#E5E8E8", srcStroke: "#AAB7B8",
  };

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
    label(x + w / 2, y + h / 2, lines, { size: opt.size ?? 14, weight: opt.weight ?? "600", fill: opt.fill ?? C.text });
  }
  function vArrow(x, yTop, yBot) { // straight, downward
    svg.append("line").attr("x1", x).attr("y1", yTop).attr("x2", x).attr("y2", yBot)
      .attr("stroke", C.arrow).attr("stroke-width", 1.6).attr("marker-end", "url(#arr)");
  }
  function elbow(pts, r = 12, { dash = false } = {}) {
    // orthogonal polyline through waypoints with smoothly rounded corners
    let d = `M ${pts[0].x} ${pts[0].y}`;
    for (let i = 1; i < pts.length - 1; i++) {
      const p0 = pts[i - 1], p = pts[i], p1 = pts[i + 1];
      const l0 = Math.hypot(p.x - p0.x, p.y - p0.y), l1 = Math.hypot(p1.x - p.x, p1.y - p.y);
      const rr = Math.min(r, l0 / 2, l1 / 2);
      const a = { x: p.x - (p.x - p0.x) / l0 * rr, y: p.y - (p.y - p0.y) / l0 * rr };
      const b = { x: p.x + (p1.x - p.x) / l1 * rr, y: p.y + (p1.y - p.y) / l1 * rr };
      d += ` L ${a.x} ${a.y} Q ${p.x} ${p.y} ${b.x} ${b.y}`;
    }
    const last = pts[pts.length - 1];
    d += ` L ${last.x} ${last.y}`;
    const p = svg.append("path").attr("fill", "none").attr("stroke", C.arrow)
      .attr("stroke-width", 1.6).attr("d", d).attr("marker-end", "url(#arr)");
    if (dash) p.attr("stroke-dasharray", "4 3");
    return p;
  }
  function stage(y, txt) {
    label(32, y, txt, { size: 12.5, weight: "700", fill: C.muted })
      .attr("transform", `rotate(-90 32 ${y})`);
  }

  // ===================== DATA SOURCES (top) =====================
  stage(90, "Data");
  const srcs = [["GEMStat", "943K"], ["USGS NWIS", "163K"], ["ADES", "694K"], ["EEA", "288K"], ["Bangladesh", "1,807"]];
  const sW = 150, gap = 24, totalSrc = srcs.length * sW + (srcs.length - 1) * gap;
  let sx = cx - totalSrc / 2;
  srcs.forEach(([n, c]) => {
    boxL(sx, 60, sW, 60, C.src, [n, c], { stroke: C.srcStroke, weight: "600", size: 13.5 });
    vArrow(sx + sW / 2, 120, 148);
    sx += sW + gap;
  });

  // ===================== HARMONIZATION / TOKENIZATION =====================
  stage(171, "Tokenize");
  boxL(cx - 380, 148, 760, 46, C.tok, "Harmonization → 20 chemistry parameters + 18 GEE auxiliary features",
    { stroke: C.srcStroke, weight: "600", size: 14 });
  vArrow(cx, 194, 216);

  // ===================== INPUT TOKEN ROW (hybrid, centered) =====================
  stage(268, "Input");
  const rowY = 216, rowH = 104;
  rrect(cx - 400, rowY, 800, rowH, "#F7F9FA", 12, "#CBD5DB", false);
  const chipY = 230, chipH = 30, chipMid = chipY + chipH / 2;
  const chip = (x, w, fill, txt, tcol = C.text) => {
    rrect(x, chipY, w, chipH, fill, 6, null, false);
    label(x + w / 2, chipMid, txt, { size: 12.5, weight: "600", fill: tcol });
    return x + w;
  };
  const cw = { cls: 54, chem: 44, aux: 52, wbt: 58 };
  const gIn = 6, gSep = 22, gCls = 12;
  const chemToks = ["As", "Fe", "Eh", "PO₄", "U", "⋯"], auxToks = ["elev", "NDVI", "TWSA", "⋯"];
  const chemW = chemToks.length * cw.chem + (chemToks.length - 1) * gIn;
  const auxW = auxToks.length * cw.aux + (auxToks.length - 1) * gIn;
  const totalW = cw.cls + gCls + chemW + gSep + auxW + gSep + cw.wbt;
  let x = cx - totalW / 2;
  x = chip(x, cw.cls, C.clsTok, "[CLS]", "#fff") + gCls;
  const chemStart = x;
  chemToks.forEach(t => { x = chip(x, cw.chem, C.chem, t) + gIn; });
  const chemEnd = x - gIn; x = chemEnd + gSep;
  const auxStart = x;
  auxToks.forEach(t => { x = chip(x, cw.aux, C.aux, t) + gIn; });
  const auxEnd = x - gIn; x = auxEnd + gSep;
  chip(x, cw.wbt, C.wbt, "WBT");
  label((chemStart + chemEnd) / 2, 278, "chemistry tokens", { size: 11.5, fill: C.chemTxt, weight: "700" });
  label((auxStart + auxEnd) / 2, 278, "auxiliary (satellite) tokens", { size: 11.5, fill: C.auxTxt, weight: "700" });
  label(cx, 302, "each token = type embedding + per-parameter value projection",
    { size: 11.5, fill: C.muted, italic: true });
  vArrow(cx, 320, 384);

  // ===================== ENCODER BLOCK (N×, AIAYN style) =====================
  stage(496, "Encoder (N×)");
  const bx = cx - 190, bw = 380, bTop = 356, bBot = 636, bh = bBot - bTop;
  rrect(bx, bTop, bw, bh, C.block, 14, C.blockStroke, false);
  const badgeY = bTop + bh / 2;
  rrect(bx - 19, badgeY - 17, 38, 34, "#fff", 8, C.blockStroke, false);
  label(bx, badgeY, "N×", { size: 16, weight: "800", fill: C.muted });

  const slX = cx - 130, slW = 260;
  // top-down (flow downward): MHA -> Add&Norm -> FFN -> Add&Norm
  boxL(slX, 384, slW, 56, C.mha, ["Multi-Head", "Self-Attention"], { size: 13.5, weight: "700" });
  boxL(slX, 452, slW, 32, C.norm, "Add & Norm", { size: 12.5, weight: "600" });
  boxL(slX, 496, slW, 48, C.ffn, "Feed Forward", { size: 13.5, weight: "700" });
  boxL(slX, 556, slW, 32, C.norm, "Add & Norm", { size: 12.5, weight: "600" });
  // internal straight flow (downward)
  vArrow(cx, 440, 452);   // MHA -> Add&Norm
  vArrow(cx, 484, 496);   // Add&Norm -> FFN
  vArrow(cx, 544, 556);   // FFN -> Add&Norm
  // curved residual connections (right side, downward bypass)
  const rx = slX + slW + 40;
  elbow([{ x: cx, y: 372 }, { x: rx, y: 372 }, { x: rx, y: 468 }, { x: slX + slW, y: 468 }], 12, { dash: true }); // around MHA
  elbow([{ x: cx, y: 490 }, { x: rx, y: 490 }, { x: rx, y: 572 }, { x: slX + slW, y: 572 }], 12, { dash: true }); // around FFN

  // ===================== [CLS] POOLING =====================
  stage(698, "Pool");
  vArrow(cx, 636, 672);
  boxL(cx - 165, 672, 330, 52, C.cls, "[CLS] pooling   →   latent  h ∈ ℝᵈ", { size: 14, weight: "700" });

  // ===================== HEADS (bottom, two curved branches) =====================
  stage(833, "Heads");
  elbow([{ x: cx, y: 724 }, { x: cx, y: 760 }, { x: cx - 170, y: 760 }, { x: cx - 170, y: 792 }], 12); // to MGM head
  elbow([{ x: cx, y: 724 }, { x: cx, y: 760 }, { x: cx + 170, y: 760 }, { x: cx + 170, y: 792 }], 12); // to classifier head
  boxL(cx - 290, 792, 240, 82, C.mgm, ["MGM Reconstruction Head", "(masked value prediction)"],
    { size: 13.5, weight: "700", fill: "#3B2415" });
  boxL(cx + 50, 792, 240, 82, C.clf, ["Classifier Head (MLP)", "WHO-threshold exceedance"],
    { size: 13.5, weight: "700", fill: "#1E3A2A" });
  label(cx - 170, 888, "pretraining objective", { size: 12, italic: true, fill: C.muted });
  label(cx + 170, 888, "downstream task", { size: 12, italic: true, fill: C.muted });
}

if (typeof globalThis !== "undefined") globalThis.drawArchitecture = drawArchitecture;
