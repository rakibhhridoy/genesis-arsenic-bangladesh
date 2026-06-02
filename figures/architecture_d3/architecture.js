/* GENESIS architecture diagram (AIAYN-style), drawn with D3.
   Single source of truth: used by index.html (browser, D3 from CDN) and
   render.mjs (headless via jsdom) to emit a paper-ready SVG.
   drawArchitecture(svg, d3): svg is a D3 selection of an <svg> element. */
function drawArchitecture(svg, d3) {
  const W = 1060, H = 985;
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
  function vArrow(x, y1, y2) { // straight, upward
    svg.append("line").attr("x1", x).attr("y1", y1).attr("x2", x).attr("y2", y2)
      .attr("stroke", C.arrow).attr("stroke-width", 1.6).attr("marker-end", "url(#arr)");
  }
  function curve(d, { dash = false } = {}) {
    const p = svg.append("path").attr("fill", "none").attr("stroke", C.arrow)
      .attr("stroke-width", 1.5).attr("d", d).attr("marker-end", "url(#arr)");
    if (dash) p.attr("stroke-dasharray", "4 3");
    return p;
  }
  function stage(y, txt) {
    label(32, y, txt, { size: 12.5, weight: "700", fill: C.muted })
      .attr("transform", `rotate(-90 32 ${y})`);
  }

  // ===================== DATA SOURCES (bottom) =====================
  stage(894, "Data");
  const srcs = [["GEMStat", "943K"], ["USGS NWIS", "163K"], ["ADES", "694K"], ["EEA", "288K"], ["Bangladesh", "1,807"]];
  const sW = 150, gap = 24, totalSrc = srcs.length * sW + (srcs.length - 1) * gap;
  let sx = cx - totalSrc / 2;
  srcs.forEach(([n, c]) => {
    boxL(sx, 864, sW, 60, C.src, [n, c], { stroke: C.srcStroke, weight: "600", size: 13.5 });
    vArrow(sx + sW / 2, 864, 840);
    sx += sW + gap;
  });

  // ===================== HARMONIZATION / TOKENIZATION =====================
  stage(815, "Tokenize");
  boxL(cx - 380, 792, 760, 46, C.tok, "Harmonization → 20 chemistry parameters + 18 GEE auxiliary features",
    { stroke: C.srcStroke, weight: "600", size: 14 });
  vArrow(cx, 792, 770);

  // ===================== INPUT TOKEN ROW (hybrid, centered) =====================
  stage(718, "Input");
  const rowY = 666, rowH = 104;
  rrect(cx - 400, rowY, 800, rowH, "#F7F9FA", 12, "#CBD5DB", false);
  const chipY = 680, chipH = 30, chipMid = chipY + chipH / 2;
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
  // group legends centered under each group
  label((chemStart + chemEnd) / 2, 728, "chemistry tokens", { size: 11.5, fill: C.chemTxt, weight: "700" });
  label((auxStart + auxEnd) / 2, 728, "auxiliary (satellite) tokens", { size: 11.5, fill: C.auxTxt, weight: "700" });
  // projection note (inside row, clearly separated)
  label(cx, 752, "each token = type embedding + per-parameter value projection",
    { size: 11.5, fill: C.muted, italic: true });
  vArrow(cx, rowY, 630);

  // ===================== ENCODER BLOCK (N×, AIAYN style) =====================
  stage(499, "Encoder (N×)");
  const bx = cx - 190, bw = 380, bTop = 350, bBot = 630, bh = bBot - bTop;
  rrect(bx, bTop, bw, bh, C.block, 14, C.blockStroke, false);
  // N× badge — single line, centered on the block's left edge, vertically centered
  const badgeY = bTop + bh / 2;
  rrect(bx - 19, badgeY - 17, 38, 34, "#fff", 8, C.blockStroke, false);
  label(bx, badgeY, "N×", { size: 16, weight: "800", fill: C.muted });

  const slX = cx - 130, slW = 260;
  // bottom-up: MHA -> Add&Norm -> FFN -> Add&Norm
  boxL(slX, 548, slW, 56, C.mha, ["Multi-Head", "Self-Attention"], { size: 13.5, weight: "700" });
  boxL(slX, 498, slW, 32, C.norm, "Add & Norm", { size: 12.5, weight: "600" });
  boxL(slX, 430, slW, 48, C.ffn, "Feed Forward", { size: 13.5, weight: "700" });
  boxL(slX, 380, slW, 32, C.norm, "Add & Norm", { size: 12.5, weight: "600" });
  // internal straight flow
  vArrow(cx, 548, 530);   // MHA -> Add&Norm
  vArrow(cx, 498, 478);   // Add&Norm -> FFN
  vArrow(cx, 430, 412);   // FFN -> Add&Norm
  // curved residual connections (right side)
  const rx = slX + slW + 40;
  curve(`M ${cx} 600 C ${rx} 600, ${rx} 514, ${slX + slW} 514`, { dash: true }); // around MHA into Add&Norm
  curve(`M ${cx} 474 C ${rx} 474, ${rx} 396, ${slX + slW} 396`, { dash: true }); // around FFN into Add&Norm

  // ===================== [CLS] POOLING =====================
  stage(287, "Pool");
  vArrow(cx, 380, 314);
  boxL(cx - 165, 262, 330, 52, C.cls, "[CLS] pooling   →   latent  h ∈ ℝᵈ", { size: 14, weight: "700" });

  // ===================== HEADS (top, two curved branches) =====================
  stage(150, "Heads");
  curve(`M ${cx} 262 C ${cx} 222, ${cx - 170} 236, ${cx - 170} 196`); // to MGM head
  curve(`M ${cx} 262 C ${cx} 222, ${cx + 170} 236, ${cx + 170} 196`); // to classifier head
  boxL(cx - 290, 114, 240, 82, C.mgm, ["MGM Reconstruction Head", "(masked value prediction)"],
    { size: 13.5, weight: "700", fill: "#3B2415" });
  boxL(cx + 50, 114, 240, 82, C.clf, ["Classifier Head (MLP)", "WHO-threshold exceedance"],
    { size: 13.5, weight: "700", fill: "#1E3A2A" });
  label(cx - 170, 96, "pretraining objective", { size: 12, italic: true, fill: C.muted });
  label(cx + 170, 96, "downstream task", { size: 12, italic: true, fill: C.muted });
}

if (typeof globalThis !== "undefined") globalThis.drawArchitecture = drawArchitecture;
