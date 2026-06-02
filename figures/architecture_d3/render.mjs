/* Headless render: build the same diagram via jsdom + D3, write architecture.svg.
   Usage: node render.mjs  (after: npm install d3 jsdom) */
import * as d3 from "d3";
import { JSDOM } from "jsdom";
import fs from "fs";

const dom = new JSDOM('<!DOCTYPE html><body></body>', { pretendToBeVisual: true });
globalThis.document = dom.window.document;

// load the shared drawing code (defines globalThis.drawArchitecture)
eval(fs.readFileSync(new URL("./architecture.js", import.meta.url), "utf8"));

const svg = d3.select(dom.window.document.body).append("svg")
  .attr("xmlns", "http://www.w3.org/2000/svg");
globalThis.drawArchitecture(svg, d3);

const out = '<?xml version="1.0" encoding="UTF-8"?>\n' + dom.window.document.body.innerHTML;
fs.writeFileSync(new URL("./architecture.svg", import.meta.url), out);
console.log("wrote architecture.svg");
