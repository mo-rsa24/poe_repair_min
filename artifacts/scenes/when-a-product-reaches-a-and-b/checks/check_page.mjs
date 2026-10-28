// Number check for where-the-product-lands.html. Runs the page's own KEY and COMPUTE blocks unchanged,
// one PASS or FAIL line per number. Usage: node check_page.mjs   (exit 1 on any FAIL)
import fs from "node:fs"; import path from "node:path"; import { fileURLToPath } from "node:url";
const here = path.dirname(fileURLToPath(import.meta.url));
const html = fs.readFileSync(path.join(here, "../where-the-product-lands.html"), "utf8");
const block = n => { const m = html.match(new RegExp(`// ==== ${n}:BEGIN[^\\n]*\\n([\\s\\S]*?)// ==== ${n}:END ====`)); if (!m) { console.log("FAIL block " + n); process.exit(1); } return m[1]; };
const api = new Function(block("KEY") + block("COMPUTE") + "\nreturn {KEY, factors, product, shares, keyRow, EPS, RS};")();
const scene = JSON.parse(fs.readFileSync(path.join(here, "scene_key.json"), "utf8"));
const walk = JSON.parse(fs.readFileSync(path.join(here, "../../../ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/answer_key.json"), "utf8"));
let fail = 0, total = 0;
const check = (name, ok, d) => { total++; if (!ok) fail++; console.log(`${ok ? "PASS" : "FAIL"}  ${name}: ${d}`); };
const near = (name, got, want, tol) => check(name, Math.abs(got - want) <= tol * Math.max(1, Math.abs(want)), `got ${got}, want ${want}`);
// 1. the embedded key is the scene key, byte for byte in value
for (const k of Object.keys(api.KEY)) check("embedded " + k, JSON.stringify(api.KEY[k]) === JSON.stringify(scene[k]), "equal to scene_key.json");
// 2. the page's compute reproduces the walk's own key rows on the same grid
const W = (c, f) => walk.find(r => r.case === c && f(r));
const pairs = [
  ["c2even", {eps:0.25}, W("construction 2, leftover even", r => r.eps === 0.25)],
  ["c2even", {eps:0.1}, W("construction 2, leftover even", r => r.eps === 0.1)],
  ["c2even", {eps:0.05}, W("construction 2, leftover even", r => r.eps === 0.05)],
  ["c2even", {eps:0}, W("construction 2, leftover even", r => r.eps === 0)],
  ["c2adj", {eps:0.1}, W("construction 2, leftover adjacent only", r => r.eps === 0.1)],
  ["c1", {r:0.05}, W("construction 1, contains labels", r => r.r_TR === 0.05)],
  ["c1", {r:0}, W("construction 1, contains labels", r => r.r_TR === 0)],
  ["b2plain", {}, W("2b factorised experts, plain product pA*pB", () => true)],
  ["b2div", {}, W("2b factorised experts, background-divided pA*pB/pbg", () => true)]];
for (const [mode, p, row] of pairs) { const got = api.shares(api.product(api.factors(mode, p)));
  for (const k of ["in_TR_blob","in_BR_blob","in_TL_blob","in_BL_blob","at_origin","Z"]) near(`page compute ${mode} ${JSON.stringify(p)} ${k} vs walk key`, got[k], row[k], 1e-9); }
// 3. every slider stop the page can reach has a key row, and the page reads the right one
for (const e of api.EPS) { check(`key row c2_even ε=${e}`, !!api.keyRow("c2even", {eps:e}), "present"); check(`key row c2_adjacent ε=${e}`, !!api.keyRow("c2adj", {eps:e}), "present"); }
for (const r of api.RS) check(`key row c1 r=${r}`, !!api.keyRow("c1", {r}), "present");
for (const q of ["TR","BR","TL","BL"]) { const got = api.shares(api.product(api.factors("support", {q}))); near(`support pin ${q} Z`, got.Z, scene.support_pins[q].Z, 1e-9); }
// 4. the numbers typed into narration match the key
const typed = [["0.989 origin, from scratch", 0.989, scene.c2_even["0"].at_origin, 3], ["0.006 top-right, from scratch", 0.006, scene.c2_even["0"].in_TR_blob, 3],
  ["0.25 step 0", 0.25, scene.c2_even["0.25"].in_TR_blob, 2], ["0.50 next door", 0.50, scene.c2_adjacent["0.1"].in_TR_blob, 2],
  ["0.989 construction 1", 0.989, scene.c1["0.05"].in_TR_blob, 3], ["0.970 construction 1 r=0", 0.970, scene.c1["0"].at_origin, 3],
  ["0.9999 2b plain", 0.9999, scene.b2_plain.in_TR_blob, 4], ["0.989 2b divided", 0.989, scene.b2_divided.in_TR_blob, 3],
  ["0.0278 at ε=0.05", 0.0278, scene.c2_even["0.05"].in_TR_blob, 4], ["0.472 at ε=0.05", 0.472, scene.c2_even["0.05"].in_BR_blob, 3],
  ["0.0625 hand check ε=0.1", 0.0625, scene.c2_even["0.1"].in_TR_blob, 4], ["0.4375 hand check ε=0.1 (key 0.4374 on the grid)", 0.4375, scene.c2_even["0.1"].in_BR_blob, 3],
  ["7.9e-10 Z from scratch", 7.9e-10, scene.c2_even["0"].Z, 11], ["3.54 Z overlap", 3.54, scene.support_pins.BR.Z, 2], ["1 - e^-4.5 = 0.9889", 0.9889, scene.one_blob_inside_3std.closed_form, 4],
  ["0.070 top-right ring at r=0", 0.070, scene.c1["0"].in_TR_blob, 3], ["0.9886 construction 1 r=0.001", 0.9886, scene.c1["0.001"].in_TR_blob, 4],
  ["Z 0.250 c1 r=0.25", 0.250, scene.c1["0.25"].Z, 3], ["Z 0.0500 c1 r=0.05", 0.0500, scene.c1["0.05"].Z, 4], ["Z 0.0100 c1 r=0.01", 0.0100, scene.c1["0.01"].Z, 4],
  ["Z 0.00100 c1 r=0.001", 0.00100, scene.c1["0.001"].Z, 5], ["Z 3.8e-7 c1 r=0", 3.8e-7, scene.c1["0"].Z, 8], ["0.006 grid at eps=0 (disclosure)", 0.006, scene.c2_even["0"].in_TR_blob, 3]];
for (const [n, shown, key, dp] of typed) check("narration " + n, Math.abs(shown - key) <= 0.5 * Math.pow(10, -dp) + 1e-15, `shown ${shown}, key ${key}`);
for (const n of ["0.989","0.006","0.9999","0.970","0.0278","7.9 × 10⁻¹⁰"]) check(`page text carries ${n}`, html.includes(n), "found");
console.log(`${fail ? "FAIL" : "PASS"}  ${total - fail} of ${total} checks`); process.exit(fail ? 1 : 0);
