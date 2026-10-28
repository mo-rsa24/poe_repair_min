// Number check for parts-world.html. Runs the page's own KEY and COMPUTE blocks unchanged against
// scene_key.json, one PASS or FAIL line per number (grid and noise sweeps summarised per field).
// Usage: node check_page.mjs   (exit 1 on any FAIL)
import fs from "node:fs"; import path from "node:path"; import { fileURLToPath } from "node:url";
const here = path.dirname(fileURLToPath(import.meta.url));
const html = fs.readFileSync(path.join(here, "../parts-world.html"), "utf8");
const block = n => { const m = html.match(new RegExp(`// ==== ${n}:BEGIN[^\\n]*\\n([\\s\\S]*?)// ==== ${n}:END ====`)); if (!m) { console.log("FAIL block " + n); process.exit(1); } return m[1]; };
const api = new Function(block("KEY") + block("COMPUTE") + "\nreturn {KEY, single, noised, mixture, VSTOPS, ASTOPS, LEFT, RIGHT, avg};")();
const key = JSON.parse(fs.readFileSync(path.join(here, "scene_key.json"), "utf8"));
const walk = JSON.parse(fs.readFileSync(path.join(here, "../../../ideas/poe-overlap-dial/example-9-parts-world/numbers.json"), "utf8"));
let fail = 0, total = 0;
const check = (name, ok, d) => { total++; if (!ok) fail++; console.log(`${ok ? "PASS" : "FAIL"}  ${name}: ${d}`); };
const rel = (a, b) => Math.abs(a - b) / Math.max(1, Math.abs(b));
// 1. the embedded key is the scene key, and its walk part is the walk's numbers.json
for (const k of Object.keys(api.KEY)) check("embedded " + k, JSON.stringify(api.KEY[k]) === JSON.stringify(key[k]), "equal to scene_key.json");
check("embedded walk key", JSON.stringify(api.KEY.walk) === JSON.stringify(walk), "equal to numbers.json");
check("variance stops", JSON.stringify(api.VSTOPS) === JSON.stringify(key.vstops), api.VSTOPS.join(", "));
check("noise stops", JSON.stringify(api.ASTOPS) === JSON.stringify(key.astops), api.ASTOPS.join(", "));
// 2. the walk's 18 numbers from the page's compute
const a = api.single(0.01, 1, 1, 0.01), b = api.single(0.01, 0.01, 0.01, 0.01), d = api.noised(0.01, 1, 1, 0.01, 0.5), mx = api.mixture();
const pairs = [["a head_mean", a.head_mean, walk.a.head_mean], ["a body_mean", a.body_mean, walk.a.body_mean], ["a logZ", a.logZ, walk.a.logZ],
  ["b mean", (b.head_mean + b.body_mean) / 2, walk.b.mean], ["b product_sd", Math.sqrt(b.head_var), walk.b.product_sd], ["b logZ", b.logZ, walk.b.logZ],
  ["b sds_from_each_expert", (1 - b.head_mean) / 0.1, walk.b.sds_from_each_expert],
  ["d head product of noised", d.head_summed, walk.d.head_mean_product_of_noised], ["d head noised product", d.head_target, walk.d.head_mean_noised_product]];
for (const p of mx) { pairs.push([`c ${p.name} logw`, p.logw, walk.c[p.name].logw_unnorm]); pairs.push([`c ${p.name} weight`, p.w, walk.c[p.name].weight]);
  pairs.push([`c ${p.name} left mean`, api.avg(p.m, api.LEFT), key.c_component_means[p.name].left]); pairs.push([`c ${p.name} right mean`, api.avg(p.m, api.RIGHT), key.c_component_means[p.name].right]); }
for (const [n, got, want] of pairs) check("page compute " + n, rel(got, want) <= 1e-12, `got ${got}, want ${want}`);
// 3. every variance stop and every noise state
let worstG = 0, nG = 0;
for (const [k, row] of Object.entries(key.grid)) { const [ch, cb, dh, db] = k.split(",").map(Number); const s = api.single(ch, cb, dh, db);
  for (const f of ["head_mean", "body_mean", "head_var", "body_var", "logZ"]) { worstG = Math.max(worstG, rel(s[f], row[f])); nG++; } }
check(`page compute on all ${Object.keys(key.grid).length} variance stops (${nG} numbers)`, worstG <= 1e-12, `worst relative gap ${worstG.toExponential(1)}`);
let worstN = 0, nN = 0;
for (const [k, row] of Object.entries(key.noise)) { const [vs, ab] = k.split("|"); const [ch, cb, dh, db] = vs.split(",").map(Number); const s = api.noised(ch, cb, dh, db, Number(ab));
  for (const f of ["head_summed", "body_summed", "head_target", "body_target"]) { worstN = Math.max(worstN, rel(s[f], row[f])); nN++; } }
check(`page compute on all ${Object.keys(key.noise).length} noise states (${nN} numbers)`, worstN <= 1e-12, `worst relative gap ${worstN.toExponential(1)}`);
// 4. numbers typed into narration and option text, against the key at the digits shown
const typed = [["0.980 head (99/101)", 0.980, walk.a.head_mean, 3], ["10 sds", 10, walk.b.sds_from_each_expert, 0], ["sd 0.071", 0.071, walk.b.product_sd, 3],
  ["-98.96 sure vs sure per pixel", -98.96, key.pixel_logZ.sure_vs_sure_opposite, 2], ["-2.904 sure vs loose per pixel", -2.904, key.pixel_logZ.sure_vs_loose_opposite, 3],
  ["-98.962927 disclosure", -98.962927, key.pixel_logZ.sure_vs_sure_opposite, 6], ["-2.904112 disclosure", -2.904112, key.pixel_logZ.sure_vs_loose_opposite, 6],
  ["-1.418963 disclosure", -1.418963, key.pixel_logZ.sure_vs_loose_zero, 6], ["-1.265512 disclosure", -1.265512, key.pixel_logZ.loose_zero_vs_loose_zero, 6],
  ["-814.936 disclosure", -814.936, key.a_dog_head_tight.logZ, 3], ["-22.703 side-by-side sum", -22.703, 16 * key.pixel_logZ.sure_vs_loose_zero, 3],
  ["-801.828 same-half sum", -801.828, 8 * key.pixel_logZ.sure_vs_sure_opposite + 8 * key.pixel_logZ.loose_zero_vs_loose_zero, 3],
  ["-803.2 same-half log w", -803.2, walk.c["cat-left x dog-left"].logw_unnorm, 1], ["-24.1 side-by-side log w", -24.1, walk.c["cat-left x dog-right"].logw_unnorm, 1],
  ["0.505 noised cat head var", 0.505, key.d_noised_head.cat_var, 3], ["1.0 noised dog head var", 1.0, key.d_noised_head.dog_var, 1],
  ["1.98 precision ratio", 1.98, key.d_noised_head.precision_ratio, 2], ["0.233 summed", 0.233, walk.d.head_mean_product_of_noised, 3],
  ["0.693 target", 0.693, walk.d.head_mean_noised_product, 3], ["0.707 noised means", 0.707, key.d_noised_head.cat_mean, 3],
  ["hand check 0.923", 0.923, key.hand_check.head_mean, 3], ["-815 tight dog head log Z", -815, key.a_dog_head_tight.logZ, 0], ["e^-779 gap", 779, key.c_gap, 0],
  ["2.98 denominator", 2.98, 1 + key.d_noised_head.precision_ratio, 2]];
for (const [n, shown, want, dp] of typed) check("typed " + n, Math.abs(shown - want) <= 0.5 * Math.pow(10, -dp) + 1e-12, `shown ${shown}, key ${want}`);
for (const t of ["99/101 = 0.980", "−803.2", "−24.1", "0.505", "1.98 : 1", "0.233", "0.693", "−98.96", "−2.904", "8 × (−98.962927) + 8 × (−2.904112) = −814.936", "w ≈ ", "(50 − 2) / 52", "0.9801980198019802", "−46.46578749493179"])
  check(`page text carries "${t}"`, html.includes(t) || html.includes(t.replace(/−/g, "-")), "found");
console.log(`${fail ? "FAIL" : "PASS"}  ${total - fail} of ${total} checks`); process.exit(fail ? 1 : 0);
