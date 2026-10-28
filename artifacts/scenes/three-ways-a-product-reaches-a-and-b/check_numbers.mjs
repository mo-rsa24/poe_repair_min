// Runs the page's own COMPUTE block and checks every slider stop against scene_key.json.
// Usage: node check_numbers.mjs   (exit 1 on any FAIL)
import fs from "node:fs";
const html = fs.readFileSync(new URL("./page.src.html", import.meta.url), "utf8");
const m = html.match(/\/\/ ==== COMPUTE:BEGIN[^\n]*\n([\s\S]*?)\/\/ ==== COMPUTE:END ====/);
const { D, shares } = new Function(m[1] + "\nreturn {D, shares};")();
const KEY = JSON.parse(fs.readFileSync(new URL("./scene_key.json", import.meta.url)));
let fail = 0, n = 0;
const near = (name, got, want) => { n++; const ok = Math.abs(got - want) <= 1e-6 * Math.abs(want) + 1e-12; if (!ok) fail++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}: page ${got.toPrecision(7)}, key ${want.toPrecision(7)}`); };
const cmp = (name, s, r) => { near(name + " TR", s.TR, r.in_TR_blob); near(name + " origin", s.origin, r.at_origin); near(name + " Z", s.Z, r.Z); };
KEY.even.forEach(r => cmp(`even eps=${r.eps}`, shares(D.P2(r.eps, "even")), r));
KEY.adjacent.forEach(r => cmp(`next door eps=${r.eps}`, shares(D.P2(r.eps, "adjacent")), r));
KEY.c1.forEach(r => cmp(`one model r=${r.r}`, shares(D.c1(r.r)), r));
cmp("parts plain", shares(D.bPlain()), KEY.b_plain);
cmp("parts divided", shares(D.bDiv()), KEY.b_divided);
["BR", "TR", "TL"].forEach(q => cmp(`B's blob on ${q}`, shares(D.pin(q)), KEY.support[q]));
console.log(`${n - fail} of ${n} pass`); process.exit(fail ? 1 : 0);
