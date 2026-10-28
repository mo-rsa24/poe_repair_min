// Walk every frame, preset, switch and handle extreme of the scene in jsdom with the real katex package, throwOnError on.
import fs from 'node:fs'; import { JSDOM, VirtualConsole } from 'jsdom'; import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const katexSrc = fs.readFileSync(require.resolve('katex/dist/katex.min.js'), 'utf8');
let html = fs.readFileSync(process.argv[2], 'utf8');
html = html.replace(/<script src="https:\/\/cdnjs[^"]+katex\.min\.js"><\/script>/, () => '<script>window.__STRICT__=true;</script><script>' + katexSrc.replace(/<\/script/g, '<\\/script') + '</script>');
const errors = []; const vc = new VirtualConsole();
vc.on('jsdomError', e => { if (!/Not implemented: HTMLCanvasElement/.test(e.message)) errors.push('jsdom: ' + e.message + (e.detail ? ' ' + e.detail : '')); });
vc.on('error', m => errors.push('console.error: ' + m));
const dom = new JSDOM('<!doctype html><html><head><meta charset="utf-8"></head><body>' + html + '</body></html>', { runScripts: 'dangerously', pretendToBeVisual: true, virtualConsole: vc, url: 'http://localhost/' });
const w = dom.window, S = w.__scene;
if (!S) { console.log('no __scene hook'); process.exit(1); }
S.fake();
const log = []; let presets = 0, switches = 0, handles = 0, steps = 0;
const step = (what, fn) => { steps++; try { fn(); const err = w.document.querySelectorAll('.katex-error').length; if (err) errors.push(`${what}: ${err} katex-error`); } catch (e) { errors.push(`${what}: ${e.message}`); } };
const words = s => s.split(/\s+/).filter(Boolean).length, sentences = s => (s.match(/[.!?](\s|$)/g) || []).length;
for (let i = 0; i < S.n; i++) {
  step(`frame ${i + 1}`, () => S.go(i));
  const cap = S.text().cap; if (words(cap) > 40 || sentences(cap) > 2) errors.push(`frame ${i + 1}: narration ${words(cap)} words, ${sentences(cap)} sentences`);
  const np = S.pres(), ns = S.sw(), keys = [...new Set(S.keys())];
  for (let j = 0; j < np; j++) { step(`frame ${i + 1} preset ${j + 1}`, () => S.applyPre(j)); presets++; }
  for (let j = 0; j < ns; j++) { step(`frame ${i + 1} switch ${j + 1} off`, () => S.toggle(j)); step(`frame ${i + 1} switch ${j + 1} back`, () => S.toggle(j)); switches++; }
  S.go(i);
  for (const k of keys) { const [lo, hi] = S.lim(k); for (const v of [lo, hi, (lo + hi) / 2]) step(`frame ${i + 1} ${k}=${v}`, () => S.set(k, v)); handles++; }
  const t = S.text(); if (/NaN|undefined|Infinity/.test(t.check)) errors.push(`frame ${i + 1}: status reads "${t.check}"`);
  log.push(`frame ${String(i + 1).padStart(2)}: ${np} presets, ${ns} switches, handles [${keys.join(', ')}]`);
}
// wait for queued redraws, then report
await new Promise(r => setTimeout(r, 300));
console.log(log.join('\n'));
console.log(`\nframes walked: ${S.n}; presets exercised: ${presets}; switches toggled both ways: ${switches}; handles driven to min, max and middle: ${handles}; actions: ${steps}`);
console.log(`errors: ${errors.length}`); errors.forEach(e => console.log('  ' + e));
process.exit(errors.length ? 1 : 0);
