// Drive every frame, preset and switch in real Chromium with the real photos, and print each status line.
import { createRequire } from 'node:module'; import path from 'node:path'; import os from 'node:os'; import http from 'node:http'; import fs from 'node:fs';
const req = createRequire(path.join(os.homedir(), '.cache/learning-render', 'package.json')); const { chromium } = req('playwright-core');
const file = path.resolve(process.argv[2]); const body = '<!doctype html><meta charset="utf-8">' + fs.readFileSync(file, 'utf8');
const srv = http.createServer((q, r) => { r.writeHead(200, { 'content-type': 'text/html; charset=utf-8' }); r.end(body); }); await new Promise(ok => srv.listen(0, '127.0.0.1', ok));
const browser = await chromium.launch({ executablePath: process.env.CHROME }); const page = await browser.newPage({ viewport: { width: 1440, height: 860 } });
const errs = []; page.on('pageerror', e => errs.push(String(e.message))); page.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
await page.goto(`http://127.0.0.1:${srv.address().port}/`); await page.waitForTimeout(1500);
const n = await page.evaluate(() => window.__scene.n); let acts = 0, kerr = 0; const overflow = [];
for (let i = 0; i < n; i++) {
  const rows = await page.evaluate((i) => { const S = window.__scene, out = []; const k = () => document.querySelectorAll('.katex-error').length; let e = 0; const ov = () => { const w = [...document.querySelectorAll('.eq')].filter(q => q.offsetParent).map(q => q.scrollWidth - q.clientWidth); const m = Math.max(0, ...w); return m > 2 ? m : 0; }; const O = [];
    S.go(i); out.push(['default', S.text().check]); e += k(); { const o = ov(); if (o) O.push('default +' + o + 'px'); }
    for (let j = 0; j < S.pres(); j++) { S.applyPre(j); out.push(['preset ' + (j + 1) + ' ' + document.querySelectorAll('#fPres button')[j].textContent, S.text().check]); e += k(); if (ov()) O.push('preset ' + (j + 1)); }
    for (let j = 0; j < S.sw(); j++) { S.go(i); S.toggle(j); out.push(['switch ' + (j + 1) + ' flipped', S.text().check]); e += k(); if (ov()) O.push('switch ' + (j + 1)); }
    return { out, e, O }; }, i);
  kerr += rows.e; acts += rows.out.length; if (rows.O.length) overflow.push(`frame ${i + 1}: ${rows.O.join(', ')}`);
  console.log(`\nframe ${i + 1}`); rows.out.forEach(([a, t]) => console.log(`  ${a}: ${t}`));
}
console.log(`\nbrowser actions: ${acts}; katex errors: ${kerr}; page errors: ${errs.length}; equations wider than their card: ${overflow.length} frames`); overflow.forEach(o => console.log('  overflow ' + o)); errs.forEach(e => console.log('  ' + e));
await browser.close(); srv.close();
