// Parse every $…$ and $$…$$ block of a markdown file with KaTeX, throwOnError on, and report failures.
import fs from 'node:fs'; import katex from 'katex';
const md = fs.readFileSync(process.argv[2], 'utf8'); let n = 0, bad = [];
const disp = [...md.matchAll(/\$\$([\s\S]+?)\$\$/g)]; const rest = md.replace(/\$\$[\s\S]+?\$\$/g, '');
const inl = [...rest.matchAll(/(?<![\\$])\$([^$\n]+?)\$(?!\$)/g)];
for (const [m, tex] of [...disp.map(d => [d, d[1], true]), ...inl.map(d => [d, d[1], false])].map(([d, t, isD]) => [t, isD])) {
  n++; try { katex.renderToString(m, { displayMode: tex, throwOnError: true, strict: 'ignore' }); } catch (e) { bad.push(e.message.slice(0, 160)); }
}
console.log(`${disp.length} display and ${inl.length} inline equations parsed, ${bad.length} errors`); bad.forEach(b => console.log('  ' + b));
