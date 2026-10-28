/* ===== scene 1: one number ===== */
handle('mu',150,195,0.1,'bell centre μ',1);handle('sg',0.05,25,0.05,'bell width σ',2);handle('bw',1,15,1,'bin width');
handle('xq',140,205,0.1,'height x',1);handle('w',0,30,0.5,'strip width w',1);handle('u',0.001,0.999,0.001,'random number u',3);handle('nd',1,2000,1,'number of draws');
BIG.nd=10;
const P1=PAGE[1];

/* bell over the heights, with a stem from each dot up to the curve */
function bellStems(id,mu,sg,o){o=o||{};const W=o.W||600,Hh=320,m={l:52,r:20,t:30,b:52},lo=145,hi=200,sx=lin(lo,hi,m.l,W-m.r);
  const peak=Math.max(npdf(mu,mu,sg),0.06),ym=Math.min(o.ymax||0.2,Math.max(0.07,peak*1.08)),sy=v=>Hh-m.b-Math.min(v,ym)/ym*(Hh-m.t-m.b);
  let s=vAxis(sy,m.l,W-m.r,niceTicks(ym),v=>f2(v),'BELL HEIGHT, PER CM',m.t-12);
  if(o.hist){const bw=5;for(let a=145;a<200;a+=bw){const c=HTS.filter(h=>h>=a&&h<a+bw).length;if(!c)continue;s+=`<rect x="${sx(a)}" y="${sy(c/(NH*bw))}" width="${sx(a+bw)-sx(a)}" height="${Hh-m.b-sy(c/(NH*bw))}" fill="var(--x-soft)" stroke="var(--panel)"/>`;}}
  if(o.ghost)s+=`<path d="${curve(v=>npdf(v,MU0,SG0),lo,hi,sx,sy)}" fill="none" stroke="var(--muted)" stroke-width="1.8" stroke-dasharray="6 5"/>`;
  s+=`<path d="${curve(v=>npdf(v,mu,sg),lo,hi,sx,sy,400)}" fill="none" stroke="var(--p)" stroke-width="2.8"/>`;
  HTS.forEach(h=>{const on=o.all!==false||h===ME;const y=sy(npdf(h,mu,sg));s+=`<line x1="${sx(h)}" x2="${sx(h)}" y1="${Hh-m.b}" y2="${y}" stroke="${on?'var(--x)':'var(--rule)'}" stroke-width="2.2"/>`;
    s+=`<circle cx="${sx(h)}" cy="${Hh-m.b}" r="${h===ME?6.5:4.5}" fill="${on?'var(--x)':'var(--rule)'}"/>`;});
  s+=hAxis(sx,150,200,Hh-m.b+8,10,'HEIGHT, CM');$(id).innerHTML=s;}
/* a function of one parameter, its current point, and the tangent there */
function slopeChart(id,f,lo,hi,cur,o){const W=440,Hh=320,m={l:52,r:20,t:30,b:52},sx=lin(lo,hi,m.l,W-m.r);
  const vs=[];for(let i=0;i<=200;i++)vs.push(f(lo+(hi-lo)*i/200));const top=Math.max(...vs),bot=Math.max(Math.min(...vs),top-45),yl=Math.floor(bot/10)*10,yh=Math.ceil((top+3)/10)*10;
  const sy=v=>Hh-m.b-(Math.max(yl,Math.min(yh,v))-yl)/(yh-yl)*(Hh-m.t-m.b);const tk=[];for(let v=yl;v<=yh;v+=10)tk.push(v);
  let s=vAxis(sy,m.l,W-m.r,tk,v=>mm(v),o.yt,m.t-12);const xs=[],ys=[];for(let i=0;i<=200;i++){const v=lo+(hi-lo)*i/200,l=f(v);if(l<yl)continue;xs.push(sx(v));ys.push(sy(l));}
  s+=`<path d="${pth(xs,ys)}" fill="none" stroke="var(--ab)" stroke-width="2.8"/>`;
  const v=f(cur),h=(hi-lo)*0.0005,sl=(f(cur+h)-f(cur-h))/(2*h),d=(hi-lo)*0.12;
  s+=`<line x1="${sx(cur-d)}" x2="${sx(cur+d)}" y1="${sy(v-sl*d)}" y2="${sy(v+sl*d)}" stroke="var(--nz)" stroke-width="2.4"/>`;
  s+=`<circle cx="${sx(cur)}" cy="${sy(v)}" r="6" fill="var(--x)"/>`;const right=sx(cur)<W-150;
  s+=txt(sx(cur)+(right?10:-10),Math.min(Hh-m.b-8,sy(v)+22),'slope '+mm(f3(sl)),{c:'num',a:right?'start':'end',s:'fill:var(--nz)'});
  s+=hAxis(sx,lo,hi,Hh-m.b+8,o.step,o.xt);$(id).innerHTML=s;}
function llChart(id,mu,sg,all,both){const W=440,Hh=320,m={l:52,r:24,t:30,b:52},lo=0.05,hi=20,sx=lin(0,hi,m.l,W-m.r),yl=-70,yh=5,sy=v=>Hh-m.b-(Math.max(yl,Math.min(yh,v))-yl)/(yh-yl)*(Hh-m.t-m.b);
  let s=vAxis(sy,m.l,W-m.r,[-60,-40,-20,0],v=>mm(v),'SCORE, LOG L',m.t-12);
  const draw=(a,col,w,dash)=>{const xs=[],ys=[];for(let i=0;i<=300;i++){const v=lo+(hi-lo)*i/300,l=logL(mu,v,a);if(l<yl)continue;xs.push(sx(v));ys.push(sy(l));}return `<path d="${pth(xs,ys)}" fill="none" stroke="${col}" stroke-width="${w}"${dash?' stroke-dasharray="6 5"':''}/>`;};
  if(both)s+=draw(!all,'var(--muted)',1.8,true);
  s+=draw(all,all?'var(--ab)':'var(--hot)',2.8);
  const v=logL(mu,sg,all);s+=`<line x1="${sx(sg)}" x2="${sx(sg)}" y1="${m.t}" y2="${Hh-m.b}" stroke="var(--x)" stroke-width="1.4"/><circle cx="${sx(sg)}" cy="${sy(v)}" r="6" fill="var(--x)"/>`;
  const right=sx(sg)<W-120;s+=txt(sx(sg)+(right?10:-10),Math.max(m.t+14,sy(v)-10),mm(f2(v)),{c:'num',a:right?'start':'end',s:'fill:var(--x)'});
  s+=hAxis(sx,0,20,Hh-m.b+8,5,'BELL WIDTH σ, CM');$(id).innerHTML=s;}
function bins(bw){const out=[];for(let a=145;a<200;a+=bw)out.push({a,n:HTS.filter(h=>h>=a&&h<a+bw).length});return out;}
let gridC=null;function grid(){if(gridC)return gridC;let best={v:-Infinity,n:0};for(let mu=160;mu<=185.0001;mu+=0.05)for(let sg=2;sg<=20.0001;sg+=0.05){best.n++;const v=logL(mu,sg,true);if(v>best.v){best.v=v;best.mu=mu;best.sg=sg;}}return gridC=best;}
let drawC={};function uDraws(){const k=P.nd+'|'+P.seed;if(drawC.k!==k){const r=rng(P.seed*31+5);drawC={k,u:[...Array(P.nd)].map(()=>Math.min(0.999999,Math.max(1e-6,r())))};}return drawC.u;}
const draws=()=>uDraws().map(Finv);
const W1='600';

const FR1=[
/* 1 */{title:'One height is one sample',lit:['x'],
 say:'Ten people, ten heights in centimetres, all example values. Each dot is one sample, and the letter x names a sample, so yours is x = 173 cm.',
 note:'The ten heights are example values chosen for this series, with 173 cm included as yours. They are not measurements of real people.',
 eq:()=>{const L=HTS.map(h=>h===ME?cl('c-x','\\mathbf{173}'):h);return `\\begin{aligned}\\{x_1,\\dots,x_{10}\\}=\\{&${L.slice(0,5).join(',\\,')},\\\\&${L.slice(5).join(',\\,')}\\}\\ \\text{cm}\\end{aligned}`;},
 eqT:()=>`\\text{one sample: }${cl('c-x','x_7')}=${cl('c-x','173')}\\text{ cm}`,
 check:()=>`Shortest ${Math.min(...HTS)} cm, tallest ${Math.max(...HTS)} cm. ${HTS.filter(h=>h>ME).length} of the 10 are taller than you.`,
 fig:el=>{el.innerHTML=`<div class="card">${SVG('g',640,230,'Ten heights as dots on a ruler, yours lit')}</div>`;},
 draw(){const W=640,sx=lin(145,195,40,W-30);let s=hAxis(sx,145,195,150,5,'HEIGHT, CM');
   HTS.forEach(h=>{const me=h===ME;s+=`<circle cx="${sx(h)}" cy="128" r="${me?10:7}" fill="${me?'var(--x)':'var(--x-soft)'}" stroke="var(--x)" stroke-width="2"/>`;});
   s+=`<line x1="${sx(ME)}" x2="${sx(ME)}" y1="70" y2="114" stroke="var(--x)" stroke-width="1.5"/>`+txt(sx(ME),60,'you, 173',{a:'middle',s:'fill:var(--x);font-weight:600'});$('g').innerHTML=s;}},
/* 2 */{title:'The pile has a middle and a spread',lit:['x','pile','data'],
 say:'Count the heights in bins and the pile shows its shape. The mean is the average height, and the variance is the average squared distance from that mean.',
 note:'Mean and variance here divide by 10, the number of heights. Dividing by 9 instead gives 60.18, the unbiased variance; frame 9 shows which one the best bell picks.',
 def:{bw:5},pres:[['2 cm bins',{bw:2}],['10 cm bins',{bw:10}]],
 eq:()=>`\\begin{aligned}\\bar x&=\\tfrac{1}{10}\\textstyle\\sum_{i=1}^{10}x_i=\\tfrac{${SUM}}{10}=${f1(MU0)}\\text{ cm}\\\\s^2&=\\tfrac{1}{10}\\textstyle\\sum_{i=1}^{10}(x_i-\\bar x)^2=\\tfrac{${f1(SS0)}}{10}=${f2(VAR0)}\\text{ cm}^2,\\quad s=${f2(SG0)}\\text{ cm}\\end{aligned}`,
 eqT:()=>`\\text{bin width}=${H('bw')}\\text{ cm}`,
 check(){const c=bins(P.bw);return `Bin width ${P.bw} cm: the tallest bar holds ${Math.max(...c.map(b=>b.n))} of the 10 heights. The bins change the picture; the mean and variance stay the same.`;},
 fig:el=>{el.innerHTML=`<div class="card">${SVG('g',640,330,'Histogram of the ten heights with the mean and one spread either side')}</div>`;},
 draw(){const W=640,Hh=330,m={l:46,r:24,t:56,b:60},sx=lin(145,200,m.l,W-m.r),c=bins(P.bw),ym=Math.max(4,...c.map(b=>b.n)),sy=v=>Hh-m.b-v/ym*(Hh-m.t-m.b);
   let s=vAxis(sy,m.l,W-m.r,[...Array(ym+1).keys()].filter(v=>ym<=6||v%2===0),v=>v,'COUNT',m.t-30);
   c.forEach(b=>{if(b.n)s+=`<rect x="${sx(b.a)+1}" y="${sy(b.n)}" width="${Math.max(1,sx(b.a+P.bw)-sx(b.a)-2)}" height="${Hh-m.b-sy(b.n)}" fill="var(--x-soft)" stroke="var(--x)"/>`;});
   HTS.forEach(h=>{s+=`<circle cx="${sx(h)}" cy="${Hh-m.b+10}" r="4" fill="var(--x)"/>`;});
   s+=`<line x1="${sx(MU0)}" x2="${sx(MU0)}" y1="${m.t-14}" y2="${Hh-m.b}" stroke="var(--ink)" stroke-width="2" stroke-dasharray="5 4"/>`+txt(sx(MU0),m.t-20,'mean '+f1(MU0),{a:'middle',c:'num'});
   const yb=m.t+4;s+=`<line x1="${sx(MU0-SG0)}" x2="${sx(MU0+SG0)}" y1="${yb}" y2="${yb}" stroke="var(--nz)" stroke-width="3"/><line x1="${sx(MU0-SG0)}" x2="${sx(MU0-SG0)}" y1="${yb-6}" y2="${yb+6}" stroke="var(--nz)" stroke-width="2"/><line x1="${sx(MU0+SG0)}" x2="${sx(MU0+SG0)}" y1="${yb-6}" y2="${yb+6}" stroke="var(--nz)" stroke-width="2"/>`;
   s+=txt(sx(MU0+SG0)+8,yb+4,'s = '+f2(SG0),{c:'num',s:'fill:var(--nz)'});s+=hAxis(sx,145,200,Hh-m.b+20,5,'HEIGHT, CM');$('g').innerHTML=s;}},
/* 3 */{title:'Lay a bell over the pile',lit:['data','fit','p'],
 say:"Drag μ to slide the bell and σ to widen it. The score adds the log of the bell's height over each of the 10 dots, so a bell that stands tall over every dot scores best.",
 note:'The dashed bell is the best one (μ = 170.8, σ = 7.36), drawn so every change is read against it. Bars are the 5 cm histogram rescaled so their total area is 1.',
 def:{mu:180,sg:4},pres:[['Too narrow',{mu:MU0,sg:3}],['Too wide',{mu:MU0,sg:15}],['Off centre',{mu:180,sg:SG0}],['Best bell',{mu:MU0,sg:SG0}]],
 eq:()=>`\\text{centre }\\mu=${H('mu')}\\text{ cm},\\qquad\\text{width }\\sigma=${H('sg')}\\text{ cm}`,
 eqT:()=>`\\text{score}=\\sum_{i=1}^{10}\\log\\big(\\text{bell height at }${cl('c-x','x_i')}\\big)=${mm(f2(logL(P.mu,P.sg,true)))}`,
 check(){const g=BEST-logL(P.mu,P.sg,true);return g<0.005?`This is the best bell: no other μ and σ score higher than ${mm(f2(BEST))}.`:`Score ${mm(f2(BEST-g))}. The best bell scores ${mm(f2(BEST))}, so this one is ${f2(g)} below it.`;},
 fig:el=>{el.innerHTML=`<div class="card">${SVG('g',600,320,'A bell curve over the ten heights, a stem from each dot up to the curve')}</div>`;},
 draw(){bellStems('g',P.mu,P.sg,{hist:true,ghost:true,ymax:0.15});}},
/* 4 */{title:"The bell's height is a density",lit:['p'],
 say:'Call the best bell p(x). Its height is measured per centimetre, so it is a density, telling how thickly heights pile up near x, and never a chance on its own.',
 note:'Area is summed in 0.01 cm strips from 100 cm to 250 cm. The best bell uses μ = 170.8 cm and σ = 7.36 cm from frame 2.',
 def:{xq:173},pres:[['At the mean',{xq:+MU0.toFixed(1)}],['200 cm',{xq:200}]],
 eq(){const q=(P.xq-MU0)**2/(2*VAR0),c0=1/(SG0*R(2*PI));return `${cl('c-p','p(x)')}=\\underbrace{\\frac{1}{\\sigma\\sqrt{2\\pi}}}_{${f4(c0)}}\\exp\\!\\Big(-\\underbrace{\\frac{(x-\\mu)^2}{2\\sigma^2}}_{${f3(q)}}\\Big)=${cl('c-p',tsci(npdf(P.xq,MU0,SG0)))}\\text{ per cm}`;},
 eqT:()=>`${cl('c-x','x')}=${H('xq')}\\text{ cm},\\qquad\\mu=${f1(MU0)},\\ \\sigma=${f2(SG0)}`,
 check(){let a=0;for(let x=100;x<250;x+=0.01)a+=npdf(x+0.005,MU0,SG0)*0.01;return `Total area under p: ${f4(a)}. The front factor 1/(σ√2π) = ${f4(1/(SG0*R(2*PI)))} is what makes it 1; squeeze σ to 0.1 cm and the peak reads ${f2(npdf(0,0,0.1))} per cm, with the area still 1.`;},
 fig:el=>{el.innerHTML=`<div class="card">${SVG('g',600,320,'The best bell with a marker at the chosen height and its density value')}</div>`;},
 draw(){const W=600,Hh=320,m={l:56,r:20,t:30,b:52},sx=lin(140,205,m.l,W-m.r),ym=0.06,sy=v=>Hh-m.b-v/ym*(Hh-m.t-m.b),x=P.xq,p=npdf(x,MU0,SG0);
   let s=vAxis(sy,m.l,W-m.r,[0,0.02,0.04,0.06],v=>f2(v),'DENSITY, PER CM',m.t-12);
   s+=`<path d="${curve(v=>npdf(v,MU0,SG0),140,205,sx,sy)}" fill="none" stroke="var(--p)" stroke-width="2.8"/>`;
   s+=`<line x1="${sx(x)}" x2="${sx(x)}" y1="${Hh-m.b}" y2="${sy(p)}" stroke="var(--x)" stroke-width="2.4"/><circle cx="${sx(x)}" cy="${sy(p)}" r="6" fill="var(--p)"/>`;
   const right=sx(x)<W-220;s+=txt(sx(x)+(right?12:-12),sy(p)-10,'p = '+(p<1e-3?sci(p):f4(p))+' per cm',{c:'num',a:right?'start':'end',s:'fill:var(--p)'});
   s+=hAxis(sx,140,200,Hh-m.b+8,10,'HEIGHT, CM');$('g').innerHTML=s;}},
/* 5 */{title:'A chance is an area',lit:['p'],
 say:'The chance of landing in a range is the area under p over that range. A narrow strip has area close to height times width, so one exact height has area zero.',
 note:'The area is F(b) − F(a), with F the running area of frame 11. The count of people in the strip comes from the ten example heights.',
 def:{xq:173,w:2},pres:[['1 cm strip',{w:1}],['10 cm strip',{w:10}],['Width 0',{w:0}]],
 eq(){const a=P.xq-P.w/2,b=P.xq+P.w/2,ar=ncdf(b,MU0,SG0)-ncdf(a,MU0,SG0),p=npdf(P.xq,MU0,SG0);return `P(a\\le X\\le b)=\\int_a^b ${cl('c-p','p(x)')}\\,dx=${f4(ar)},\\qquad ${cl('c-p','p(x)')}\\cdot w=${f4(p*P.w)}`;},
 eqT:()=>`${cl('c-x','x')}=${H('xq')}\\text{ cm},\\quad w=${H('w')}\\text{ cm},\\quad a=${f2(P.xq-P.w/2)},\\ b=${f2(P.xq+P.w/2)}`,
 check(){const a=P.xq-P.w/2,b=P.xq+P.w/2,ar=ncdf(b,MU0,SG0)-ncdf(a,MU0,SG0);if(P.w===0)return `Width 0: area 0. A single exact height has chance zero, even though p(${f1(P.xq)}) = ${f4(npdf(P.xq,MU0,SG0))} per cm.`;
   const k=HTS.filter(h=>h>=a&&h<=b).length;return `${k} of the 10 people ${k===1?'falls':'fall'} between ${f1(a)} and ${f1(b)} cm, a share of ${f1(k/10)}. The bell gives ${f3(ar)}.`;},
 fig:el=>{el.innerHTML=`<div class="card">${SVG('g',600,320,'The best bell with a shaded strip whose area is the chance')}</div>`;},
 draw(){const W=600,Hh=320,m={l:56,r:20,t:30,b:52},sx=lin(140,205,m.l,W-m.r),ym=0.06,sy=v=>Hh-m.b-v/ym*(Hh-m.t-m.b),a=Math.max(140,P.xq-P.w/2),b=Math.min(205,P.xq+P.w/2);
   let s=vAxis(sy,m.l,W-m.r,[0,0.02,0.04,0.06],v=>f2(v),'DENSITY, PER CM',m.t-12);
   if(b>a)s+=`<path d="${area(v=>npdf(v,MU0,SG0),a,b,sx,sy,80)}" fill="var(--p-soft)" stroke="var(--p)" stroke-width="1"/>`;
   s+=`<path d="${curve(v=>npdf(v,MU0,SG0),140,205,sx,sy)}" fill="none" stroke="var(--p)" stroke-width="2.8"/>`;
   const ar=ncdf(P.xq+P.w/2,MU0,SG0)-ncdf(P.xq-P.w/2,MU0,SG0);s+=txt(sx(P.xq),m.t+6,'area '+f4(ar),{c:'num',a:'middle',s:'fill:var(--p)'});
   HTS.forEach(h=>{const inn=h>=P.xq-P.w/2&&h<=P.xq+P.w/2;s+=`<circle cx="${sx(h)}" cy="${Hh-m.b+38}" r="${inn?6:4}" fill="${inn?'var(--x)':'var(--x-soft)'}" stroke="var(--x)"/>`;});
   s+=hAxis(sx,140,200,Hh-m.b+4,10,'');s+=txt(W-m.r,Hh-4,'HEIGHT, CM',{c:'ax',a:'end'});$('g').innerHTML=s;}},
/* 6 */{title:'One formula, read two ways',lit:['p','fit'],flip:true,
 say:'Fix μ and let x move, and the formula is a density over heights. Fix your height at 173 and let μ move, and the same formula is a likelihood, scoring each possible centre.',
 note:'For one height and a Gaussian, the likelihood curve over μ is a bell centred at 173, because the formula is symmetric in x and μ. With ten heights the likelihood is a product of ten such curves and no longer has area 1 over μ, which is why a likelihood is a score for parameters and never a density for them.',
 def:{xq:173,mu:MU0,read:false},pres:[['x = 173, μ = 170.8',{xq:173,mu:MU0}],['x = 180',{xq:180}],['μ = 180',{read:true,mu:180}]],
 sw:[{k:'read',on:'Reading: likelihood, data fixed, μ moves',off:'Reading: density, μ fixed, x moves'}],
 eq(){const x=P.read?cl('c-mu','173'):H('xq'),mu=P.read?H('mu'):cl('c-mu',f1(MU0)),mv='\\text{moves}',fx='\\text{fixed}';
   return `p\\big(\\underbrace{${x}}_{x,\\ ${P.read?fx:mv}};\\ \\underbrace{${mu}}_{\\mu,\\ ${P.read?mv:fx}},\\ \\underbrace{${cl('c-mu',f2(SG0))}}_{\\sigma,\\ ${fx}}\\big)=${cl('c-p',f4(P.read?npdf(173,P.mu,SG0):npdf(P.xq,MU0,SG0)))}`;},
 eqT:()=>`p(x;\\mu,\\sigma)=\\frac{1}{\\sigma\\sqrt{2\\pi}}\\exp\\!\\Big(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\Big)`,
 check(){return P.read?`Read as a likelihood: the height 173 stays fixed and the curve runs over μ, peaking at μ = 173. At μ = ${f1(P.mu)} it reads ${f4(npdf(173,P.mu,SG0))}.`:`Read as a density: μ = ${f1(MU0)} stays fixed and the curve runs over heights, with area 1. At x = ${f1(P.xq)} it reads ${f4(npdf(P.xq,MU0,SG0))} per cm.`;},
 fig:el=>{el.innerHTML=`<div class="card">${SVG('g',600,320,'One formula drawn as a curve over heights or as a curve over the bell centre')}</div>`;},
 draw(){const W=600,Hh=320,m={l:56,r:20,t:30,b:56},sx=lin(140,205,m.l,W-m.r),ym=0.06,sy=v=>Hh-m.b-v/ym*(Hh-m.t-m.b);
   let s=vAxis(sy,m.l,W-m.r,[0,0.02,0.04,0.06],v=>f2(v),'VALUE OF THE FORMULA',m.t-12);
   const dens=v=>npdf(v,MU0,SG0),lik=v=>npdf(173,v,SG0),on=P.read?lik:dens,off=P.read?dens:lik;
   s+=`<path d="${curve(off,140,205,sx,sy)}" fill="none" stroke="var(--muted)" stroke-width="1.6" stroke-dasharray="5 5"/>`;
   s+=`<path d="${curve(on,140,205,sx,sy)}" fill="none" stroke="${P.read?'var(--ab)':'var(--p)'}" stroke-width="2.8"/>`;
   const cur=P.read?P.mu:P.xq,v=on(cur);s+=`<line x1="${sx(cur)}" x2="${sx(cur)}" y1="${Hh-m.b}" y2="${sy(v)}" stroke="var(--x)" stroke-width="2"/><circle cx="${sx(cur)}" cy="${sy(v)}" r="6" fill="var(--x)"/>`;
   const sp=P.read?MU0:173;s+=`<circle cx="${sx(sp)}" cy="${sy(npdf(173,MU0,SG0))}" r="9" fill="none" stroke="var(--nz)" stroke-width="2"/>`;
   const right=sx(cur)<W-170;s+=txt(sx(cur)+(right?12:-12),Math.max(m.t+12,sy(v)-12),f4(v),{c:'num',a:right?'start':'end',s:'fill:var(--x)'});
   s+=hAxis(sx,140,200,Hh-m.b+8,10,P.read?'BELL CENTRE μ, CM':'HEIGHT x, CM');$('g').innerHTML=s;}},
/* 7 */{title:'Likelihood multiplies all ten heights',lit:['data','fit'],
 say:'Likelihood multiplies p at every sample, so one badly covered dot drags the whole product down. Taking the log turns the product of ten into a sum of ten.',
 note:'Left: the bell and a stem up to it from each height. Right: the score log L against the width σ, with the centre μ held at its current value.',
 def:{mu:MU0,sg:SG0,all:true},pres:[['σ = 3',{sg:3}],['μ = 180',{mu:180}],['Best bell',{mu:MU0,sg:SG0}]],
 sw:[{k:'all',on:'Using all 10 heights',off:'Using only your 173 cm',warnOff:true}],
 eq(){const ll=logL(P.mu,P.sg,P.all);if(P.all)return `\\begin{aligned}L(\\mu,\\sigma)&=\\prod_{i=1}^{10}${cl('c-p','p(x_i;\\mu,\\sigma)')}=${tsci(Math.exp(ll))}\\\\\\log L&=\\sum_{i=1}^{10}\\log ${cl('c-p','p(x_i;\\mu,\\sigma)')}=${mm(f2(ll))}\\end{aligned}`;
   return `\\log L=\\log ${cl('c-p','p(173;\\mu,\\sigma)')}+${cl('c-hot','\\cancel{\\textstyle\\sum_{i\\ne7}\\log p(x_i;\\mu,\\sigma)}')}=${mm(f2(ll))}`;},
 eqT:()=>`\\mu=${H('mu')}\\text{ cm},\\qquad\\sigma=${H('sg')}\\text{ cm}`,
 check(){if(!P.all)return 'One term left: the score ignores nine of the ten people.';const t=HTS.map(h=>lpdf(h,P.mu,P.sg)),i=t.indexOf(Math.min(...t));return `The lowest of the ten terms is ${mm(f2(t[i]))}, at ${HTS[i]} cm, the dot this bell covers worst.`;},
 bad:()=>!P.all,
 fig:el=>{el.innerHTML=`<div class="pair"><div class="card">${SVGS('gl',440,320,'The bell over the ten heights')}</div><div class="card">${SVGS('gr',440,320,'Score against bell width')}</div></div>`;},
 draw(){bellStems('gl',P.mu,P.sg,{all:P.all,ymax:0.15,W:440});llChart('gr',P.mu,P.sg,P.all,false);}},
/* 8 */{title:'The best centre is the average',lit:['fit','p'],
 say:'Slide μ until the tangent on the score lies flat. The slope is the sum of the ten distances xᵢ − μ, so it is zero exactly where μ is the average height.',
 note:'The tangent is the slope of log L against μ at the current μ, with σ held at 7.36 cm. Setting the slope to zero and solving gives Σ(x_i − μ) = 0, so μ = (1/10)Σx_i.',
 def:{mu:177,sg:SG0},pres:[['μ = x̄',{mu:MU0}],['μ at the median, 171',{mu:171}],['μ = 160',{mu:160}]],
 eq(){const d=HTS.reduce((a,h)=>a+(h-P.mu),0);return `\\frac{\\partial\\log L}{\\partial\\mu}=\\sum_{i=1}^{10}\\frac{x_i-\\mu}{\\sigma^2}=\\frac{${mm(f1(d))}}{${f2(VAR0)}}=${mm(f3(d/P.sg**2))}`;},
 eqT:()=>`\\mu=${H('mu')}\\text{ cm},\\qquad \\text{zero at }\\mu=${lk(P1+'#f2','\\bar x')}=${f1(MU0)}`,
 check(){const d=HTS.reduce((a,h)=>a+(h-P.mu),0);return Math.abs(d)<0.05?`Slope 0 at μ = ${f1(P.mu)} cm, the average height: the ten distances cancel.`:`The ten distances add to ${mm(f1(d))} cm, so the score still ${d>0?'rises':'falls'} as μ grows. Score here ${mm(f2(logL(P.mu,SG0,true)))}, best ${mm(f2(BEST))}.`;},
 fig:el=>{el.innerHTML=`<div class="pair"><div class="card">${SVGS('gl',440,320,'The bell over the ten heights')}</div><div class="card">${SVGS('gr',440,320,'Score against the bell centre with the tangent at the current centre')}</div></div>`;},
 draw(){bellStems('gl',P.mu,SG0,{ymax:0.15,ghost:true,W:440});slopeChart('gr',v=>logL(v,SG0,true),155,190,P.mu,{yt:'SCORE, LOG L',xt:'BELL CENTRE μ, CM',step:5});}},
/* 9 */{title:'The best width is the spread',lit:['fit','p'],
 say:'Now slide σ with μ at the average. The tangent lies flat where σ² equals the average squared distance, the variance the pile gave, so fitting by likelihood returns the two summary numbers.',
 note:'Grid search: μ from 160 to 185 cm and σ from 2 to 20 cm, both in 0.05 cm steps, every bell scored on all ten heights. Dividing by 9 instead of 10 gives σ = 7.76 cm, which scores lower.',
 def:{mu:MU0,sg:11},pres:[['σ = s',{sg:SG0}],['σ dividing by 9',{sg:R(SS0/9)}],['σ = 4',{sg:4}]],
 eq(){const d=-NH/P.sg+SS0/P.sg**3;return `\\frac{\\partial\\log L}{\\partial\\sigma}=-\\frac{10}{\\sigma}+\\sum_{i=1}^{10}\\frac{(x_i-\\bar x)^2}{\\sigma^3}=${mm(f3(-NH/P.sg))}+${f3(SS0/P.sg**3)}=${mm(f3(d))}`;},
 eqT:()=>`\\sigma=${H('sg')}\\text{ cm},\\qquad\\text{zero at }\\sigma^2=${lk(P1+'#f2','s^2')}=${f2(VAR0)}`,
 check(){const g=grid();return `Searching ${comma(g.n)} bells finds the top at μ = ${f2(g.mu)}, σ = ${f2(g.sg)} (the 0.05 grid step nearest 7.36), score ${mm(f2(g.v))}.`;},
 fig:el=>{el.innerHTML=`<div class="pair"><div class="card">${SVGS('gl',440,320,'The bell over the ten heights')}</div><div class="card">${SVGS('gr',440,320,'Score against the bell width with the tangent at the current width')}</div></div>`;},
 draw(){bellStems('gl',MU0,P.sg,{ymax:0.15,ghost:true,W:440});slopeChart('gr',v=>logL(MU0,v,true),2,20,P.sg,{yt:'SCORE, LOG L',xt:'BELL WIDTH σ, CM',step:5});}},
/* 10 */{title:'One height alone shrinks σ to zero',lit:['x','fit'],
 say:'Score only your own 173 cm and every narrower bell scores higher, so the best fit is a spike of zero width. That is why fitting needs all ten heights.',
 note:'Journey 03 found the same by simulation: with one sample the fitted variance was 0 in every trial (mle_var_n1_mean = 0.0 in generative-models/03-sampling-from-a-graph/artifacts/substrate/facts.json).',
 def:{mu:173,sg:2,all:false},pres:[['σ = 1',{sg:1}],['σ = 0.1',{sg:0.1}],['Use all 10',{all:true}]],
 sw:[{k:'all',on:'Using all 10 heights',off:'Using only your 173 cm',warnOff:true}],
 eq(){const v=logL(P.mu,P.sg,P.all);if(!P.all)return `\\begin{aligned}\\log L&=\\log ${cl('c-p','p(173;\\mu,\\sigma)')}\\\\&=-\\log\\big(\\sigma\\sqrt{2\\pi}\\big)-\\frac{(173-\\mu)^2}{2\\sigma^2}=${mm(f2(v))}\\end{aligned}`;
   return `\\log L=\\sum_{i=1}^{10}\\log ${cl('c-p','p(x_i;\\mu,\\sigma)')}=${mm(f2(v))}`;},
 eqT:()=>`\\mu=${H('mu')}\\text{ cm},\\qquad\\sigma=${H('sg')}\\text{ cm}`,
 check(){if(P.all)return `With all ten heights the score peaks at σ = ${f2(SG0)} and falls on both sides, so the spike is gone.`;
   const v=logL(P.mu,P.sg,false),v2=logL(P.mu,P.sg/2,false);return `At σ = ${f2(P.sg)} the score is ${mm(f2(v))}; halve σ and it rises to ${mm(f2(v2))}. The score has no top above σ = 0.`;},
 bad:()=>!P.all,
 fig:el=>{el.innerHTML=`<div class="pair"><div class="card">${SVGS('gl',440,320,'The bell over your height alone')}</div><div class="card">${SVGS('gr',440,320,'Score against bell width for one height and for ten')}</div></div>`;},
 draw(){bellStems('gl',P.mu,P.sg,{all:P.all,ymax:0.25,W:440});llChart('gr',P.mu,P.sg,P.all,true);}},
/* 11 */{title:'The running area is the CDF',lit:['p','area','F'],
 say:'Sweep from the far left, keeping a running total of the area under p. Call the running total F(x); it climbs from 0 to 1, and its slope at any x is p(x).',
 note:'F is computed from the error function (Abramowitz and Stegun 7.1.26, error below 1.5 × 10⁻⁷). The slope check differentiates F numerically over ±0.01 cm.',
 def:{xq:173},pres:[['x = μ',{xq:+MU0.toFixed(1)}],['x = 190',{xq:190}]],
 eq:()=>`F(${cl('c-x','x')})=\\int_{-\\infty}^{x}${cl('c-p','p(s)')}\\,ds=${f4(ncdf(P.xq,MU0,SG0))}`,
 eqT:()=>`${cl('c-x','x')}=${H('xq')}\\text{ cm}`,
 check:()=>{const k=HTS.filter(h=>h<=P.xq).length,sl=(ncdf(P.xq+0.01,MU0,SG0)-ncdf(P.xq-0.01,MU0,SG0))/0.02;return `${k} of the 10 people are at or below ${f1(P.xq)} cm, a share of ${f1(k/10)}. The slope of F here is ${f4(sl)} per cm, equal to p(${f1(P.xq)}).`;},
 fig:el=>{el.innerHTML=`<div class="card">${SVG('g',600,450,'The bell with the area left of x shaded, and below it the running area F')}</div>`;},
 draw(){const W=600,m={l:56,r:20},sx=lin(140,205,m.l,W-m.r),x=P.xq,F=ncdf(x,MU0,SG0);
   const sy1=v=>170-v/0.06*140,sy2=v=>400-v*160;
   let s=vAxis(sy1,m.l,W-m.r,[0,0.03,0.06],v=>f2(v),'DENSITY p',22);
   s+=`<path d="${area(v=>npdf(v,MU0,SG0),140,Math.min(x,205),sx,sy1)}" fill="var(--p-soft)"/><path d="${curve(v=>npdf(v,MU0,SG0),140,205,sx,sy1)}" fill="none" stroke="var(--p)" stroke-width="2.6"/>`;
   s+=vAxis(sy2,m.l,W-m.r,[0,0.5,1],v=>v,'RUNNING AREA F',224);
   s+=`<path d="${curve(v=>ncdf(v,MU0,SG0),140,205,sx,sy2)}" fill="none" stroke="var(--ink)" stroke-width="2.6"/>`;
   s+=`<line x1="${sx(x)}" x2="${sx(x)}" y1="${sy1(0)}" y2="${sy2(0)}" stroke="var(--x)" stroke-width="1.4" stroke-dasharray="4 4"/><circle cx="${sx(x)}" cy="${sy2(F)}" r="6" fill="var(--x)"/>`;
   const right=sx(x)<W-150;s+=txt(sx(x)+(right?10:-10),sy2(F)-10,'F = '+f4(F),{c:'num',a:right?'start':'end',s:'fill:var(--x)'});
   s+=hAxis(sx,140,200,408,10,'HEIGHT, CM');$('g').innerHTML=s;}},
/* 12 */{title:'F read backwards is F⁻¹',lit:['F','invert','Finv'],flip:true,
 say:'Read F upwards and a height x gives a share F(x). Read the same curve from the side and a share u gives back a height F⁻¹(u), one sample from the bell.',
 note:'F⁻¹(u) is computed as μ + σ·Φ⁻¹(u), with Φ⁻¹ the standard bell\u2019s inverse (Acklam\u2019s rational approximation). Journey 03 lists Φ⁻¹(0.975) = 1.96 (invcdf_normal_at_0975 in its facts.json). Because F only rises, each u has exactly one height.',
 def:{xq:173,u:+F173.toFixed(3),read:false},pres:[['x = 173',{xq:173}],['u = 0.5',{read:true,u:0.5}],['u = 0.975',{read:true,u:0.975}]],
 sw:[{k:'read',on:'Reading across: u gives F⁻¹(u)',off:'Reading up: x gives F(x)'}],
 eq:()=>P.read?`${cl('c-x','x')}=F^{-1}(${cl('c-eps','u')})=${cl('c-x',f1(Finv(P.u)))}\\text{ cm}`:`${cl('c-eps','u')}=F(${cl('c-x','x')})=${cl('c-eps',f4(ncdf(P.xq,MU0,SG0)))}`,
 eqT:()=>P.read?`${cl('c-eps','u')}=${H('u')}`:`${cl('c-x','x')}=${H('xq')}\\text{ cm}`,
 check(){if(!P.read)return `Up from ${f1(P.xq)} cm: F = ${f4(ncdf(P.xq,MU0,SG0))}. Switch the reading and feed that share back in to get ${f1(P.xq)} cm again.`;
   const x=Finv(P.u),zq=probit(P.u);if(Math.abs(P.u-0.975)<0.0005)return `u = 0.975 lands ${f2(zq)} spreads above the mean: ${f1(MU0)} + ${f2(zq)} × ${f2(SG0)} = ${f1(x)} cm.`;
   if(Math.abs(x-ME)<0.15)return `u = ${f3(P.u)} gives ${f1(x)} cm, your height, since F(173) = ${f4(F173)}.`;return `u = ${f3(P.u)}: a share ${f3(P.u)} of the bell lies below ${f1(x)} cm.`;},
 fig:el=>{el.innerHTML=`<div class="card">${SVG('g',600,340,'The running area F, read up from a height or across from a share u')}</div>`;},
 draw(){const W=600,Hh=340,m={l:56,r:20,t:24,b:54},sx=lin(140,205,m.l,W-m.r),sy=v=>Hh-m.b-v*(Hh-m.t-m.b);
   const x=P.read?Finv(P.u):P.xq,u=P.read?P.u:ncdf(P.xq,MU0,SG0),col=P.read?'var(--eps)':'var(--x)';
   let s=vAxis(sy,m.l,W-m.r,[0,0.25,0.5,0.75,1],v=>v,'SHARE u = F(x)',m.t-8);
   s+=`<path d="${curve(v=>ncdf(v,MU0,SG0),140,205,sx,sy)}" fill="none" stroke="var(--ink)" stroke-width="2.6"/>`;
   const d=P.read?`M${m.l} ${sy(u)}H${sx(x)}V${Hh-m.b}`:`M${sx(x)} ${Hh-m.b}V${sy(u)}H${m.l}`;
   s+=`<path class="draw" pathLength="1" d="${d}" fill="none" stroke="${col}" stroke-width="2.2"/>`;
   s+=`<circle cx="${m.l}" cy="${sy(u)}" r="6" fill="var(--eps)"/><circle cx="${sx(x)}" cy="${Hh-m.b}" r="7" fill="var(--x)"/>`;
   s+=txt(m.l+10,sy(u)-8,'u = '+f3(u),{c:'num',s:'fill:var(--eps)'});
   const right=sx(x)<W-140;s+=txt(sx(x)+(right?10:-10),Hh-m.b-10,f1(x)+' cm',{c:'num',a:right?'start':'end',s:'fill:var(--x)'});
   s+=hAxis(sx,140,200,Hh-m.b+6,10,'HEIGHT x, CM');$('g').innerHTML=s;}},
/* 13 */{title:"Many u's rebuild the bell",lit:['Finv','feed','new'],
 say:"Feed in many evenly spread random u's and the heights pile up in the bell's shape. All the randomness lives in u, and F⁻¹ is a fixed rule that only moves it.",
 note:'The u\u2019s come from a seeded generator, so the same count always gives the same draws. Bars are 2.5 cm wide and rescaled to area 1.',
 def:{nd:200},pres:[['10 draws',{nd:10}],['2000 draws',{nd:2000}],['New draws',{nd:200,seedInc:1}]],
 eq:()=>`${cl('c-eps','u_j')}\\sim\\text{Uniform}(0,1),\\qquad ${cl('c-x','x_j')}=F^{-1}(${cl('c-eps','u_j')}),\\qquad j=1,\\dots,${P.nd}`,
 eqT:()=>`\\text{draws}=${H('nd')}`,
 check(){const d=draws();const mu=d.reduce((a,b)=>a+b,0)/d.length,s=R(d.reduce((a,b)=>a+(b-mu)**2,0)/d.length);return `${d.length} draws: mean ${f1(mu)} cm, spread ${f2(s)} cm, against the bell\u2019s ${f1(MU0)} and ${f2(SG0)}.`;},
 fig:el=>{el.innerHTML=`<div class="card">${SVG('g',600,360,'A strip of random u values above a histogram of the heights they give')}</div>`;},handoff:[PAGE[2]+'#f1','Does F⁻¹ work for an image? Scene 2 →'],
 draw(){const W=600,m={l:56,r:20},d=draws(),us=uDraws();const su=lin(0,1,m.l,W-m.r),sx=lin(140,205,m.l,W-m.r);
   let s=`<line class="axis" x1="${m.l}" x2="${W-m.r}" y1="52" y2="52"/>`+txt(m.l,20,'RANDOM u, 0 TO 1',{c:'ax'});us.slice(0,300).forEach(u=>{s+=`<line x1="${su(u)}" x2="${su(u)}" y1="32" y2="50" stroke="var(--eps)" stroke-opacity=".55"/>`;});
   s+=txt(m.l-6,57,'0',{c:'tick',a:'end'})+txt(W-m.r,70,'1',{c:'tick',a:'end'});
   const bw=2.5,ym=0.07,sy=v=>290-Math.min(v,ym)/ym*190;s+=vAxis(sy,m.l,W-m.r,[0,0.035,0.07],v=>f3(v),'',0);
   for(let a=140;a<205;a+=bw){const c=d.filter(h=>h>=a&&h<a+bw).length;if(c)s+=`<rect x="${sx(a)}" y="${sy(c/(d.length*bw))}" width="${sx(a+bw)-sx(a)-1}" height="${290-sy(c/(d.length*bw))}" fill="var(--x-soft)" stroke="var(--x)" stroke-width=".8"/>`;}
   s+=`<path d="${curve(v=>npdf(v,MU0,SG0),140,205,sx,sy)}" fill="none" stroke="var(--p)" stroke-width="2.4"/>`;
   HTS.forEach(h=>{s+=`<circle cx="${sx(h)}" cy="326" r="4.5" fill="var(--ink)"/>`;});s+=txt(sx(186.5),330,'the ten real heights',{c:'tick'});
   s+=hAxis(sx,140,200,290,10,'');s+=txt(W-m.r,352,'HEIGHT, CM',{c:'ax',a:'end'});$('g').innerHTML=s;}}
];
setupPage({no:1,frames:FR1,base:{mu:MU0,sg:SG0,bw:5,xq:173,w:2,u:0.5,nd:200,seed:1,all:true,read:false},map:MAP1});
