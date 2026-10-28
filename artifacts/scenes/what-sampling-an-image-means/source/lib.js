/* ===== shared library for the three scenes: maths, noise, KaTeX handles, charts, map, walk ===== */
const R=Math.sqrt, PI=Math.PI, LN2PI=Math.log(2*PI), T=100;
const $=id=>document.getElementById(id);
const SUP='⁰¹²³⁴⁵⁶⁷⁸⁹';
const sub=n=>String(n).split('').map(c=>'₀₁₂₃₄₅₆₇₈₉'[c]||c).join('');
const f1=v=>v.toFixed(1), f2=v=>v.toFixed(2), f3=v=>v.toFixed(3), f4=v=>v.toFixed(4);
const mm=s=>String(s).replace(/^-/,'−');
const sci=v=>{if(v===0)return '0';const e=Math.floor(Math.log10(Math.abs(v)));const m=v/Math.pow(10,e);return m.toFixed(3)+' × 10'+(e<0?'⁻':'')+String(Math.abs(e)).split('').map(c=>SUP[c]).join('');};
const tsci=v=>{if(v===0)return '0';const e=Math.floor(Math.log10(Math.abs(v)));if(e>=-3&&e<4)return f4(v);const m=v/Math.pow(10,e);return `${m.toFixed(3)}\\times10^{${e}}`;};
const comma=n=>String(n).replace(/\B(?=(\d{3})+(?!\d))/g,',');
const tcomma=n=>comma(n).replace(/,/g,'{,}');
const PAGE={1:'index.html',2:'many-numbers.html',3:'going-back.html'};
const SCENES=[[1,'One number'],[2,'Many numbers'],[3,'Going back']];

/* ---------- the ten heights: example values, shared by all three scenes ---------- */
const HTS=[158,163,165,168,170,172,173,176,179,184], ME=173, NH=10, MEI=HTS.indexOf(ME);
const SUM=HTS.reduce((a,b)=>a+b,0), MU0=SUM/NH, SS0=HTS.reduce((a,h)=>a+(h-MU0)**2,0), VAR0=SS0/NH, SG0=R(VAR0);
const Z=HTS.map(h=>(h-MU0)/SG0), toCm=z=>MU0+SG0*z;
const nearH=cm=>HTS.reduce((b,h)=>Math.min(b,Math.abs(h-cm)),Infinity);

/* ---------- bells ---------- */
function erf(x){const s=x<0?-1:1;x=Math.abs(x);const t=1/(1+0.3275911*x);return s*(1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x));}
const npdf=(x,m,s)=>Math.exp(-((x-m)**2)/(2*s*s))/(s*R(2*PI));
const lpdf=(x,m,s)=>-Math.log(s)-0.5*LN2PI-((x-m)**2)/(2*s*s);
const ncdf=(x,m,s)=>0.5*(1+erf((x-m)/(s*Math.SQRT2)));
function probit(p){const a=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00],
  b=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01],
  c=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00],
  d=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00];
  const pl=0.02425;let q,r;
  if(p<pl){q=R(-2*Math.log(p));return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);}
  if(p<=1-pl){q=p-0.5;r=q*q;return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);}
  q=R(-2*Math.log(1-p));return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);}
const Finv=u=>MU0+SG0*probit(u);
const logL=(mu,sg,all)=>all?HTS.reduce((a,h)=>a+lpdf(h,mu,sg),0):lpdf(ME,mu,sg);
const BEST=logL(MU0,SG0,true), F173=ncdf(ME,MU0,SG0);
/* invert any increasing function on [lo,hi] by bisection */
function invert(F,u,lo,hi){for(let i=0;i<80;i++){const m=(lo+hi)/2;if(F(m)<u)lo=m;else hi=m;}return (lo+hi)/2;}

/* ---------- Step 4's cosine schedule ---------- */
function schedule(delta,clip){
  const C=new Float64Array(T+1);
  for(let k=0;k<=T;k++){const c=Math.cos(((k/T)+delta)/(1+delta)*PI*0.5);C[k]=c*c;}
  const c0=C[0];for(let k=0;k<=T;k++)C[k]/=c0;
  const beta=new Float64Array(T),abar=new Float64Array(T);let p=1;
  for(let t=0;t<T;t++){let b=1-C[t+1]/C[t];if(b<0)b=0;if(clip&&b>0.999)b=0.999;beta[t]=b;p*=1-b;abar[t]=p;}
  return {C,beta,abar};
}
const S=schedule(0.008,true);
const ab=t=>t<0?1:S.abar[t];
function coef(t){const b=S.beta[t],a=S.abar[t],ap=ab(t-1);return {A:R(ap)*b/(1-a),B:R(1-b)*(1-ap)/(1-a),V:b*(1-ap)/(1-a)};}

/* ---------- seeded noise ---------- */
function rng(seed){let a=seed>>>0;return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
function gauss(seed,n){const r=rng(seed),o=new Float32Array(n);for(let i=0;i<n;i+=2){let u=r();if(u<1e-12)u=1e-12;const v=r(),m=R(-2*Math.log(u));o[i]=m*Math.cos(2*PI*v);if(i+1<n)o[i+1]=m*Math.sin(2*PI*v);}return o;}

/* ---------- best guess of x0 and the reverse chain, for heights ---------- */
function den1(x,t,zs,pw){const a=S.abar[t],ra=R(a),v=1-a;let mx=-Infinity;
  const lw=zs.map((z,i)=>{const l=Math.log(pw?pw[i]:1)-(x-ra*z)**2/(2*v);if(l>mx)mx=l;return l;});
  let s=0;const w=lw.map(l=>{const e=Math.exp(l-mx);s+=e;return e;});let o=0;w.forEach((e,i)=>{w[i]=e/s;o+=w[i]*zs[i];});
  const lik=zs.map(z=>Math.exp(-((x-ra*z)**2)/(2*v)));return {x0:o,w,lik};}
const denBell=(x,t,m,s2)=>{const a=S.abar[t],ra=R(a);return m+s2*ra*(x-ra*m)/(a*s2+1-a);};
function chain1(seed,nc,guess){
  const out=[];for(let c=0;c<nc;c++){const r=rng(seed*9973+c*131+1);const g=()=>{let u=r();if(u<1e-12)u=1e-12;return R(-2*Math.log(u))*Math.cos(2*PI*r());};
    let x=g();const tr=new Float64Array(101);tr[0]=x;
    for(let t=99;t>=0;t--){const x0=guess(x,t),k=coef(t);x=k.A*x0+k.B*x+(k.V>0?R(k.V)*g():0);tr[100-t]=x;}
    out.push(tr);}
  return out;}

/* ---------- images ---------- */
const N0=128*128*3, D=48*48*3;
const X0=new Float32Array(N0);let ready0=false;
const DOGS=[0,1,2,3].map(()=>new Float32Array(D)),THUMB=[];let readyD=false;
const DOGNAME=['the golden puppy','the border collie','the husky','the labrador'];
function loadImgs(src0,srcD){
  if(src0){const im0=new Image();im0.onload=()=>{const c=document.createElement('canvas');c.width=c.height=128;const g=c.getContext('2d');if(!g)return;g.drawImage(im0,0,0,128,128);
    const d=g.getImageData(0,0,128,128).data;let j=0;for(let i=0;i<d.length;i+=4){X0[j++]=d[i]/127.5-1;X0[j++]=d[i+1]/127.5-1;X0[j++]=d[i+2]/127.5-1;}THUMB0=c.toDataURL();ready0=true;inval();};im0.src=src0;}
  if(srcD){const imD=new Image();imD.onload=()=>{const c=document.createElement('canvas');c.width=192;c.height=48;const g=c.getContext('2d');if(!g)return;g.drawImage(imD,0,0);
    for(let k=0;k<4;k++){const d=g.getImageData(k*48,0,48,48).data;let j=0;for(let i=0;i<d.length;i+=4){DOGS[k][j++]=d[i]/127.5-1;DOGS[k][j++]=d[i+1]/127.5-1;DOGS[k][j++]=d[i+2]/127.5-1;}
      const t=document.createElement('canvas');t.width=t.height=48;t.getContext('2d').putImageData(g.getImageData(k*48,0,48,48),0,0);THUMB[k]=t.toDataURL();}
    readyD=true;imgCache={};inval();};imD.src=srcD;}
}
let THUMB0='';
function paint(arr,cv,n){if(!cv)return;const g=cv.getContext&&cv.getContext('2d');if(!g)return;cv.width=cv.height=n;const im=g.createImageData(n,n),d=im.data;let j=0;
  for(let i=0;i<d.length;i+=4){for(let c=0;c<3;c++){let v=arr[j++];v=v<-1?-1:v>1?1:v;d[i+c]=Math.round((v+1)*127.5);}d[i+3]=255;}g.putImageData(im,0,0);}
/* a greyscale n×n image from n·n values in −1..1 */
function paintGrey(arr,cv,n){if(!cv)return;const g=cv.getContext&&cv.getContext('2d');if(!g)return;cv.width=cv.height=n;const im=g.createImageData(n,n),d=im.data;
  for(let i=0;i<n*n;i++){let v=arr[i];v=v<-1?-1:v>1?1:v;const c=Math.round((v+1)*127.5);d[4*i]=d[4*i+1]=d[4*i+2]=c;d[4*i+3]=255;}g.putImageData(im,0,0);}
const rms=(a,b,n)=>{let s=0;for(let i=0;i<n;i++){const e=a[i]-b[i];s+=e*e;}return R(s/n);};
let imgCache={};
function chainImg(seed,fresh){
  const key=seed+'|'+fresh;if(imgCache[key])return imgCache[key];
  let x=gauss(seed*7919+11,D);const xs=[Float32Array.from(x)],hs=[],ws=[];
  for(let t=99;t>=0;t--){
    const a=S.abar[t],ra=R(a),v=1-a;
    const lw=DOGS.map(d=>{let s=0;for(let i=0;i<D;i++){const e=x[i]-ra*d[i];s+=e*e;}return -s/(2*v);});
    const mx=Math.max(...lw);let sm=0;const w=lw.map(l=>{const e=Math.exp(l-mx);sm+=e;return e;}).map(e=>e/sm);
    const x0=new Float32Array(D);for(let k=0;k<4;k++){if(w[k]<1e-12)continue;const d=DOGS[k];for(let i=0;i<D;i++)x0[i]+=w[k]*d[i];}
    const nx=new Float32Array(D);
    if(fresh){const k=coef(t),e=k.V>0?gauss(seed*104729+t*31+7,D):null,sv=R(k.V);for(let i=0;i<D;i++)nx[i]=k.A*x0[i]+k.B*x[i]+(e?sv*e[i]:0);}
    else{const ap=ab(t-1),re=R(1-a);for(let i=0;i<D;i++){const eh=(x[i]-ra*x0[i])/re;nx[i]=R(ap)*x0[i]+R(1-ap)*eh;}}
    hs.push(x0);ws.push(w);x=nx;xs.push(x);
  }
  const fin=xs[100];let end=0,dist=Infinity;DOGS.forEach((d,k)=>{const r=rms(fin,d,D);if(r<dist){dist=r;end=k;}});
  return imgCache[key]={xs,hs,ws,end,dist};
}

/* ---------- KaTeX with draggable numbers and links on symbols ---------- */
const KO={displayMode:true,output:'htmlAndMathml',strict:false,throwOnError:!!window.__STRICT__,trust:c=>c.command==='\\htmlData'||c.command==='\\htmlClass'||c.command==='\\href'};
const cl=(c,t)=>`\\htmlClass{${c}}{${t}}`;
const lk=(href,t)=>`\\href{${href}}{${t}}`;
const lim={},stp={},LABEL={},DIG={};
function handle(k,lo,hi,step,label,dig){lim[k]=[lo,hi];stp[k]=step;LABEL[k]=label;if(dig!==undefined)DIG[k]=dig;}
const fmtK=(k,v)=>DIG[k]!==undefined?v.toFixed(DIG[k]):String(Math.round(v));
const H=k=>`\\htmlData{k=${k}}{${mm(fmtK(k,P[k]))}}`;
const clampK=(k,v)=>{v=Math.min(lim[k][1],Math.max(lim[k][0],v));return DIG[k]!==undefined?Math.round(v/stp[k])*stp[k]:Math.round(v);};
let activeKey=null,focusKey=null;
function K(el,tex){if(!tex){el.innerHTML='';return;}katex.render(tex,el,KO);
  el.querySelectorAll('[data-k]').forEach(n=>{const k=n.dataset.k;n.tabIndex=0;n.setAttribute('role','slider');
    n.setAttribute('aria-label',LABEL[k]+', drag sideways or use the arrow keys');n.setAttribute('aria-valuenow',P[k]);n.setAttribute('aria-valuemin',lim[k][0]);n.setAttribute('aria-valuemax',lim[k][1]);
    if(k===activeKey)n.classList.add('active');});
  if(focusKey&&!el.contains(document.activeElement)){const n=el.querySelector(`[data-k="${focusKey}"]`);if(n)n.focus({preventScroll:true});}}

/* ---------- chart helpers ---------- */
const lin=(lo,hi,a,b)=>v=>a+(v-lo)/(hi-lo)*(b-a);
const pth=(xs,ys)=>xs.map((x,i)=>`${i?'L':'M'}${x.toFixed(1)} ${ys[i].toFixed(1)}`).join('');
const txt=(x,y,s,o)=>{o=o||{};return `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}"${o.c?` class="${o.c}"`:''}${o.a?` text-anchor="${o.a}"`:''}${o.s?` style="${o.s}"`:''}>${s}</text>`;};
function hAxis(sx,lo,hi,y,step,title,fmt){let s=`<line class="axis" x1="${sx(lo)}" x2="${sx(hi)}" y1="${y}" y2="${y}"/>`;
  for(let v=Math.ceil(lo/step-1e-9)*step;v<=hi+1e-9;v+=step)s+=`<line class="axis" x1="${sx(v)}" x2="${sx(v)}" y1="${y}" y2="${y+5}"/>`+txt(sx(v),y+18,fmt?fmt(v):mm(+v.toFixed(2)),{c:'tick',a:'middle'});
  if(title)s+=txt(sx(hi),y+36,title,{c:'ax',a:'end'});return s;}
function vAxis(sy,x,x2,ticks,fmt,title,ytop){let s='';ticks.forEach(v=>{s+=`<line class="grid" x1="${x}" x2="${x2}" y1="${sy(v)}" y2="${sy(v)}"/>`+txt(x-6,sy(v)+4,fmt(v),{c:'tick',a:'end'});});
  if(title)s+=txt(x+4,ytop,title,{c:'ax'});return s;}
const curve=(f,lo,hi,sx,sy,n)=>{n=n||240;const xs=[],ys=[];for(let i=0;i<=n;i++){const v=lo+(hi-lo)*i/n;xs.push(sx(v));ys.push(sy(f(v)));}return pth(xs,ys);};
const area=(f,lo,hi,sx,sy,n)=>{n=n||120;const xs=[sx(lo)],ys=[sy(0)];for(let i=0;i<=n;i++){const v=lo+(hi-lo)*i/n;xs.push(sx(v));ys.push(sy(f(v)));}xs.push(sx(hi));ys.push(sy(0));return pth(xs,ys)+'Z';};
const SVG=(id,W,Hh,label)=>`<div class="chart-scroll"><svg class="chart" id="${id}" viewBox="0 0 ${W} ${Hh}" role="img" aria-label="${label}"></svg></div>`;
const SVGS=SVG;
const niceTicks=(ym)=>{const st=ym>0.2?0.1:ym>0.09?0.05:0.02,tk=[];for(let v=0;v<=ym+1e-9;v+=st)tk.push(+v.toFixed(2));return tk;};

/* ---------- the concept map: one row of nodes joined by named operations ---------- */
/* ROW: [{id,label,href?}, {edge:id,label}|{gap:true}, {id,...}, ...] */
function layoutRow(row,y){let x=8;const out={nodes:[],edges:[]};let prev=null,pendingEdge=null;
  row.forEach(it=>{if(it.edge){pendingEdge=it;return;}if(it.gap){x+=26;out.nodes.length&&(out.nodes[out.nodes.length-1].gapAfter=true);prev=null;return;}
    const w=Math.max(44,it.label.length*7.4+20);
    if(pendingEdge&&prev){const g=Math.max(54,pendingEdge.label.length*6.7+22);out.edges.push({id:pendingEdge.edge,label:pendingEdge.label,x1:prev.x+prev.w,x2:x+g,y});x+=g;}
    const n={id:it.id,label:it.label,href:it.href,x,w,y};out.nodes.push(n);x+=w;prev=n;pendingEdge=null;});
  out.W=x+8;return out;}
function mapSVG(rows,lit,opt){opt=opt||{};let s='',W=0;const all=rows.map((r,i)=>{const L=layoutRow(r,22+i*56);W=Math.max(W,L.W);return L;});
  const Hh=all.length*56+4;const isLit=id=>!lit||lit.includes(id);
  s+=`<defs><marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L8 4L0 8z" fill="var(--muted)"/></marker></defs>`;
  all.forEach(L=>{L.edges.forEach(e=>{const on=isLit(e.id);s+=`<g class="m-edge${on&&lit?' lit':''}"><line x1="${e.x1+3}" x2="${e.x2-4}" y1="${e.y+14}" y2="${e.y+14}" stroke="${on&&lit?'var(--nz)':'var(--muted)'}" stroke-width="${on&&lit?2.6:1.4}" marker-end="url(#ah)"/>`+txt((e.x1+e.x2)/2,e.y+5,e.label,{c:'mlab',a:'middle'})+`</g>`;});
    L.nodes.forEach(n=>{const on=isLit(n.id);const inner=`<rect x="${n.x}" y="${n.y+2}" width="${n.w}" height="24" rx="12" fill="${on&&lit?'var(--x-soft)':'var(--panel)'}" stroke="${on&&lit?'var(--x)':'var(--rule)'}" stroke-width="${on&&lit?2:1}"/>`+txt(n.x+n.w/2,n.y+18.5,n.label,{c:'mnode',a:'middle'});
      s+=n.href?`<a href="${n.href}" class="m-node${on&&lit?' lit':''}">${inner}</a>`:`<g class="m-node${on&&lit?' lit':''}">${inner}</g>`;});});
  return {s,W,Hh};}

/* ---------- the three scenes' concept maps ---------- */
const MAP1=[[{id:'x',label:'one height x',href:PAGE[1]+'#f1'},{edge:'pile',label:'pile up ten'},{id:'data',label:'ten heights',href:PAGE[1]+'#f2'},{edge:'fit',label:'fit: make L largest'},
  {id:'p',label:'bell p(x)',href:PAGE[1]+'#f4'},{edge:'area',label:'running area'},{id:'F',label:'F(x)',href:PAGE[1]+'#f11'},{edge:'invert',label:'read backwards'},
  {id:'Finv',label:'F⁻¹(u)',href:PAGE[1]+'#f12'},{edge:'feed',label:'feed random u'},{id:'new',label:'a new height',href:PAGE[1]+'#f13'}]];
const MAP2=[[{id:'img',label:'2 × 2 image',href:PAGE[2]+'#f1'},{edge:'flat',label:'list 4 numbers'},{id:'pt',label:'point in ℝ⁴',href:PAGE[2]+'#f1'},{edge:'dens',label:'density'},
  {id:'p4',label:'p(x) on ℝ⁴',href:PAGE[2]+'#f2'},{edge:'marg',label:'integrate'},{id:'pix',label:'one pixel',href:PAGE[2]+'#f3'},{gap:true},
  {id:'us',label:'u₁ … u₄',href:PAGE[2]+'#f4'},{edge:'order',label:'F⁻¹ in order'},{id:'s1',label:'image',href:PAGE[2]+'#f4'},{gap:true},
  {id:'z',label:'z ~ N(0, I)',href:PAGE[2]+'#f7'},{edge:'G',label:'transport G'},{id:'s2',label:'image',href:PAGE[2]+'#f7'},{gap:true},
  {id:'x0',label:'photo x₀',href:PAGE[2]+'#f9'},{edge:'noise',label:'add noise'},{id:'xt',label:'xₜ',href:PAGE[3]+'#f1'}]];
const MAP3=[[{id:'xt',label:'noisy xₜ',href:PAGE[3]+'#f1'},{edge:'bayes',label:'Bayes'},{id:'w',label:'weights on x₀',href:PAGE[3]+'#f2'},{edge:'avg',label:'average'},{id:'xh',label:'best guess x̂₀',href:PAGE[3]+'#f3'},
  {edge:'step',label:'step back'},{id:'xt1',label:'xₜ₋₁',href:PAGE[3]+'#f4'},{edge:'rep',label:'repeat 100 times'},{id:'smp',label:'a sample',href:PAGE[3]+'#f6'},{gap:true},
  {id:'pz',label:'p(x)',href:PAGE[1]+'#f4'},{edge:'lab',label:'× p(tall | x)'},{id:'pc',label:'p(x | tall)',href:PAGE[3]+'#f8'},{gap:true},
  {id:'data',label:'training data',href:PAGE[3]+'#f9'},{edge:'dec',label:'decides'},{id:'out',label:'what comes out',href:PAGE[3]+'#f9'}]];

/* ---------- the walk ---------- */
const P={};let fi=0,curPre=-1,built=false;
let FR=[],BASE={},MAPROWS=[];
function setupPage(cfg){FR=cfg.frames;BASE=cfg.base;MAPROWS=cfg.map;
  $('series').innerHTML=SCENES.map(([n,t])=>n===cfg.no?`<span class="sc on">${n} · ${t}</span>`:`<a class="sc" href="${PAGE[n]}">${n} · ${t}</a>`).join('');
  $('prev').onclick=()=>go(fi-1);$('next').onclick=()=>go(fi+1);
  addEventListener('hashchange',()=>{const h=/^#f(\d+)$/.exec(location.hash||'');if(h)go(+h[1]-1);});
  const m=/^#f(\d+)$/.exec(location.hash||'');enter(m?Math.max(0,Math.min(FR.length-1,+m[1]-1)):0);}
function go(i){i=Math.max(0,Math.min(FR.length-1,i));if(i!==fi||!built)enter(i);}
function drawMap(){const f=FR[fi];const full=f.fullMap;const svg=$('map');svg.style.display=full?'none':'';
  $('mapfull').innerHTML=full?full.map((row,i)=>{const R1=mapSVG([row],null);return `<p class="maprow">${SCENES[i][0]} · ${SCENES[i][1]}</p><svg class="map" viewBox="0 0 ${R1.W} ${R1.Hh}" style="width:${R1.W}px" role="img" aria-label="Concept map of scene ${i+1}">${R1.s}</svg>`;}).join(''):'';
  if(full){$('mapwrap').classList.remove('dimmed');return;}
  const M=mapSVG(MAPROWS,f.lit||[]);svg.setAttribute('viewBox',`0 0 ${M.W} ${M.Hh}`);svg.style.width=M.W+'px';svg.innerHTML=M.s;
  $('mapwrap').classList.toggle('dimmed',!full);$('mapwrap').classList.toggle('full',!!full);
  const first=svg.querySelector('.lit');if(first&&first.getBBox){try{const b=first.getBBox(),sc=svg.getBoundingClientRect().width/M.W||1;$('mapwrap').scrollLeft=Math.max(0,b.x*sc-24);}catch(e){}}}
function enter(i){fi=i;document.body.dataset.frame=String(i+1);const f=FR[i];for(const k in P)delete P[k];Object.assign(P,BASE,f.def||{});curPre=-1;built=true;
  $('fTitle').textContent=f.title;$('cap').textContent=f.say;$('fNote').textContent=f.note||'';
  $('fnum').textContent=`${i+1} / ${FR.length}`;$('progbar').style.width=((i+1)/FR.length*100).toFixed(1)+'%';$('prev').disabled=i===0;$('next').disabled=i===FR.length-1;
  $('fPres').innerHTML=(f.pres||[]).map(([l],j)=>`<button class="pre" data-j="${j}" aria-pressed="false">${l}</button>`).join('');
  $('fPres').querySelectorAll('button').forEach(b=>b.onclick=()=>applyPre(+b.dataset.j));
  $('fSw').innerHTML=(f.sw||[]).map((s,j)=>`<button class="sw" data-j="${j}"></button>`).join('');
  $('fSw').querySelectorAll('button').forEach(b=>b.onclick=()=>{const s=f.sw[+b.dataset.j];P[s.k]=!P[s.k];curPre=-1;markPre();inval();});
  $('flip').hidden=!f.flip;const ho=$('handoff');if(f.handoff){ho.href=f.handoff[0];ho.textContent=f.handoff[1];ho.hidden=false;}else ho.hidden=true;
  drawMap();
  const fig=$('fFig');fig.classList.remove('enter');void fig.offsetWidth;fig.classList.add('enter');f.fig(fig);
  try{history.replaceState(null,'','#f'+(i+1));}catch(e){}
  drawNow();}
function applyPre(j){const f=FR[fi],p=f.pres[j][1],seed=P.seed;const keep={};(f.sw||[]).forEach(s=>keep[s.k]=P[s.k]);
  for(const k in P)delete P[k];Object.assign(P,BASE,f.def||{},keep);const q=Object.assign({},p);delete q.seedInc;Object.assign(P,q);P.seed=p.seedInc?seed+1:seed;curPre=j;markPre();inval();}
function markPre(){$('fPres').querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',+b.dataset.j===curPre));}
function drawNow(){const f=FR[fi];K($('fEq'),f.eq?f.eq():'');K($('fEqT'),f.eqT?f.eqT():'');
  $('eqs').hidden=!$('fEq').innerHTML&&!$('fEqT').innerHTML;
  const c=$('fCheck');c.innerHTML=f.check?f.check():'';c.classList.toggle('bad',!!(f.bad&&f.bad()));
  (f.sw||[]).forEach((s,j)=>{const b=$('fSw').querySelector(`[data-j="${j}"]`);const on=!!P[s.k];b.textContent=on?s.on:s.off;b.setAttribute('aria-pressed',on);b.classList.toggle('offstate',!!s.warnOff&&!on);});
  f.draw();}
let raf=0;const inval=()=>{if(!raf)raf=requestAnimationFrame(()=>{raf=0;drawNow();});};
function setK(k,v){P[k]=clampK(k,v);}
function userMoved(){if(curPre>=0){curPre=-1;markPre();}}
document.addEventListener('pointerdown',e=>{
  const el=e.target.closest&&e.target.closest('[data-k]');if(!el)return;
  e.preventDefault();const k=el.dataset.k,x0=e.clientX,v0=P[k];activeKey=k;userMoved();
  const per=(lim[k][1]-lim[k][0])/400*(k==='t'?1.6:1);
  const mv=ev=>{setK(k,v0+(ev.clientX-x0)*per);inval();};
  const up=()=>{removeEventListener('pointermove',mv);removeEventListener('pointerup',up);removeEventListener('pointercancel',up);activeKey=null;inval();};
  addEventListener('pointermove',mv);addEventListener('pointerup',up);addEventListener('pointercancel',up);
});
document.addEventListener('focusin',e=>{const n=e.target;focusKey=(n.dataset&&n.dataset.k)?n.dataset.k:null;});
const BIG={};
document.addEventListener('keydown',e=>{
  const n=e.target,k=n.dataset&&n.dataset.k;
  if(k){if(e.key==='Home'){e.preventDefault();userMoved();P[k]=(FR[fi].def&&k in FR[fi].def)?FR[fi].def[k]:BASE[k];inval();return;}
    const d={ArrowRight:1,ArrowUp:1,ArrowLeft:-1,ArrowDown:-1}[e.key];if(!d)return;
    e.preventDefault();userMoved();setK(k,P[k]+d*stp[k]*(e.shiftKey?10:1)*(BIG[k]||1));inval();return;}
  if(e.target.closest&&e.target.closest('input,textarea,select'))return;
  if(e.key==='ArrowRight'){e.preventDefault();go(fi+1);}else if(e.key==='ArrowLeft'){e.preventDefault();go(fi-1);}
});
/* test hook: the node and browser checks drive every page through this */
window.__scene={go:i=>enter(i),frame:()=>fi,pres:()=>(FR[fi].pres||[]).length,sw:()=>(FR[fi].sw||[]).length,
  applyPre:j=>{applyPre(j);drawNow();},toggle:j=>{const s=FR[fi].sw[j];P[s.k]=!P[s.k];drawNow();},
  keys:()=>[...document.querySelectorAll('#fEq [data-k],#fEqT [data-k]')].map(n=>n.dataset.k),
  set:(k,v)=>{setK(k,v);drawNow();},lim:k=>lim[k],P:()=>Object.assign({},P),text:()=>({cap:$('cap').textContent,check:$('fCheck').textContent}),
  fake:()=>{const g=gauss(5,N0);for(let i=0;i<N0;i++)X0[i]=Math.tanh(g[i]);DOGS.forEach((d,k)=>{const h=gauss(17+k,D);for(let i=0;i<D;i++)d[i]=Math.tanh(h[i]);THUMB[k]='data:,';});ready0=readyD=true;imgCache={};}};
Object.defineProperty(window.__scene,'n',{get:()=>FR.length});
