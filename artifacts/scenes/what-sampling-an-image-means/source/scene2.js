/* ===== scene 2: many numbers ===== */
handle('p1',-1,1,0.01,'pixel 1 value',2);handle('p2',-1,1,0.01,'pixel 2 value',2);handle('p3',-1,1,0.01,'pixel 3 value',2);handle('p4',-1,1,0.01,'pixel 4 value',2);
handle('d',1,4,1,'number of pixels d');handle('q1',-1.6,1.6,0.01,'pixel 1 value',2);
handle('u1',0.001,0.999,0.001,'random number u₁',3);handle('u2',0.001,0.999,0.001,'random number u₂',3);handle('u3',0.001,0.999,0.001,'random number u₃',3);handle('u4',0.001,0.999,0.001,'random number u₄',3);
handle('m1',-1.5,1.5,0.01,'network output μ₁',2);handle('sg2',0.05,1,0.01,'spread σ',2);handle('z0',-2.5,2.5,0.01,'noise number z₀',2);handle('c',0,1.6,0.01,'scale c in G',2);handle('t',0,99,1,'step t');
loadImgs('%%DOG_B64%%','');
const P2=PAGE[2];
/* the 2×2 toy: two real images, a little spread, chosen by hand */
const S2=0.1, IA=[1,1,-1,-1], IB=[-1,-1,1,1], COMP=[IA,IB];
const n1=(x,m)=>npdf(x,m,S2);
const pToy=x=>0.5*COMP.reduce((a,m)=>a+Math.exp(-m.reduce((q,mi,i)=>q+(x[i]-mi)**2,0)/(2*S2*S2)),0)*Math.pow(2*PI*S2*S2,-2);
const marg1=v=>0.5*n1(v,1)+0.5*n1(v,-1);
const NORM=d=>Math.pow(2*PI*S2*S2,-d/2);
/* draw the four pixels in order, each from its conditional CDF given the ones already drawn */
function drawInOrder(us){const x=[],steps=[];for(let i=0;i<4;i++){const lw=COMP.map(m=>x.reduce((a,v,j)=>a+Math.log(n1(v,m[j])),Math.log(0.5)));const mx=Math.max(...lw);let w=lw.map(l=>Math.exp(l-mx));const sw=w.reduce((a,b)=>a+b,0);w=w.map(v=>v/sw);
  const F=v=>w[0]*ncdf(v,COMP[0][i],S2)+w[1]*ncdf(v,COMP[1][i],S2);x.push(invert(F,us[i],-2,2));steps.push({w,F});}return {x,steps};}
function drawAlone(us){return us.map(u=>invert(v=>0.5*ncdf(v,1,S2)+0.5*ncdf(v,-1,S2),u,-2,2));}
const isReal=x=>{const s=x.map(v=>v>0?1:0).join('');return s==='1100'||s==='0011';};
const grey=v=>{const c=Math.round((Math.max(-1,Math.min(1,v))+1)*127.5);return `rgb(${c},${c},${c})`;};
const tile=(x,X,Y,sz,o)=>{o=o||{};let s='';[[0,0],[1,0],[0,1],[1,1]].forEach(([i,j],k)=>{s+=`<rect x="${X+i*sz}" y="${Y+j*sz}" width="${sz}" height="${sz}" fill="${grey(x[k])}" stroke="${o.stroke||'var(--muted)'}" stroke-width="${o.sw||1}"/>`;});return s;};
/* the (x1,x2) slice of the toy, as a heat map, drawn once */
let heat2=null;function heatMap(sx,sy){if(heat2)return heat2;let s='';const n=64,lo=-1.6,hi=1.6,st=(hi-lo)/n,pk=0.5/(2*PI*S2*S2);
  for(let i=0;i<n;i++)for(let j=0;j<n;j++){const x1=lo+(i+.5)*st,x2=lo+(j+.5)*st,v=0.5*(n1(x1,1)*n1(x2,1)+n1(x1,-1)*n1(x2,-1));const o=Math.min(1,v/pk);if(o<0.004)continue;
    s+=`<rect x="${sx(lo+i*st).toFixed(1)}" y="${sy(lo+(j+1)*st).toFixed(1)}" width="${(sx(st)-sx(0)+0.6).toFixed(1)}" height="${(sy(0)-sy(st)+0.6).toFixed(1)}" fill="var(--p)" fill-opacity="${(0.08+0.9*o).toFixed(3)}"/>`;}
  return heat2=s;}
const sliceArea=()=>{let a=0;const st=0.004;for(let x1=-1.8;x1<1.8;x1+=st)for(let x2=-1.8;x2<1.8;x2+=st){a+=0.5*(n1(x1+st/2,1)*n1(x2+st/2,1)+n1(x1+st/2,-1)*n1(x2+st/2,-1))*st*st;}return a;};
let areaC=null;
/* training images for frame 8: eight draws from the toy, seeded */
const TRAIN=(()=>{const r=rng(4242),g=gauss(777,32),o=[];for(let n=0;n<8;n++){const m=r()<0.5?IA:IB;o.push(m.map((v,i)=>v+S2*g[n*4+i]));}return o;})();
const logLc=c=>TRAIN.reduce((a,x)=>a+Math.log(0.5*COMP.reduce((q,m)=>q+Math.exp(-m.reduce((r,mi,i)=>r+(x[i]-c*mi)**2,0)/(2*S2*S2)),0)*NORM(4)),0);
let bestC=null;function cBest(){if(bestC)return bestC;let b={v:-Infinity};for(let c=0;c<=1.6+1e-9;c+=0.001){const v=logLc(c);if(v>b.v)b={v,c};}return bestC=b;}
/* frame 7: the same 300 noise draws pushed through F⁻¹ and through G */
const ZS=(()=>{const g=gauss(2468,1500),o=[];for(let n=0;n<300;n++)o.push([g[5*n],g[5*n+1],g[5*n+2],g[5*n+3],g[5*n+4]]);return o;})();
const UZ=(()=>{const r=rng(1357);return [...Array(300)].map(()=>r());})();
const Gz=(z,c)=>IA.map((a,i)=>Math.sign(z[0]||1e-9)*(c===undefined?1:c)*a+S2*z[i+1]);
let mean0C={},n3C={};
function meanImg(){const k=P.t;if(mean0C.k!==k){const a=R(S.abar[P.t]),o=new Float32Array(N0);for(let i=0;i<N0;i++)o[i]=a*X0[i];mean0C={k,v:o};}return mean0C.v;}
function noisy3(){const k=P.t+'|'+P.seed;if(n3C.k!==k){const a=R(S.abar[P.t]),b=R(1-S.abar[P.t]);n3C={k,v:[0,1,2].map(j=>{const e=gauss(P.seed*7919+j*101+3,N0),o=new Float32Array(N0);for(let i=0;i<N0;i++)o[i]=a*X0[i]+b*e[i];return o;})};}return n3C.v;}

const FR2=[
/* 1 */{title:'An image is a point',lit:['img','flat','pt'],
 say:'List the four pixel values of a 2 × 2 grey image and the image becomes one point with four coordinates. Your height was a point with one coordinate; the Labrador photo is a point with 49,152.',
 note:'Pixel values run from −1 (black) to 1 (white), the scale Step 4 uses. Pixels are numbered row by row: x₁ and x₂ on top, x₃ and x₄ below. The Labrador has 128 × 128 pixels with 3 colour values each.',
 def:{p1:1,p2:1,p3:-1,p4:-1},pres:[['Top bright',{p1:1,p2:1,p3:-1,p4:-1}],['Bottom bright',{p1:-1,p2:-1,p3:1,p4:1}],['A checkerboard',{p1:1,p2:-1,p3:-1,p4:1}]],
 eq:()=>`${cl('c-x','x')}=(x_1,x_2,x_3,x_4)=(${H('p1')},${H('p2')},${H('p3')},${H('p4')})\\in\\mathbb{R}^4`,
 eqT:()=>`\\text{the Labrador: }128\\times128\\times3=${tcomma(N0)}\\text{ coordinates}`,
 check:()=>`This image is the point (${[P.p1,P.p2,P.p3,P.p4].map(v=>mm(f2(v))).join(', ')}). Change any pixel and the point moves along that one axis.`,
 fig:el=>{el.innerHTML=`<div class="card">${SVG('g',640,300,'A 2 by 2 grey image, its four values as bars, and the Labrador photo')}</div>`;},
 draw(){const x=[P.p1,P.p2,P.p3,P.p4];let s=txt(20,22,'THE IMAGE',{c:'ax'})+tile(x,20,40,90)+txt(110,240,'2 × 2 = 4 numbers',{a:'middle',c:'num'});
   s+=txt(240,22,'ITS FOUR NUMBERS',{c:'ax'});const by=v=>140-v*90;s+=`<line class="axis" x1="236" x2="430" y1="140" y2="140"/>`;
   x.forEach((v,i)=>{const X=250+i*45;s+=`<rect x="${X}" y="${Math.min(by(v),140)}" width="30" height="${Math.abs(by(v)-140)}" fill="var(--x)"/>`+txt(X+15,v>=0?by(v)-6:by(v)+16,mm(f2(v)),{c:'num',a:'middle'})+txt(X+15,262,'x'+sub(i+1),{a:'middle',c:'tick'});});
   s+=txt(470,22,'ONE PHOTO',{c:'ax'});s+=THUMB0?`<image href="${THUMB0}" x="470" y="40" width="150" height="150"/>`:`<rect x="470" y="40" width="150" height="150" fill="var(--shade)"/>`;
   s+=txt(545,214,'49,152 numbers',{a:'middle',c:'num'});$('g').innerHTML=s;}},
/* 2 */{title:'A density over whole images',lit:['pt','dens','p4'],
 say:'In this toy only two images really occur, top bright and bottom bright, each with a little spread. A density p(x) over all four numbers says how densely real images crowd near each point, and its volume is 1.',
 note:'The toy is chosen by hand: p(x) = ½·N(x; a, s²I) + ½·N(x; b, s²I) with a = (1, 1, −1, −1), b = (−1, −1, 1, 1) and s = 0.1. The picture shows two of the four numbers, x₁ and x₂, with the other two integrated away; that slice’s volume is summed on a 0.004 grid.',
 def:{d:4},pres:[['d = 1',{d:1}],['d = 2',{d:2}]],
 eq:()=>`\\begin{aligned}${cl('c-p','p(x)')}&=\\tfrac12\\,\\mathcal N(x;\\,a,\\,s^2I)+\\tfrac12\\,\\mathcal N(x;\\,b,\\,s^2I)\\\\\\mathcal N(x;m,s^2I)&=\\underbrace{(2\\pi s^2)^{-d/2}}_{\\text{normaliser}}\\,e^{-\\lVert x-m\\rVert^2/2s^2}\\end{aligned}`,
 eqT:()=>`d=${H('d')}:\\quad(2\\pi s^2)^{-d/2}=${f3(NORM(P.d))},\\qquad p(a)=${f2(pToy(IA))}`,
 check(){if(areaC===null)areaC=sliceArea();return `Volume of the (x₁, x₂) slice: ${f4(areaC)}. The normaliser grows from ${f3(NORM(1))} for one number to ${f2(NORM(4))} for four, because the same total of 1 is packed into a smaller region in more dimensions.`;},
 fig:el=>{el.innerHTML=`<div class="card" style="max-width:560px">${SVG('g',480,440,'Heat map of the density over pixel 1 and pixel 2, with two bumps')}</div>`;},
 draw(){const W=480,m={l:56,r:24,t:30,b:56},sx=lin(-1.6,1.6,m.l,W-m.r),sy=lin(-1.6,1.6,440-m.b,m.t);
   let s=`<rect x="${m.l}" y="${m.t}" width="${W-m.l-m.r}" height="${440-m.t-m.b}" fill="none" stroke="var(--rule)"/>`+heatMap(sx,sy);
   [-1,0,1].forEach(v=>{s+=txt(sx(v),440-m.b+18,mm(v),{c:'tick',a:'middle'})+txt(m.l-8,sy(v)+4,mm(v),{c:'tick',a:'end'});});
   s+=txt(sx(1),sy(1)-22,'top bright',{a:'middle',s:'fill:var(--p);font-weight:600'})+txt(sx(-1),sy(-1)+34,'bottom bright',{a:'middle',s:'fill:var(--p);font-weight:600'});
   s+=txt(W-m.r,440-14,'PIXEL 1 VALUE x₁',{c:'ax',a:'end'})+txt(m.l,m.t-10,'PIXEL 2 VALUE x₂',{c:'ax'});$('g').innerHTML=s;}},
/* 3 */{title:'One pixel alone is a marginal',lit:['p4','marg','pix'],flip:true,
 say:'Read the two pixels together and you see two bumps on the diagonal. Integrate pixel 2 away and pixel 1 alone has two bumps of its own, at −1 and +1, each 1.995 high.',
 note:'The marginal is p(x₁) = ∫ p(x₁, x₂) dx₂. The page checks it by summing the joint over x₂ in 0.002 steps at the chosen x₁ and comparing with the closed form ½N(x₁; 1, s²) + ½N(x₁; −1, s²).',
 def:{q1:1,mg:false},pres:[['x₁ = 1',{q1:1}],['x₁ = 0',{q1:0}],['x₁ = −0.9',{q1:-0.9}]],
 sw:[{k:'mg',on:'Reading: pixel 1 alone',off:'Reading: both pixels together'}],
 eq:()=>`${cl('c-p','p(x_1)')}=\\int ${cl('c-p','p(x_1,x_2)')}\\,dx_2=\\tfrac12\\,\\mathcal N(x_1;1,s^2)+\\tfrac12\\,\\mathcal N(x_1;-1,s^2)`,
 eqT:()=>`x_1=${H('q1')}:\\qquad ${cl('c-p','p(x_1)')}=${f4(marg1(P.q1))}`,
 check(){let a=0;const st=0.002;for(let x2=-2;x2<2;x2+=st)a+=0.5*(n1(P.q1,1)*n1(x2+st/2,1)+n1(P.q1,-1)*n1(x2+st/2,-1))*st;return `Summing the joint over x₂ at x₁ = ${f2(P.q1)} gives ${f4(a)}; the formula gives ${f4(marg1(P.q1))}.`;},
 fig:el=>{el.innerHTML=`<div class="card" style="max-width:560px">${SVG('g',480,470,'The two-pixel heat map with the pixel 1 marginal drawn underneath')}</div>`;},
 draw(){const W=480,m={l:56,r:24},sx=lin(-1.6,1.6,m.l,W-m.r),sy=lin(-1.6,1.6,300,28);
   let s=`<g opacity="${P.mg?0.3:1}">`+`<rect x="${m.l}" y="28" width="${W-m.l-m.r}" height="272" fill="none" stroke="var(--rule)"/>`+heatMap(sx,sy)+`</g>`;
   s+=`<line x1="${sx(P.q1)}" x2="${sx(P.q1)}" y1="28" y2="300" stroke="var(--x)" stroke-width="2" stroke-dasharray="${P.mg?'4 4':'0'}"/>`;
   s+=txt(m.l,20,'BOTH PIXELS: x₂ UP, x₁ ACROSS',{c:'ax'});
   const my=v=>440-v/2.2*110;s+=`<line class="axis" x1="${m.l}" x2="${W-m.r}" y1="440" y2="440"/>`;
   if(P.mg){s+=`<path d="${curve(marg1,-1.6,1.6,sx,my,300)}" fill="none" stroke="var(--p)" stroke-width="2.6"/><circle cx="${sx(P.q1)}" cy="${my(marg1(P.q1))}" r="6" fill="var(--x)"/>`+txt(sx(P.q1)+(P.q1<0.8?10:-10),Math.max(334,my(marg1(P.q1))-8),f3(marg1(P.q1)),{c:'num',a:P.q1<0.8?'start':'end',s:'fill:var(--x)'});}
   s+=txt(m.l,322,'PIXEL 1 ALONE',{c:'ax'});[-1,0,1].forEach(v=>{s+=txt(sx(v),458,mm(v),{c:'tick',a:'middle'});});$('g').innerHTML=s;}},
/* 4 */{title:'Four numbers need F⁻¹ four times, in order',lit:['us','order','s1'],
 say:'One u can only make one number, so a four-pixel image takes four u’s. Draw pixel 1 from its own F, then each next pixel from its F given the pixels already drawn.',
 note:'Each conditional F is the toy’s exact conditional, a mix of the two bumps weighted by how well each real image explains the pixels already drawn, inverted by bisection. Any pixel order works when the conditionals are known; PixelCNN-style networks learn them. Writing them as a table is hopeless: 2⁶⁴ − 1 = 1.845 × 10¹⁹ free numbers for 8 × 8 black and white (journey 03’s facts.json), a count 118,370 digits long for 128 × 128 colour at 256 levels.',
 def:{u1:0.7,u2:0.4,u3:0.6,u4:0.5},pres:[['u₁ = 0.3',{u1:0.3}],['u₁ = 0.5',{u1:0.5}]],
 eq:()=>`x_i=F_i^{-1}\\big(${cl('c-eps','u_i')}\\mid x_1,\\dots,x_{i-1}\\big),\\qquad i=1,2,3,4`,
 eqT:()=>`${cl('c-eps','u')}=(${H('u1')},${H('u2')},${H('u3')},${H('u4')})`,
 check(){const r=drawInOrder([P.u1,P.u2,P.u3,P.u4]);return `Image (${r.x.map(v=>mm(f3(v))).join(', ')}). After pixel 1 the weight on top bright is ${f3(r.steps[1].w[0])}, so the other three pixels follow it: ${isReal(r.x)?'a real image':'not a real image'}.`;},
 fig:el=>{el.innerHTML=`<div class="card">${SVG('g',680,290,'Four conditional CDFs in a row, each with its u read across to a pixel value, and the assembled 2 by 2 image')}</div>`;},
 draw(){const us=[P.u1,P.u2,P.u3,P.u4],r=drawInOrder(us);let s='';const L=['x₁','x₂ | x₁','x₃ | x₁, x₂','x₄ | x₁…x₃'];
   r.steps.forEach((st,i)=>{const X0=10+i*132,sx=lin(-1.5,1.5,X0+16,X0+120),sy=v=>210-v*150;s+=txt(X0+16,24,L[i],{c:'num'});
     s+=`<line class="axis" x1="${X0+16}" x2="${X0+120}" y1="210" y2="210"/><line class="axis" x1="${X0+16}" x2="${X0+16}" y1="60" y2="210"/>`;
     s+=`<path d="${curve(st.F,-1.5,1.5,sx,sy,160)}" fill="none" stroke="var(--ink)" stroke-width="2"/>`;
     s+=`<path d="M${X0+16} ${sy(us[i])}H${sx(r.x[i])}V210" fill="none" stroke="var(--eps)" stroke-width="2"/><circle cx="${sx(r.x[i])}" cy="210" r="5" fill="var(--x)"/>`;
     s+=txt(sx(r.x[i]),232,mm(f2(r.x[i])),{c:'num',a:'middle',s:'fill:var(--x)'});s+=txt(X0+20,sy(us[i])-6,f2(us[i]),{c:'num',s:'fill:var(--eps)'});});
   s+=txt(560,24,'THE IMAGE',{c:'ax'})+tile(r.x,560,50,52)+txt(612,190,isReal(r.x)?'real':'not real',{a:'middle',s:`fill:${isReal(r.x)?'var(--ab)':'var(--hot)'};font-weight:600`});$('g').innerHTML=s;}},
/* 5 */{title:'Skip the order and pixels disagree',lit:['pix','order'],
 say:'Draw each pixel from its own marginal and every pixel is right on its own, yet the four ignore each other. Only 2 of the 16 bright and dark patterns are real images.',
 note:'Each pixel’s marginal puts half its chance near +1 and half near −1, so drawing pixels alone gives each of the 16 bright and dark patterns a chance of 1/16, and 2 of them are real. Journey 03’s four-pixel stripes toy gives the same 0.125 (stripes_independent_valid in its facts.json).',
 def:{cond:false},pres:[['New 16 draws',{seedInc:1}]],
 sw:[{k:'cond',on:'Each pixel given the ones before',off:'Each pixel on its own',warnOff:true}],
 eq:()=>P.cond?`p(x)=p(x_1)\\,p(x_2\\mid x_1)\\,p(x_3\\mid x_1,x_2)\\,p(x_4\\mid x_1,x_2,x_3)`:`p(x)\\overset{?}{=}p(x_1)\\,p(x_2)\\,p(x_3)\\,p(x_4)`,
 eqT:()=>`\\text{share of real images}=${P.cond?'1':'\\tfrac{2}{16}=0.125'}`,
 check(){const d=draws16();const v=d.filter(isReal).length;return `${v} of these 16 draws ${v===1?'is':'are'} one of the two real images.`;},
 bad:()=>!P.cond,
 fig:el=>{el.innerHTML=`<div class="card">${SVG('g',600,330,'Sixteen drawn 2 by 2 images, each marked real or not')}</div>`;},
 draw(){const d=draws16();let s=txt(20,22,'SIXTEEN DRAWS',{c:'ax'});
   d.forEach((x,j)=>{const X=24+(j%8)*72,Y=46+Math.floor(j/8)*140,ok=isReal(x);s+=tile(x,X,Y,26);s+=`<rect x="${X-4}" y="${Y-4}" width="60" height="60" fill="none" stroke="${ok?'var(--ab)':'var(--hot)'}" stroke-width="${ok?2.5:1.5}"${ok?'':' stroke-dasharray="4 3"'}/>`+txt(X+26,Y+80,ok?'✓':'✗',{a:'middle',s:`fill:${ok?'var(--ab)':'var(--hot)'};font-size:16px`});});
   $('g').innerHTML=s;}},
/* 6 */{title:'A network outputs numbers, a formula makes them a density',lit:['p4'],
 say:'A network only outputs numbers, such as a mean μ for each pixel. Put them into the Gaussian formula and its normaliser makes the area 1 for any μ, so the network never has to.',
 note:'The same holds for a softmax over 256 levels per pixel: dividing by the sum makes the chances add to 1 whatever numbers the network gives. A third route skips the normaliser altogether: a network that learns the score ∇log p(x) never needs it, because a constant factor vanishes when the log is differentiated (Step 5 of the composition ladder).',
 def:{m1:0.6,sg2:0.3},pres:[['μ₁ = −1',{m1:-1}],['σ = 0.1',{sg2:0.1}],['σ = 1',{sg2:1}]],
 eq:()=>`p(x\\mid\\mu)=\\prod_{i=1}^{4}\\underbrace{\\frac{1}{\\sigma\\sqrt{2\\pi}}}_{${f3(1/(P.sg2*R(2*PI)))}}\\exp\\!\\Big(-\\frac{(x_i-\\mu_i)^2}{2\\sigma^2}\\Big)`,
 eqT:()=>`\\mu=(${H('m1')},\\,0.80,\\,-0.70,\\,-0.90),\\qquad\\sigma=${H('sg2')}`,
 check(){let a=0;const st=0.001;for(let v=P.m1-10*P.sg2;v<P.m1+10*P.sg2;v+=st)a+=npdf(v+st/2,P.m1,P.sg2)*st;return `Area under pixel 1’s bell: ${f4(a)}, for μ₁ = ${f2(P.m1)} and σ = ${f2(P.sg2)}. The four bells multiply, so the volume over all four pixels is 1 × 1 × 1 × 1.`;},
 fig:el=>{el.innerHTML=`<div class="card">${SVG('g',640,300,'The network output as four numbers, and for each pixel a bell centred on its number')}</div>`;},
 draw(){const mu=[P.m1,0.8,-0.7,-0.9];let s=txt(20,22,'NETWORK OUTPUT μ',{c:'ax'});const by=v=>140-v*70;s+=`<line class="axis" x1="16" x2="200" y1="140" y2="140"/>`;
   mu.forEach((v,i)=>{const X=26+i*44;s+=`<rect x="${X}" y="${Math.min(by(v),140)}" width="28" height="${Math.abs(by(v)-140)}" fill="var(--x0)"/>`+txt(X+14,v>=0?by(v)-6:by(v)+16,mm(f2(v)),{c:'num',a:'middle'})+txt(X+14,262,'μ'+sub(i+1),{a:'middle',c:'tick'});});
   s+=txt(250,22,'ONE BELL PER PIXEL, EACH OF AREA 1',{c:'ax'});
   mu.forEach((v,i)=>{const X0=250+(i%2)*190,Y0=40+Math.floor(i/2)*126,sx=lin(-2,2,X0,X0+170),pk=1/(P.sg2*R(2*PI)),sy=q=>Y0+96-Math.min(q,4)/Math.max(pk,1)*86;
     s+=`<line class="axis" x1="${X0}" x2="${X0+170}" y1="${Y0+96}" y2="${Y0+96}"/><path d="${area(q=>npdf(q,v,P.sg2),-2,2,sx,sy,160)}" fill="var(--p-soft)" stroke="var(--p)" stroke-width="1.6"/>`+txt(X0,Y0+114,'x'+sub(i+1),{c:'tick'});});
   $('g').innerHTML=s;}},
/* 7 */{title:'F⁻¹ and G both move noise',lit:['z','G','s2'],flip:true,
 say:'Scene 1 pushed one uniform number through F⁻¹ to get a height. G pushes five Gaussian numbers to get a 2 × 2 image: the sign of z₀ picks the image and the rest add its spread.',
 note:'G is written by hand for this toy and is not trained: G(z) = sign(z₀)·a + 0.1·(z₁, z₂, z₃, z₄). It turns N(0, I) in five numbers into exactly the toy’s two-bump density. A GAN generator or a VAE decoder is a learned G, and a normalizing flow is an invertible one whose density follows from the change-of-variables formula.',
 def:{z0:0.4,two:true},pres:[['z₀ = −0.4',{z0:-0.4}],['z₀ = 0.05',{z0:0.05}]],
 sw:[{k:'two',on:'Five numbers: z through G',off:'One number: u through F⁻¹'}],
 eq:()=>P.two?`G(${cl('c-eps','z')})=\\operatorname{sign}(${cl('c-eps','z_0')})\\,a+0.1\\,(${cl('c-eps','z_1')},${cl('c-eps','z_2')},${cl('c-eps','z_3')},${cl('c-eps','z_4')})`:`${cl('c-x','x')}=${lk(PAGE[1]+'#f12','F^{-1}')}(${cl('c-eps','u')}),\\qquad ${cl('c-eps','u')}\\sim\\text{Uniform}(0,1)`,
 eqT:()=>P.two?`${cl('c-eps','z_0')}=${H('z0')}\\ \\Rightarrow\\ G(z)=(${Gz([P.z0,0.3,-0.5,0.2,0.1]).map(v=>f2(v)).join(',\\,')})`:`\\text{300 draws of }${cl('c-eps','u')}\\text{ in, 300 heights out}`,
 check(){if(!P.two){return `300 u’s in, 300 heights out, mean ${f1(UZ.map(Finv).reduce((a,b)=>a+b,0)/300)} cm. Nothing is created: each u is carried to one height.`;}
   const out=ZS.map(z=>Gz(z)),top=out.filter(x=>x[0]>0).length;return `300 draws of z in, 300 images out: ${top} land on top bright and ${300-top} on bottom bright. Mass is moved and never created, so the outputs’ volume stays 1.`;},
 fig:el=>{el.innerHTML=`<div class="pair"><div class="card">${SVGS('gl',440,320,'The noise going in')}</div><div class="card">${SVGS('gr',440,320,'What comes out')}</div></div>`;},
 draw(){const W=440,Hh=320,m={l:52,r:20,t:30,b:52};
   if(!P.two){const su=lin(0,1,m.l,W-m.r),sx=lin(140,205,m.l,W-m.r);let a=txt(m.l,m.t-10,'IN: 300 UNIFORM u',{c:'ax'});UZ.forEach((u,i)=>{a+=`<circle cx="${su(u)}" cy="${80+(i*37)%180}" r="2.6" fill="var(--eps)" opacity=".6"/>`;});a+=hAxis(su,0,1,Hh-m.b+8,0.25,'u');$('gl').innerHTML=a;
     let b=txt(m.l,m.t-10,'OUT: 300 HEIGHTS',{c:'ax'});UZ.forEach((u,i)=>{const h=Finv(u);if(h>=140&&h<=205)b+=`<circle cx="${sx(h)}" cy="${80+(i*37)%180}" r="2.6" fill="var(--x)" opacity=".6"/>`;});b+=hAxis(sx,140,200,Hh-m.b+8,20,'HEIGHT, CM');$('gr').innerHTML=b;return;}
   const sz=lin(-3,3,m.l,W-m.r),szy=lin(-3,3,Hh-m.b,m.t),sx=lin(-1.6,1.6,m.l,W-m.r),sy=lin(-1.6,1.6,Hh-m.b,m.t);
   let a=txt(m.l,m.t-10,'IN: z₀ ACROSS, z₁ UP',{c:'ax'})+`<line class="grid" x1="${sz(0)}" x2="${sz(0)}" y1="${m.t}" y2="${Hh-m.b}"/>`;ZS.forEach(z=>{a+=`<circle cx="${sz(z[0])}" cy="${szy(z[1])}" r="2.4" fill="var(--eps)" opacity=".5"/>`;});
   a+=`<circle cx="${sz(P.z0)}" cy="${szy(0.3)}" r="7" fill="none" stroke="var(--nz)" stroke-width="2.4"/>`+hAxis(sz,-3,3,Hh-m.b+8,1,'');$('gl').innerHTML=a;
   let b=txt(m.l,m.t-10,'OUT: x₁ ACROSS, x₂ UP',{c:'ax'});ZS.forEach(z=>{const x=Gz(z);b+=`<circle cx="${sx(x[0])}" cy="${sy(x[1])}" r="2.4" fill="var(--x)" opacity=".5"/>`;});
   const g=Gz([P.z0,0.3,-0.5,0.2,0.1]);b+=`<circle cx="${sx(g[0])}" cy="${sy(g[1])}" r="7" fill="none" stroke="var(--nz)" stroke-width="2.4"/>`+hAxis(sx,-1.5,1.5,Hh-m.b+8,0.5,'');$('gr').innerHTML=b;}},
/* 8 */{title:'The noise is fixed, the map is learned',lit:['z','G'],
 say:'N(0, I) is fixed in advance and never learned. What is learned is G, here one scale c, chosen so the pushed images match the training images, by the same likelihood score as scene 1.',
 note:'Eight training images are drawn from the toy with a seeded generator. Under G with scale c the images follow ½N(x; c·a, s²I) + ½N(x; −c·a, s²I), and log L(c) sums the log of that density over the eight. Flows and VAEs train G on likelihood or a bound on it; a GAN uses a discriminator’s score instead; diffusion trains the best guess of scene 3.',
 def:{c:0.5},pres:[['c = 0',{c:0}],['c = 1',{c:1}],['c = 1.4',{c:1.4}]],
 eq:()=>`\\begin{aligned}\\log L(c)=\\sum_{n=1}^{8}\\log\\Big[&\\tfrac12\\mathcal N(x^{(n)};\\,c\\,a,\\,s^2I)\\\\&+\\tfrac12\\mathcal N(x^{(n)};\\,-c\\,a,\\,s^2I)\\Big]=${mm(f1(logLc(P.c)))}\\end{aligned}`,
 eqT:()=>`G_c(${cl('c-eps','z')})=\\operatorname{sign}(${cl('c-eps','z_0')})\\,${H('c')}\\,a+0.1\\,(${cl('c-eps','z_1')},\\dots,${cl('c-eps','z_4')})`,
 check(){const b=cBest();return `The score is highest at c = ${f3(b.c)}, log L = ${mm(f1(b.v))}, close to the c = 1 the toy was built with; eight images cannot pin it exactly.`;},
 fig:el=>{el.innerHTML=`<div class="pair"><div class="card">${SVGS('gl',440,320,'Training images and where G puts its two bumps')}</div><div class="card">${SVGS('gr',440,320,'Score against the scale c')}</div></div>`;},
 draw(){const W=440,Hh=320,m={l:52,r:20,t:30,b:52},sx=lin(-1.6,1.6,m.l,W-m.r),sy=lin(-1.6,1.6,Hh-m.b,m.t);
   let a=txt(m.l,m.t-10,'x₁ ACROSS, x₂ UP',{c:'ax'});[1,-1].forEach(sg=>{a+=`<circle cx="${sx(sg*P.c)}" cy="${sy(sg*P.c)}" r="${(sx(0.2)-sx(0))}" fill="var(--p-soft)" stroke="var(--p)"/>`;});
   TRAIN.forEach(x=>{a+=`<circle cx="${sx(x[0])}" cy="${sy(x[1])}" r="4.5" fill="var(--x)"/>`;});a+=hAxis(sx,-1.5,1.5,Hh-m.b+8,0.5,'');$('gl').innerHTML=a;
   const sc=lin(0,1.6,m.l,W-m.r),vs=[];for(let c=0;c<=1.6;c+=0.01)vs.push(logLc(c));const top=Math.max(...vs),yl=top-300,syl=v=>Hh-m.b-(Math.max(yl,v)-yl)/(top+20-yl)*(Hh-m.t-m.b);
   let b=vAxis(syl,m.l,W-m.r,[Math.round(top/100)*100-200,Math.round(top/100)*100-100,Math.round(top/100)*100].filter(v=>v>=yl),v=>mm(v),'SCORE, LOG L',m.t-12);
   const xs=[],ys=[];for(let c=0;c<=1.6;c+=0.005){xs.push(sc(c));ys.push(syl(logLc(c)));}b+=`<path d="${pth(xs,ys)}" fill="none" stroke="var(--ab)" stroke-width="2.6"/>`;
   b+=`<circle cx="${sc(P.c)}" cy="${syl(logLc(P.c))}" r="6" fill="var(--x)"/>`+hAxis(sc,0,1.6,Hh-m.b+8,0.4,'SCALE c');$('gr').innerHTML=b;}},
/* 9 */{title:'Add noise to one photo',lit:['x0','noise','xt'],
 say:"Fix one training photo and call it x₀. Its noisy copies scatter around √ᾱₜ·x₀, the mean for this one photo, with spread √(1 − ᾱₜ), where ᾱₜ is Step 4's kept share.",
 note:'Schedule: Step 4’s cosine schedule, T = 100, offset 0.008, β clipped at 0.999, computed in the page. Distances are root-mean-square per pixel value over all 49,152 values.',
 def:{t:50},pres:[['t = 10',{t:10}],['t = 90',{t:90}],['New draws',{t:50,seedInc:1}]],
 eq(){const a=S.abar[P.t];return `\\begin{aligned}q(${cl('c-x','x_t')}\\mid ${cl('c-x0','x_0')})&=\\mathcal N\\big(\\underbrace{${cl('c-ab','\\sqrt{\\bar\\alpha_t}')}\\,${cl('c-x0','x_0')}}_{\\text{mean, }${f4(R(a))}\\,x_0},\\ \\underbrace{(1-${cl('c-ab','\\bar\\alpha_t')})}_{${f4(1-a)}}\\,I\\big)\\\\${cl('c-x','x_t')}&=${cl('c-ab','\\sqrt{\\bar\\alpha_t}')}\\,${cl('c-x0','x_0')}+${cl('c-nz','\\sqrt{1-\\bar\\alpha_t}')}\\,${cl('c-eps','\\varepsilon')}\\end{aligned}`;},
 eqT:()=>`t=${H('t')}\\text{ of }99,\\qquad ${cl('c-ab','\\bar\\alpha_t')}=${cl('c-ab',f4(S.abar[P.t]))}`,
 check(){if(!ready0)return '';const d=noisy3(),mean=meanImg();const d1=d.map(x=>rms(x,mean,N0)),avg=new Float32Array(N0);for(let i=0;i<N0;i++)avg[i]=(d[0][i]+d[1][i]+d[2][i])/3;
   return `Each draw sits ${f3((d1[0]+d1[1]+d1[2])/3)} from the mean image per pixel (predicted ${f3(R(1-S.abar[P.t]))}). Their average sits ${f3(rms(avg,mean,N0))} (predicted ${f3(R((1-S.abar[P.t])/3))}).`;},
 fig:el=>{el.innerHTML=`<div class="card"><div class="imgrow" style="grid-template-columns:repeat(5,minmax(0,1fr))">
   <figure><canvas class="pix" id="a0"></canvas><figcaption class="c-x0">x₀, one photo</figcaption></figure>
   <figure><canvas class="pix" id="a1"></canvas><figcaption class="c-ab">mean √ᾱₜ·x₀</figcaption></figure>
   <figure><canvas class="pix" id="a2"></canvas><figcaption class="c-x">draw 1</figcaption></figure>
   <figure><canvas class="pix" id="a3"></canvas><figcaption class="c-x">draw 2</figcaption></figure>
   <figure><canvas class="pix" id="a4"></canvas><figcaption class="c-x">draw 3</figcaption></figure></div>
   <p class="howto">The mean image is darker grey because √ᾱₜ shrinks every value toward 0, the middle of the −1 to 1 scale.</p></div>
`;},handoff:[PAGE[3]+'#f1','Sampling never has x₀. How do we go back? Scene 3 →'],
 draw(){if(!ready0)return;paint(X0,$('a0'),128);paint(meanImg(),$('a1'),128);noisy3().forEach((x,k)=>paint(x,$('a'+(k+2)),128));}}
];
function draws16(){const r=rng(P.seed*977+3),out=[];for(let j=0;j<16;j++){const us=[r(),r(),r(),r()].map(u=>Math.min(0.999,Math.max(0.001,u)));out.push(P.cond?drawInOrder(us).x:drawAlone(us));}return out;}
setupPage({no:2,frames:FR2,base:{p1:1,p2:1,p3:-1,p4:-1,d:4,q1:1,mg:false,u1:0.7,u2:0.4,u3:0.6,u4:0.5,cond:false,m1:0.6,sg2:0.3,z0:0.4,two:true,c:0.5,t:50,seed:1},map:MAP2});
