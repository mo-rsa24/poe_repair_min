/* ===== scene 3: going back ===== */
handle('t',0,99,1,'step t');handle('xt',-3.5,3.5,0.01,'noisy value',2);handle('ri',1,10,1,'training height number');handle('nc',1,100,1,'number of chains');
handle('nh',1,10,1,'number of training heights');handle('k',160,190,0.1,'threshold height for tall',1);
loadImgs('','%%DOGS_B64%%');
const P3=PAGE[3];
const ORD=[6,0,9,3,7,1,8,4,2,5];
const prior22=()=>HTS.map(h=>P.skew&&h===ME?5:1);
let c19={};function chains19(){const k=P.nc+'|'+P.seed;if(c19.k!==k)c19={k,v:chain1(P.seed,P.nc,(x,t)=>den1(x,t,Z).x0)};return c19.v;}
const train21=()=>ORD.slice(0,P.nh).map(i=>HTS[i]);
const nearH21=cm=>train21().reduce((b,h)=>Math.min(b,Math.abs(h-cm)),Infinity);
let c21={};function ends21(){const k=P.nh+'|'+P.exact+'|'+P.seed;if(c21.k!==k){const zs=ORD.slice(0,P.nh).map(i=>Z[i]);const m=zs.reduce((a,b)=>a+b,0)/zs.length,s2=Math.max(1e-4,zs.reduce((a,b)=>a+(b-m)**2,0)/zs.length);
  const g=P.exact?(x,t)=>den1(x,t,zs).x0:(x,t)=>denBell(x,t,m,s2);c21={k,v:chain1(P.seed+100,200,g).map(tr=>toCm(tr[100]))};}return c21.v;}
let mdC=null;function meanDog(){if(mdC)return mdC;const o=new Float32Array(D);DOGS.forEach(d=>{for(let i=0;i<D;i++)o[i]+=d[i]/4;});return mdC=o;}
/* the two bells whose product is one step back, over x_{t-1} */
function stepBells(t,xt,x0){const b=S.beta[t],ap=ab(t-1),k=coef(t);return {A:{m:xt/R(1-b),s:R(b/(1-b))},B:{m:R(ap)*x0,s:R(1-ap)},C:{m:k.A*x0+k.B*xt,s:R(k.V)},k};}
const tall=x=>1/(1+Math.exp(-(x-P.k)/2));
const pTall=()=>{let a=0;for(let x=120;x<230;x+=0.01)a+=tall(x+0.005)*npdf(x+0.005,MU0,SG0)*0.01;return a;};

const FR3=[
/* 1 */{title:'One table, read across or down',lit:['xt','bayes','w'],flip:true,
 say:'Each cell holds the chance of one training height and one noisy value together. A row adds noise to one height; a column, divided by its sum, is Bayes: the chance of each height behind one noisy value.',
 note:'Heights are rescaled to mean 0 and spread 1 before noise is added, as images are scaled to −1 to 1. Each cell is p(x₀ = xᵢ, xₜ) = (1/10)·q(xₜ | xᵢ), with q from Step 4\u2019s cosine schedule. A row divided by its prior 1/10 is q(xₜ | xᵢ); a column divided by its sum is p(x₀ = xᵢ | xₜ).',
 def:{t:50,xt:0.3,ri:7,back:false},pres:[['t = 10',{t:10}],['t = 90',{t:90}],['Row: 158 cm',{ri:1}]],
 sw:[{k:'back',on:'Reading down a column: Bayes, backward',off:'Reading along a row: adding noise, forward'}],
 eq:()=>`\\underbrace{p(x_0{=}x_i,\\ ${cl('c-x','x_t')})}_{\\text{one cell}}=\\underbrace{p(x_i)}_{1/10}\\ \\underbrace{q(${cl('c-x','x_t')}\\mid x_i)}_{\\text{a row: forward}}=\\underbrace{p(${cl('c-x','x_t')})}_{\\text{column sum}}\\ \\underbrace{p(x_i\\mid ${cl('c-x','x_t')})}_{\\text{a column: backward}}`,
 eqT:()=>P.back?`t=${H('t')},\\qquad ${cl('c-x','x_t')}=${H('xt')}`:`t=${H('t')},\\qquad \\text{row }${H('ri')}:\\ x_0=${HTS[P.ri-1]}\\text{ cm}`,
 check(){const t=P.t,a=S.abar[t];if(!P.back){const z=Z[P.ri-1];return `Row ${HTS[P.ri-1]} cm: the noisy values form a bell centred at √ᾱₜ × ${f2(z)} = ${f3(R(a)*z)} with spread ${f3(R(1-a))}, area 1 across the row.`;}
   const r=den1(P.xt,t,Z),cs=r.lik.reduce((q,v)=>q+v,0),i=r.w.indexOf(Math.max(...r.w));return `Column xₜ = ${f2(P.xt)}: dividing its ten cells by their sum gives weights adding to ${f4(r.w.reduce((q,v)=>q+v,0))}, largest ${f3(r.w[i])} on ${HTS[i]} cm.`;},
 fig:el=>{el.innerHTML=`<div class="card">${SVG('g',620,420,'A grid of ten training heights by noisy values, with one row or one column picked out below')}</div>`;},
 draw(){const W=620,m={l:64,r:20},sx=lin(-3.5,3.5,m.l,W-m.r),t=P.t,a=S.abar[t],ra=R(a),v=1-a,rows=10,rh=17,top=34;let s=txt(m.l,20,'TRAINING HEIGHT DOWN, NOISY VALUE ACROSS',{c:'ax'});
   let mx=0;const cell=(i,x)=>Math.exp(-((x-ra*Z[i])**2)/(2*v));for(let i=0;i<rows;i++)for(let j=0;j<70;j++)mx=Math.max(mx,cell(i,-3.5+(j+.5)*0.1));
   for(let i=0;i<rows;i++){const y=top+(rows-1-i)*rh;for(let j=0;j<70;j++){const x=-3.5+j*0.1,o=cell(i,x+0.05)/mx;if(o<0.01)continue;s+=`<rect x="${sx(x).toFixed(1)}" y="${y}" width="${(sx(0.1)-sx(0)+0.5).toFixed(1)}" height="${rh}" fill="var(--p)" fill-opacity="${(0.06+0.85*o).toFixed(3)}"/>`;}
     s+=txt(m.l-6,y+12.5,HTS[i],{c:'tick',a:'end'});}
   const gy=top+rows*rh;s+=`<rect x="${m.l}" y="${top}" width="${W-m.l-m.r}" height="${rows*rh}" fill="none" stroke="var(--rule)"/>`;
   if(!P.back){const i=P.ri-1,y=top+(rows-1-i)*rh;s+=`<rect x="${m.l}" y="${y}" width="${W-m.l-m.r}" height="${rh}" fill="none" stroke="var(--nz)" stroke-width="2.4"/>`;
     const by=q=>400-q*R(2*PI*v)*120;s+=txt(m.l,gy+36,'THIS ROW: q(xₜ | x₀), NOISY VALUES FOR ONE HEIGHT',{c:'ax'});s+=`<line class="axis" x1="${m.l}" x2="${W-m.r}" y1="400" y2="400"/>`;
     s+=`<path d="${area(x=>npdf(x,ra*Z[i],R(v)),-3.5,3.5,sx,by,200)}" fill="var(--p-soft)" stroke="var(--p)" stroke-width="2"/>`;[-3,-2,-1,0,1,2,3].forEach(q=>{s+=txt(sx(q),416,mm(q),{c:'tick',a:'middle'});});}
   else{const r=den1(P.xt,t,Z);s+=`<rect x="${sx(P.xt)-3}" y="${top}" width="6" height="${rows*rh}" fill="none" stroke="var(--nz)" stroke-width="2.4"/>`;
     s+=txt(m.l,gy+36,'THIS COLUMN, DIVIDED BY ITS SUM: p(x₀ | xₜ)',{c:'ax'});const hx=lin(150,190,m.l,W-m.r),wm=Math.max(0.2,...r.w),by=q=>400-q/wm*120;
     s+=`<line class="axis" x1="${m.l}" x2="${W-m.r}" y1="400" y2="400"/>`;HTS.forEach((h,i)=>{s+=`<rect x="${hx(h)-6}" y="${by(r.w[i])}" width="12" height="${400-by(r.w[i])}" fill="${i===MEI?'var(--x0)':'var(--x0-soft)'}" stroke="var(--x0)"/>`;});
     [150,160,170,180,190].forEach(q=>{s+=txt(hx(q),416,q,{c:'tick',a:'middle'});});}
   [-3,0,3].forEach(q=>{s+=txt(sx(q),top+rows*rh+14,mm(q),{c:'tick',a:'middle'});});$('g').innerHTML=s;}},
/* 2 */{title:'Prior times likelihood, divided by the total',lit:['xt','bayes','w'],
 say:"Each column is Bayes' theorem: prior times likelihood, divided by the total. The prior is each height's chance before seeing noise, and the likelihood is how well it explains the noisy value.",
 note:'Each row of bars is scaled to its own tallest bar, so compare shapes within a row. With the uneven prior, 173 cm counts five times and the other nine once, then all ten are divided by 14.',
 def:{t:50,xt:0.3,skew:false},pres:[['t = 10',{t:10}],['t = 90',{t:90}]],
 sw:[{k:'skew',on:'Uneven prior: 173 cm counts five times',off:'Even prior: 1 in 10 each'}],
 eq:()=>`p(x_0{=}x_i\\mid ${cl('c-x','x_t')})=\\frac{q(${cl('c-x','x_t')}\\mid x_i)\\,p(x_i)}{\\sum_{j=1}^{10}q(${cl('c-x','x_t')}\\mid x_j)\\,p(x_j)}`,
 eqT:()=>`t=${H('t')},\\qquad ${cl('c-x','x_t')}=${H('xt')}`,
 check(){const r=den1(P.xt,P.t,Z,prior22());const sm=r.w.reduce((a,b)=>a+b,0),i=r.w.indexOf(Math.max(...r.w));return `The posterior adds to ${f4(sm)}. Its largest weight is ${f3(r.w[i])}, on ${HTS[i]} cm.`;},
 fig:el=>{el.innerHTML=`<div class="card">${SVG('g',600,390,'Three rows of ten bars: prior, likelihood and posterior for each training height')}</div>`;},
 draw(){const W=600,m={l:110,r:20},hx=lin(150,190,m.l,W-m.r),pw=prior22(),r=den1(P.xt,P.t,Z,pw),ps=pw.reduce((a,b)=>a+b,0);const lk2=r.lik,lm=Math.max(...lk2);
   const rows=[['prior',pw.map(v=>v/ps),'var(--muted)'],['likelihood',lk2.map(v=>v/lm),'var(--x)'],['posterior',r.w,'var(--x0)']];let s=txt(W-m.r,18,'EACH ROW SCALED TO ITS OWN TALLEST BAR',{c:'ax',a:'end'});
   rows.forEach(([l,vals,col],k)=>{const base=110+k*110,mx=Math.max(...vals,1e-9);s+=txt(m.l-12,base-30,l,{a:'end',s:`fill:${col};font-weight:600`});s+=`<line class="axis" x1="${m.l}" x2="${W-m.r}" y1="${base}" y2="${base}"/>`;
     vals.forEach((v,i)=>{const h=v/mx*76;s+=`<rect x="${hx(HTS[i])-6}" y="${base-h}" width="12" height="${h}" fill="${col}" fill-opacity="${i===MEI?1:.55}"/>`;});});
   s+=hAxis(hx,150,190,334,10,'');s+=txt(W-m.r,382,'TRAINING HEIGHT, CM',{c:'ax',a:'end'});$('g').innerHTML=s;}},
/* 3 */{title:'The best guess is the weighted average',lit:['w','avg','xh'],
 say:'Average the training heights with those weights and you get the best guess x̂₀ of where the noisy value came from. At step 99 the weights are nearly even, so the guess is the mean from scene 1.',
 note:'The weights are exact for a model that knows only these ten heights. The guess is the posterior mean, E[x₀ | xₜ], the value that makes the expected squared error smallest. At t = 99 the kept share ᾱ is about 2.4 × 10⁻⁷, so every height explains the noise almost equally.',
 def:{t:50,xt:0.3},pres:[['t = 99, pure noise',{t:99}],['t = 5, almost clean',{t:5}],['Noisy value −1.5',{xt:-1.5}]],
 eq:()=>`${cl('c-x0','\\hat x_0')}(${cl('c-x','x_t')})=\\sum_{i=1}^{10}w_i\\,x_i,\\qquad w_i=p(x_0{=}x_i\\mid ${cl('c-x','x_t')})`,
 eqT:()=>`t=${H('t')},\\quad ${cl('c-x','x_t')}=${H('xt')},\\quad ${cl('c-x0','\\hat x_0')}=${cl('c-x0',f1(toCm(den1(P.xt,P.t,Z).x0)))}\\text{ cm},\\quad ${lk(PAGE[1]+'#f2','\\bar x')}=${f1(MU0)}`,
 check(){const r=den1(P.xt,P.t,Z),ne=1/r.w.reduce((a,w)=>a+w*w,0);return `The weights spread over the equivalent of ${f1(ne)} heights, so the guess is ${f1(toCm(r.x0))} cm. At t = 99 it is ${f1(toCm(den1(P.xt,99,Z).x0))} cm, the average height x̄ = ${f1(MU0)}.`;},
 fig:el=>{el.innerHTML=`<div class="card">${SVG('g',600,400,'Ten bumps, one per training height, cut by the noisy value; below, the weights and their average')}</div>`;},
 draw(){const W=600,m={l:40,r:20},sx=lin(-3.5,3.5,m.l,W-m.r),a=S.abar[P.t],ra=R(a),v=1-a,r=den1(P.xt,P.t,Z);
   const pk=1/R(2*PI*v),sy=q=>170-q/pk*130;let s=txt(m.l,20,'WHERE EACH HEIGHT\u2019S NOISY COPIES LAND',{c:'ax'});
   Z.forEach((z,i)=>{s+=`<path d="${curve(x=>Math.exp(-((x-ra*z)**2)/(2*v))/R(2*PI*v),-3.5,3.5,sx,sy,160)}" fill="none" stroke="${i===MEI?'var(--x0)':'var(--muted)'}" stroke-opacity="${i===MEI?1:.6}" stroke-width="${i===MEI?2.2:1.3}"/>`;});
   s+=`<line class="axis" x1="${m.l}" x2="${W-m.r}" y1="170" y2="170"/>`;[-3,-2,-1,0,1,2,3].forEach(q=>{s+=txt(sx(q),186,mm(q),{c:'tick',a:'middle'});});
   s+=`<line x1="${sx(P.xt)}" x2="${sx(P.xt)}" y1="34" y2="176" stroke="var(--x)" stroke-width="2.4"/>`+txt(sx(P.xt)+(P.xt<2?8:-8),48,'noisy value',{a:P.xt<2?'start':'end',s:'fill:var(--x)'});
   const hx=lin(150,190,m.l,W-m.r),wm=Math.max(0.2,...r.w),by=v2=>360-v2/wm*110,im=r.w.indexOf(Math.max(...r.w));s+=txt(m.l,222,'WEIGHT ON EACH TRAINING HEIGHT',{c:'ax'});
   HTS.forEach((h,i)=>{s+=`<rect x="${hx(h)-6}" y="${by(r.w[i])}" width="12" height="${360-by(r.w[i])}" fill="${i===MEI?'var(--x0)':'var(--x0-soft)'}" stroke="var(--x0)"/>`;});
   s+=txt(hx(HTS[im]),by(r.w[im])-6,f2(r.w[im]),{c:'num',a:'middle',s:'fill:var(--x0)'});const g=toCm(r.x0);
   s+=`<line x1="${hx(g)}" x2="${hx(g)}" y1="236" y2="362" stroke="var(--x0)" stroke-width="2" stroke-dasharray="5 4"/>`+txt(hx(g)+(g<180?8:-8),246,'guess '+f1(g)+' cm',{c:'num',a:g<180?'start':'end',s:'fill:var(--x0)'});
   s+=hAxis(hx,150,190,364,10,'');$('g').innerHTML=s;}},
/* 4 */{title:'One step back, if x₀ were known',lit:['step','xt1'],
 say:'With xₜ and x₀ both known, two bells pin down the earlier value xₜ₋₁: one from xₜ, one from x₀. Their product is a single bell, so this step back is Gaussian.',
 note:'Bayes again: q(xₜ₋₁ | xₜ, x₀) ∝ q(xₜ | xₜ₋₁) q(xₜ₋₁ | x₀). As a function of xₜ₋₁ the first factor is a bell centred at xₜ/√(1 − βₜ) with variance βₜ/(1 − βₜ); the second is centred at √ᾱₜ₋₁·x₀ with variance 1 − ᾱₜ₋₁. A product of two bells is a bell whose precision is the sum of the two precisions and whose mean is the precision-weighted mean (Ho, Jain and Abbeel 2020, equations 6 and 7). The page checks it by multiplying the two bells on a grid.',
 def:{t:50,xt:0.3,ri:7},pres:[['t = 5',{t:5}],['t = 95',{t:95}],['x₀ = 158 cm',{ri:1}]],
 eq(){const k=coef(P.t);return `\\begin{aligned}q(${cl('c-x','x_{t-1}')}\\mid ${cl('c-x','x_t')},${cl('c-x0','x_0')})&=\\mathcal N\\big(a_t\\,${cl('c-x0','x_0')}+b_t\\,${cl('c-x','x_t')},\\ \\sigma_t^2\\big)\\\\a_t&=\\tfrac{\\sqrt{\\bar\\alpha_{t-1}}\\,\\beta_t}{1-\\bar\\alpha_t}=${f4(k.A)}\\\\b_t&=\\tfrac{\\sqrt{1-\\beta_t}\\,(1-\\bar\\alpha_{t-1})}{1-\\bar\\alpha_t}=${f4(k.B)}\\\\\\sigma_t^2&=\\tfrac{\\beta_t(1-\\bar\\alpha_{t-1})}{1-\\bar\\alpha_t}=${tsci(k.V)}\\end{aligned}`;},
 eqT:()=>`t=${H('t')},\\quad ${cl('c-x','x_t')}=${H('xt')},\\quad x_0=\\text{height }${H('ri')}=${HTS[P.ri-1]}\\text{ cm}`,
 check(){const sb=stepBells(P.t,P.xt,Z[P.ri-1]);let w=0,m1=0,m2=0;const lo=sb.C.m-8*sb.C.s,hi=sb.C.m+8*sb.C.s,st=(hi-lo)/4000;for(let x=lo;x<hi;x+=st){const p=npdf(x,sb.A.m,sb.A.s)*npdf(x,sb.B.m,sb.B.s);w+=p;m1+=p*x;m2+=p*x*x;}
   const mu=m1/w,vr=m2/w-mu*mu;const fv=v=>v<1e-3?sci(v):f4(v);return `Multiplying the two bells on a grid gives mean ${f4(mu)} and variance ${fv(vr)}; the formula gives ${f4(sb.C.m)} and ${fv(sb.C.s**2)}.`;},
 fig:el=>{el.innerHTML=`<div class="card">${SVG('g',600,330,'Two bells over the earlier value and their product')}</div>`;},
 draw(){const sb=stepBells(P.t,P.xt,Z[P.ri-1]),W=600,Hh=330,m={l:40,r:20,t:30,b:52};const lo=Math.min(sb.A.m-3.5*sb.A.s,sb.B.m-3.5*sb.B.s),hi=Math.max(sb.A.m+3.5*sb.A.s,sb.B.m+3.5*sb.B.s),sx=lin(lo,hi,m.l,W-m.r);
   const pk=Math.max(npdf(0,0,sb.A.s),npdf(0,0,sb.B.s),npdf(0,0,sb.C.s)),sy=q=>Hh-m.b-q/pk*(Hh-m.t-m.b);let s=txt(m.l,m.t-12,'BELLS OVER THE EARLIER VALUE xₜ₋₁',{c:'ax'});
   s+=`<path d="${curve(x=>npdf(x,sb.A.m,sb.A.s),lo,hi,sx,sy,300)}" fill="none" stroke="var(--x)" stroke-width="2.2"/><path d="${curve(x=>npdf(x,sb.B.m,sb.B.s),lo,hi,sx,sy,300)}" fill="none" stroke="var(--x0)" stroke-width="2.2"/>`;
   s+=`<path d="${area(x=>npdf(x,sb.C.m,sb.C.s),lo,hi,sx,sy,400)}" fill="var(--ab-soft)" stroke="var(--ab)" stroke-width="2.6"/>`;
   const lab=(m0,s0,t0,col,side)=>{const x=sx(m0)+side*(8+s0/(hi-lo)*(W-m.l-m.r)),y=Math.max(m.t+14,sy(npdf(m0+side*s0,m0,s0)));return txt(Math.min(W-m.r-4,Math.max(m.l+4,x)),y,t0,{a:side<0?'end':'start',s:`fill:${col};font-weight:600`});};
   s+=lab(sb.A.m,sb.A.s,'from xₜ','var(--x)',-1)+lab(sb.C.m,sb.C.s,'product','var(--ab)',1)+lab(sb.B.m,sb.B.s,'from x₀','var(--x0)',sb.B.m<sb.A.m?-1:1);
   s+=hAxis(sx,Math.ceil(lo*2)/2,Math.floor(hi*2)/2,Hh-m.b+8,(hi-lo)>3?1:0.5,'RESCALED UNITS');$('g').innerHTML=s;}},
/* 5 */{title:'In practice, use the best guess',lit:['xh','step','xt1'],
 say:'Sampling has no x₀, so the step uses x̂₀ instead. The exact step back mixes ten bells, one per height; a single bell built on x̂₀ has the same mean and a little less spread.',
 note:'The exact reverse step for this training set is p(xₜ₋₁ | xₜ) = Σᵢ wᵢ q(xₜ₋₁ | xₜ, xᵢ). Its mean is aₜ x̂₀ + bₜ xₜ, the same as the single bell, because the mean of a mix is the weighted mean of its parts; its variance is σₜ² + aₜ²·Var_w(xᵢ). The gap aₜ²·Var_w(xᵢ) shrinks as steps get small, which is assumption 3.',
 def:{t:50,xt:0.3},pres:[['t = 99',{t:99}],['t = 3',{t:3}],['t = 3, xₜ = 0.05',{t:3,xt:0.05}]],
 eq:()=>`p(${cl('c-x','x_{t-1}')}\\mid ${cl('c-x','x_t')})=\\sum_{i=1}^{10}w_i\\,q(${cl('c-x','x_{t-1}')}\\mid ${cl('c-x','x_t')},x_i)\\ \\approx\\ \\mathcal N\\big(a_t\\,${cl('c-x0','\\hat x_0')}+b_t\\,${cl('c-x','x_t')},\\ \\sigma_t^2\\big)`,
 eqT:()=>`t=${H('t')},\\qquad ${cl('c-x','x_t')}=${H('xt')}`,
 check(){const r=den1(P.xt,P.t,Z),k=coef(P.t),vw=r.w.reduce((a,w,i)=>a+w*(Z[i]-r.x0)**2,0);return `Both means are ${f4(k.A*r.x0+k.B*P.xt)}. Spread: exact mix ${f4(R(k.V+k.A*k.A*vw))}, single bell ${f4(R(k.V))}.`;},
 fig:el=>{el.innerHTML=`<div class="card">${SVG('g',600,330,'The exact reverse step as a mix of ten bells, and the single bell that uses the best guess')}</div>`;},
 draw(){const r=den1(P.xt,P.t,Z),k=coef(P.t),W=600,Hh=330,m={l:40,r:20,t:30,b:52},sv=R(k.V),vw=r.w.reduce((a,w,i)=>a+w*(Z[i]-r.x0)**2,0),mean=k.A*r.x0+k.B*P.xt,sd=R(k.V+k.A*k.A*vw);
   const lo=mean-4.5*sd,hi=mean+4.5*sd,sx=lin(lo,hi,m.l,W-m.r);const mix=x=>r.w.reduce((a,w,i)=>a+w*npdf(x,k.A*Z[i]+k.B*P.xt,sv),0);
   let pk=0;for(let i=0;i<=400;i++){const x=lo+(hi-lo)*i/400;pk=Math.max(pk,mix(x),npdf(x,mean,sv));}const sy=q=>Hh-m.b-q/pk*(Hh-m.t-m.b);
   let s=txt(m.l,m.t-12,'OVER THE EARLIER VALUE xₜ₋₁',{c:'ax'});s+=`<path d="${area(mix,lo,hi,sx,sy,500)}" fill="var(--p-soft)" stroke="var(--p)" stroke-width="2.2"/>`;
   s+=`<path d="${curve(x=>npdf(x,mean,sv),lo,hi,sx,sy,500)}" fill="none" stroke="var(--ab)" stroke-width="2.6" stroke-dasharray="6 4"/>`;
   s+=txt(W-m.r,m.t+8,'shaded: exact mix of ten',{a:'end',s:'fill:var(--p);font-weight:600'})+txt(W-m.r,m.t+28,'dashed: one bell with x̂₀',{a:'end',s:'fill:var(--ab);font-weight:600'});
   const st=(hi-lo)>4?1:(hi-lo)>1.5?0.5:(hi-lo)>0.6?0.2:0.05;s+=hAxis(sx,Math.ceil(lo/st)*st,Math.floor(hi/st)*st,Hh-m.b+8,st,'RESCALED UNITS',v=>mm(+v.toFixed(2)));$('g').innerHTML=s;}},
/* 6 */{title:'Step back from noise to a height',lit:['step','xt1','rep','smp'],
 say:'Start from pure noise at step 99. Each step guesses x₀, moves part of the way toward the guess and adds a little fresh noise, and every chain ends on a height.',
 note:'Each line is one chain run with the step of frame 5. A chain ends on a height when it lands within 0.5 cm of one of the ten.',
 def:{nc:20,t:50},pres:[['5 chains',{nc:5}],['100 chains',{nc:100}],['New noise',{seedInc:1}]],
 eq(){const k=coef(P.t);return `${cl('c-x','x_{t-1}')}=\\underbrace{${f4(k.A)}}_{a_t}\\,${cl('c-x0','\\hat x_0')}+\\underbrace{${f4(k.B)}}_{b_t}\\,${cl('c-x','x_t')}+\\underbrace{${f4(R(k.V))}}_{\\sigma_t}\\,${cl('c-eps','\\varepsilon_t')}`;},
 eqT:()=>`t=${H('t')},\\qquad\\text{chains}=${H('nc')}`,
 check(){const c=chains19();const k=c.filter(tr=>nearH(toCm(tr[100]))<0.5).length;return `${k} of ${c.length} chains end within 0.5 cm of a training height.`;},
 fig:el=>{el.innerHTML=`<div class="card">${SVG('g',600,380,'Chains of heights from step 99 on the left to clean on the right, ending on the ten training heights')}</div>`;},
 draw(){const W=600,Hh=380,m={l:50,r:40,t:28,b:48},sx=j=>m.l+j/100*(W-m.l-m.r),sy=lin(140,200,Hh-m.b,m.t),c=chains19();
   let s=vAxis(sy,m.l,W-m.r,[150,160,170,180,190],v=>v,'HEIGHT, CM',m.t-12);
   HTS.forEach(h=>{s+=`<line x1="${W-m.r+2}" x2="${W-m.r+14}" y1="${sy(h)}" y2="${sy(h)}" stroke="var(--x0)" stroke-width="2.4"/>`;});
   c.forEach(tr=>{const xs=[],ys=[];for(let j=0;j<=100;j++){xs.push(sx(j));ys.push(sy(Math.max(140,Math.min(200,toCm(tr[j])))));}s+=`<path class="draw" pathLength="1" d="${pth(xs,ys)}" fill="none" stroke="var(--x)" stroke-opacity="${c.length>30?.35:.7}" stroke-width="1.4"/>`;});
   const jx=sx(99-P.t);s+=`<line x1="${jx}" x2="${jx}" y1="${m.t}" y2="${Hh-m.b}" stroke="var(--nz)" stroke-width="1.5"/>`;
   [[0,'99'],[49,'50'],[99,'0']].forEach(([j,l])=>{s+=txt(sx(j),Hh-m.b+18,l,{c:'tick',a:'middle'});});s+=txt(sx(100)+4,Hh-m.b+18,'clean',{c:'tick',a:'start'});
   s+=txt(m.l,Hh-6,'STEP T, COUNTING DOWN',{c:'ax'});$('g').innerHTML=s;}},
/* 7 */{title:'The same steps end on a dog',lit:['rep','smp'],
 say:'Run the same steps on four dog photos, starting from a pure-noise image. At step 99 the guess is the average of all four dogs, and it sharpens into one dog as t falls.',
 note:'The best guess here is exact for a model that knows only these four photos; a trained network learns an approximation of it from the same photos. Switching the fresh noise off turns the 100 steps into a fixed function of the starting noise, the generator of scene 2. Click a thumbnail to jump to that step.',
 def:{t:97,fresh:true},pres:[['t = 99',{t:99}],['t = 0',{t:0}],['New starting noise',{seedInc:1}]],
 sw:[{k:'fresh',on:'Fresh noise at each step',off:'No fresh noise: a fixed map'}],
 eq:()=>`${cl('c-x0','\\hat x_0')}(${cl('c-x','x_t')})=\\sum_{k=1}^{4}w_k\\,x^{(k)},\\qquad w_k\\propto\\exp\\!\\Big(-\\frac{\\lVert ${cl('c-x','x_t')}-${cl('c-ab','\\sqrt{\\bar\\alpha_t}')}\\,x^{(k)}\\rVert^2}{2(1-${cl('c-ab','\\bar\\alpha_t')})}\\Big)`,
 eqT(){if(!readyD)return `t=${H('t')}`;const c=chainImg(P.seed,P.fresh),w=c.ws[99-P.t];return `t=${H('t')},\\qquad w=(${w.map(f2).join(',\\,')})`;},
 check(){if(!readyD)return '';const c=chainImg(P.seed,P.fresh),avgD=rms(c.hs[0],meanDog(),D);
   return `The chain ends on ${DOGNAME[c.end]}, ${c.dist<0.0005?'matching it to three decimals':f3(c.dist)+' from it per value'}. At step 99 the guess is ${f3(avgD)} from the average of the four dogs.`;},
 fig:el=>{el.innerHTML=`<div class="card"><div class="imgrow top3" style="grid-template-columns:repeat(3,minmax(0,1fr))">
   <figure><canvas class="pix" id="b0"></canvas><figcaption class="c-x" id="b0c">xₜ</figcaption></figure>
   <figure><canvas class="pix" id="b1"></canvas><figcaption class="c-x0">best guess x̂₀</figcaption></figure>
   <figure><canvas class="pix" id="b2"></canvas><figcaption>the clean end</figcaption></figure></div>
   <div class="wbars" id="wb"></div><div class="snaps" id="sn" data-optional></div></div>`;
   const ts=[99,98,97,96,95,50,0];$('sn').innerHTML=ts.map(t=>`<button data-t="${t}" aria-label="Jump to step ${t}"><canvas width="48" height="48"></canvas>t = ${t}</button>`).join('');
   $('sn').querySelectorAll('button').forEach(b=>b.onclick=()=>{P.t=+b.dataset.t;userMoved();inval();});},
 draw(){if(!readyD)return;const c=chainImg(P.seed,P.fresh),j=99-P.t;paint(c.xs[j],$('b0'),48);paint(c.hs[j],$('b1'),48);paint(c.xs[100],$('b2'),48);$('b0c').textContent='x'+sub(P.t);
   $('wb').innerHTML=c.ws[j].map((w,k)=>`<div><img src="${THUMB[k]}" alt="${DOGNAME[k]}"><span class="bar"><i style="width:${(w*100).toFixed(1)}%"></i></span>${f2(w)}</div>`).join('');
   $('sn').querySelectorAll('button').forEach(b=>{const t=+b.dataset.t;b.setAttribute('aria-pressed',t===P.t);paint(c.xs[99-t],b.querySelector('canvas'),48);});}},
/* 8 */{title:'A label enters by Bayes too',lit:['pz','lab','pc'],
 say:'Asking for a dog, or here for a tall person, multiplies the density by the chance of that label at each x, then divides by the total. The result is a new density, p(x | tall), with area 1.',
 note:'p(tall | x) here is a smooth step chosen for illustration, 1/(1 + e^(−(x − k)/2)); nothing on the page trains a classifier. Outside this page, classifier guidance steers a diffusion sampler with ∇log p(x | y) = ∇log p(x) + ∇log p(y | x) (Dhariwal and Nichol 2021): the label\u2019s term adds to the score, and the normaliser P(y) drops out.',
 def:{k:175},pres:[['k = 165',{k:165}],['k = 185',{k:185}]],
 eq:()=>`\\begin{aligned}${cl('c-ab','p(x\\mid\\text{tall})')}&=\\frac{p(\\text{tall}\\mid x)\\ ${lk(PAGE[1]+'#f4',cl('c-p','p(x)'))}}{P(\\text{tall})}\\\\P(\\text{tall})&=\\int p(\\text{tall}\\mid x)\\,${cl('c-p','p(x)')}\\,dx=${f4(pTall())}\\end{aligned}`,
 eqT:()=>`p(\\text{tall}\\mid x)=\\frac{1}{1+e^{-(x-k)/2}},\\qquad k=${H('k')}\\text{ cm}`,
 check(){const pt=pTall();let a=0,m=0;for(let x=120;x<230;x+=0.01){const v=tall(x+0.005)*npdf(x+0.005,MU0,SG0)/pt;a+=v*0.01;m+=v*(x+0.005)*0.01;}return `Area under p(x | tall): ${f4(a)}. Its mean is ${f1(m)} cm, up from ${f1(MU0)}; a share ${f3(pt)} of everyone counts as tall.`;},
 fig:el=>{el.innerHTML=`<div class="card">${SVG('g',600,400,'The chance of the label tall at each height, the bell, and the bell given tall')}</div>`;},
 draw(){const W=600,m={l:56,r:20},sx=lin(140,205,m.l,W-m.r),pt=pTall();const sy1=v=>110-v*80;let s=txt(m.l,20,'CHANCE OF THE LABEL, p(tall | x)',{c:'ax'});
   s+=`<line class="axis" x1="${m.l}" x2="${W-m.r}" y1="110" y2="110"/><path d="${curve(tall,140,205,sx,sy1)}" fill="none" stroke="var(--nz)" stroke-width="2.4"/>`+txt(m.l-6,sy1(1)+4,'1',{c:'tick',a:'end'})+txt(m.l-6,114,'0',{c:'tick',a:'end'});
   const ym=0.1,sy=v=>350-v/ym*190;s+=txt(m.l,148,'DENSITY, PER CM',{c:'ax'});s+=vAxis(sy,m.l,W-m.r,[0,0.05,0.1],v=>f2(v),'',0);
   s+=`<path d="${curve(v=>npdf(v,MU0,SG0),140,205,sx,sy)}" fill="none" stroke="var(--p)" stroke-width="2.2"/><path d="${area(v=>tall(v)*npdf(v,MU0,SG0)/pt,140,205,sx,sy,300)}" fill="var(--ab-soft)" stroke="var(--ab)" stroke-width="2.6"/>`;
   s+=`<line x1="${sx(P.k)}" x2="${sx(P.k)}" y1="30" y2="350" stroke="var(--nz)" stroke-dasharray="4 4"/>`;
   s+=txt(sx(160),sy(npdf(160,MU0,SG0))-10,'p(x)',{a:'end',s:'fill:var(--p);font-weight:600'});let px=150;for(let x=150;x<200;x+=0.5)if(tall(x)*npdf(x,MU0,SG0)>tall(px)*npdf(px,MU0,SG0))px=x;
   s+=txt(sx(px)+12,sy(tall(px)*npdf(px,MU0,SG0)/pt)-6,'p(x | tall)',{s:'fill:var(--ab);font-weight:600'});
   s+=hAxis(sx,140,200,356,10,'HEIGHT, CM');$('g').innerHTML=s;}},
/* 9 */{title:'What the training data decides',lit:['data','dec','out'],
 say:'With the exact best guess the sampler can only return training heights, so the data is everything it knows. A smoother guess, such as a fitted bell, returns new heights in between.',
 note:'Outside this page: an exact denoiser makes a diffusion model reproduce its training set, and trained networks produce new images because they cannot represent that exact denoiser (Kadkhodaie et al. 2024). The fitted bell stands in for a smoother model; it is not a network. Heights join the training set in the order 173, 158, 184, 168, 176, 163, 179, 170, 165, 172.',
 def:{nh:10,exact:true},pres:[['2 training heights',{nh:2}],['5 training heights',{nh:5}]],
 sw:[{k:'exact',on:'Best guess from the training heights',off:'Best guess from a fitted bell'}],
 eq:()=>P.exact?`${cl('c-x0','\\hat x_0')}(${cl('c-x','x_t')})=\\sum_{i=1}^{${P.nh}}w_i\\,x_i`:`${cl('c-x0','\\hat x_0')}(${cl('c-x','x_t')})=m+\\frac{s^2\\sqrt{\\bar\\alpha_t}\\,(${cl('c-x','x_t')}-\\sqrt{\\bar\\alpha_t}\\,m)}{\\bar\\alpha_t s^2+1-\\bar\\alpha_t}`,
 eqT:()=>`\\text{training heights}=${H('nh')}`,
 check(){const e=ends21();const k=e.filter(cm=>nearH21(cm)<0.5).length;return `${k} of ${e.length} chains (${Math.round(100*k/e.length)}%) end within 0.5 cm of one of the ${P.nh} training heights${P.exact?'':`; the other ${e.length-k} are heights the data never had`}.`;},
 fig:el=>{el.innerHTML=`<div class="card">${SVG('g',600,340,'Histogram of 200 sampled heights with the training heights marked')}</div>`;},
 draw(){const W=600,Hh=340,m={l:50,r:20,t:30,b:60},sx=lin(140,205,m.l,W-m.r),e=ends21(),c={};e.forEach(cm=>{const b=Math.round(cm);c[b]=(c[b]||0)+1;});
   const ym=Math.max(10,...Object.values(c)),sy=v=>Hh-m.b-v/ym*(Hh-m.t-m.b);let s=vAxis(sy,m.l,W-m.r,[0,Math.round(ym/2),ym],v=>v,'CHAINS ENDING NEAREST EACH WHOLE CM',m.t-12);
   Object.entries(c).forEach(([b,n])=>{const a=+b-0.5;if(a<140||a>=205)return;s+=`<rect x="${sx(a)}" y="${sy(n)}" width="${Math.max(1,sx(a+1)-sx(a)-1)}" height="${Hh-m.b-sy(n)}" fill="var(--x)"/>`;});
   train21().forEach(h=>{s+=`<circle cx="${sx(h)}" cy="${Hh-m.b+12}" r="5" fill="var(--x0)"/>`;});
   s+=hAxis(sx,140,200,Hh-m.b+22,10,'');s+=txt(W-m.r,Hh-4,'HEIGHT, CM',{c:'ax',a:'end'});$('g').innerHTML=s;}},
/* 10 */{title:'How many assumptions it takes',lit:['data','dec'],
 say:'Five assumptions carry the whole chain from heights to a dog. Each links to the frame where it does its work.',
 note:'Assumption 4: the log of a Gaussian density is −(x − μ)²/2σ² plus a constant, so maximising Gaussian likelihood is the same as minimising squared error, the training loss of Ho, Jain and Abbeel (2020).',
 check:()=>'One of the five, Gaussian forward noise, is a free choice. The other four are bets about the data and the network.',
 fig:el=>{const A=[[PAGE[2]+'#f2','The training photos are independent draws from one p(x | dog).','scene 2, frame 2'],[PAGE[2]+'#f9','The noise added going forward is Gaussian. This one is a choice.','scene 2, frame 9',1],
   [P3+'#f5','Steps are small, so each step back is close to one bell.','scene 3, frame 5'],[PAGE[1]+'#f7','Training by squared error is training by Gaussian likelihood.','scene 1, frame 7'],[P3+'#f3','The network can learn the best guess x̂₀(xₜ).','scene 3, frame 3']];
   el.innerHTML=`<div class="alist">${A.map(([h,t,l,c],i)=>`<a class="${c?'choice':''}" href="${h}">${i+1}. ${t}<span>${l} · ${c?'a choice':'a bet'}</span></a>`).join('')}</div>`;},
 draw(){}},
/* 11 */{title:'Does the architecture carry probability?',lit:['xh'],
 say:'Swap a convnet for a transformer and the loss and the sampler stay the same. The network is a fixed function, and what its design adds is a preference for which functions are easy to learn.',
 note:'Outside this page: locality and shared weights in convolution, and the all-pairs comparison in attention, are built-in preferences (inductive biases). Neither layer computes a probability; the probability comes from the Gaussian noise, the squared-error loss and the sampler.',
 def:{tf:false},sw:[{k:'tf',on:'Inside the box: a transformer',off:'Inside the box: a convnet'}],
 eq:()=>`\\begin{aligned}\\text{loss}&=\\big\\lVert ${cl('c-eps','\\varepsilon')}-\\hat\\varepsilon_\\theta(${cl('c-x','x_t')},t)\\big\\rVert^2&&\\text{unchanged}\\\\${cl('c-x','x_{t-1}')}&=a_t\\,${cl('c-x0','\\hat x_0')}+b_t\\,${cl('c-x','x_t')}+\\sigma_t\\,${cl('c-eps','\\varepsilon_t')}&&\\text{unchanged}\\end{aligned}`,
 check:()=>P.tf?'Transformer: every patch compares itself with every other patch from the first layer, with fewer built-in bets about distance.':'Convnet: each output looks at a small neighbourhood with the same filter everywhere, a built-in bet that nearby pixels matter most.',
 fig:el=>{el.innerHTML=`<div class="card">${SVG('g',640,300,'A noisy image goes into the network box and a noise prediction comes out; the box contents change with the switch')}</div>`;},
 draw(){let s=`<rect x="20" y="100" width="90" height="90" fill="var(--x-soft)" stroke="var(--x)"/>`+txt(65,214,'noisy image',{a:'middle',s:'fill:var(--x)'});
   s+=`<line x1="116" x2="196" y1="145" y2="145" stroke="var(--muted)" stroke-width="2"/><rect x="200" y="50" width="240" height="190" rx="10" fill="var(--shade)" stroke="var(--ink)" stroke-width="2"/>`;
   const gx=232,gy=78,cz=17;for(let i=0;i<8;i++)for(let j=0;j<8;j++)s+=`<rect x="${gx+i*cz}" y="${gy+j*cz}" width="${cz}" height="${cz}" fill="var(--panel)" stroke="var(--rule)"/>`;
   if(!P.tf){s+=`<rect x="${gx+2*cz}" y="${gy+3*cz}" width="${3*cz}" height="${3*cz}" fill="var(--x0-soft)" stroke="var(--x0)" stroke-width="2.5"/>`;s+=txt(320,228,'3 × 3 filter, slid everywhere',{a:'middle',c:'num'});}
   else{const c=[gx+3.5*cz,gy+3.5*cz];[[0,0],[7,0],[0,7],[7,7],[1,5],[6,2],[3,7],[5,0]].forEach(([i,j])=>{s+=`<line x1="${c[0]}" y1="${c[1]}" x2="${gx+(i+.5)*cz}" y2="${gy+(j+.5)*cz}" stroke="var(--x0)" stroke-width="1.6"/>`;});s+=`<circle cx="${c[0]}" cy="${c[1]}" r="6" fill="var(--x0)"/>`;s+=txt(320,228,'one patch attends to all',{a:'middle',c:'num'});}
   s+=`<line x1="444" x2="520" y1="145" y2="145" stroke="var(--muted)" stroke-width="2"/><rect x="526" y="100" width="90" height="90" fill="var(--eps-soft)" stroke="var(--eps)"/>`+txt(571,214,'predicted noise',{a:'middle',s:'fill:var(--eps)'});
   s+=txt(320,34,P.tf?'TRANSFORMER':'CONVNET',{a:'middle',c:'ax'});$('g').innerHTML=s;}},
/* 12 */{title:'The whole map',fullMap:[MAP1[0],MAP2[0],MAP3[0]],
 say:'Every box above is a concept and every arrow an operation that turns one into the next. Click any box to go to the frame that teaches it.',
 note:'Top row: scene 1, one number. Middle row: scene 2, many numbers. Bottom row: scene 3, going back.',
 check:()=>'Three scenes, one toy each: ten heights, a 2 × 2 image, four dogs. The same moves carry across: a density, a fixed map from noise, and Bayes to go back.',
 fig:el=>{const C=[[PAGE[1],'One number','A sample is one height; fitting picks the bell by likelihood; F⁻¹ turns a random u into a height.'],[PAGE[2],'Many numbers','An image is a point; pixels must be drawn in order; G turns Gaussian noise into an image.'],[P3,'Going back','Bayes turns the noise step around; the best guess is a weighted average; small steps back end on a sample.']];
   el.innerHTML=`<div class="alist">${C.map(([h,t,d],i)=>`<a href="${h}#f1">${i+1}. ${t}<span>${d}</span></a>`).join('')}</div>`;},
 draw(){}}
];
setupPage({no:3,frames:FR3,base:{t:50,xt:0.3,ri:7,back:false,skew:false,nc:20,seed:1,fresh:true,nh:10,exact:true,k:175,tf:false},map:MAP3});
