/* Live replay of advisory runs. Needs d3 v7 loaded first (global `d3`).
   Usage:
     AdvisoryReplay.mount(document.getElementById('replay'), {
       runs: rows.map(AdvisoryReplay.normalize),   // in replay order
       setting: 'dedicated',                       // key into NAMES below
       measured: true,                             // false shows a "sample" banner
     });
   Returns {destroy(), restart(), setSpeed(n), toggle()}.
   Adapted for the CS 5130 dashboard from the team's reference module (assets/replay.js):
   colours come from the page's CSS variables, unreadable verdicts get their own stamp,
   failed runs are not replayed, and the controls announce their state (aria-pressed). */
(function(){
'use strict';
const TOKENS={cy:['--cyan','#22D3EE'],vi:['--violet','#8B5CF6'],vs:['--violet-soft','#C4B5FD'],bad:['--pink','#FF5C8A'],amber:['--amber','#FBBF24'],mute:['--muted','#A9A4C8'],line:['--line','#2A2650'],fg:['--text','#ECEAF7'],panel:['--card-top','#1A1735'],bg:['--bg','#07060F']};
const C={};
function readTokens(){const cs=getComputedStyle(document.documentElement);Object.entries(TOKENS).forEach(([k,[v,f]])=>{C[k]=(cs.getPropertyValue(v)||'').trim()||f})}
readTokens();
const NAMES={none:['No verifier','answer goes straight out'],self:['Self-check','advisor re-reads'],peer:['Peer check','second advisor'],dedicated:['Dedicated verifier','+ calculator'],weakened:['Verifier','no calculator'],cross_family:['Cross-family verifier','gpt-oss-20b + calculator'],'cross-family':['Cross-family verifier','gpt-oss-20b + calculator']};
const lerp=(a,b,t)=>a+(b-a)*Math.max(0,Math.min(1,t));
const seg=(p,a,b)=>Math.max(0,Math.min(1,(p-a)/(b-a)));
const reduce=()=>window.matchMedia&&matchMedia('(prefers-reduced-motion: reduce)').matches;

/* Map one dashboard runs.json row to what the replay needs.
   Real values: verdict is "accept" | "reject" | "abstain" | null (null = the verifier's reply
   could not be read, or no verifier for `none`); status is "unchecked" | "accepted" |
   "revised" | "human_review" | "failed"; human_review_route is "abstain" | "unreadable" |
   "revision_not_accepted". Failed runs return null and are not replayed. */
function normalize(r){
  if(r.status==='failed')return null;
  const raw=String(r.verdict||'').toLowerCase();
  const hr=r.status==='human_review';
  let v='none';
  if(r.condition&&r.condition!=='none'){ v=raw==='reject'?'reject':raw==='abstain'?'abstain':raw==='accept'?'accept':'unreadable'; }
  return {id:r.run_id,task:r.task_id,draftCorrect:!!r.initial_correct,finalCorrect:!!r.final_correct,verdict:v,humanReview:hr,route:r.human_review_route||null,cost:+r.cost_usd||0};
}

const CSS=()=>`.ar-grid{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(250px,1fr);gap:16px;align-items:start}
@media(max-width:860px){.ar-grid{grid-template-columns:1fr}}
.ar-side{display:flex;flex-direction:column;gap:12px;min-width:0}
.ar-ctls{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}
.ar-btn{font:inherit;font-size:13px;color:${C.mute};background:${C.panel};border:1px solid ${C.line};border-radius:8px;padding:7px 14px;cursor:pointer}
.ar-btn:hover{color:${C.fg}}.ar-btn.on{color:#fff;border-color:${C.cy};box-shadow:0 0 14px rgba(34,211,238,.4)}
.ar-btn:focus-visible{outline:2px solid ${C.fg};outline-offset:2px}
.ar-cnts{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.ar-cnt{background:rgba(255,255,255,.04);border:1px solid ${C.line};border-radius:10px;padding:8px 10px;display:flex;flex-direction:column;gap:2px}
.ar-cnt span{font-size:12px;color:${C.mute}}.ar-cnt b{font-size:21px;font-variant-numeric:tabular-nums;color:#fff}
.ar-log{display:flex;flex-direction:column;gap:4px;font-size:12px;min-height:150px;overflow:hidden}
.ar-log div{display:flex;gap:8px;padding:5px 8px;border-radius:6px;background:rgba(255,255,255,.035);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ar-banner{font-size:12px;color:${C.amber};margin-top:6px}
svg.ar text{fill:${C.mute};font-size:12px}`;

function mount(el,opts){
  readTokens();
  const runs=(opts.runs||[]).filter(Boolean), N=runs.length, setting=opts.setting||'none';
  const nm=NAMES[setting]||['Verifier',''];
  if(!document.getElementById('ar-css')){const s=document.createElement('style');s.id='ar-css';s.textContent=CSS();document.head.appendChild(s)}
  const root=d3.select(el);root.selectAll('*').remove();
  const st={paused:false,speed:1,idx:0,pt:0,wait:0,last:0,tot:null,hist:[]};
  const ctl=root.append('div').attr('class','ar-ctls');
  const bPlay=ctl.append('button').attr('type','button').attr('class','ar-btn').attr('aria-pressed','false').attr('aria-label','Pause the replay').text('Pause');
  [1,2,4].forEach(v=>ctl.append('button').attr('type','button').attr('class','ar-btn ar-spd').attr('data-v',v).attr('aria-label','Speed '+v+'×').attr('aria-pressed',String(v===1)).text(v+'×').on('click',()=>api.setSpeed(v)));
  ctl.append('button').attr('type','button').attr('class','ar-btn').text('Restart').on('click',()=>api.restart());
  const grid=root.append('div').attr('class','ar-grid');
  const L=grid.append('div').style('min-width',0),R=grid.append('div').attr('class','ar-side');
  const svg=L.append('svg').attr('class','ar').attr('width','100%').attr('viewBox','0 0 760 330').attr('role','img').attr('aria-label','Animated replay of '+N+' advisor runs through the verification pipeline');
  const df=svg.append('defs'),gl=df.append('filter').attr('id','ar-glow').attr('x','-60%').attr('y','-60%').attr('width','220%').attr('height','220%');
  gl.append('feGaussianBlur').attr('stdDeviation',5).attr('result','b');const mg=gl.append('feMerge');mg.append('feMergeNode').attr('in','b');mg.append('feMergeNode').attr('in','SourceGraphic');
  const box=(x,y,w,h,l1,l2)=>{svg.append('rect').attr('x',x).attr('y',y).attr('width',w).attr('height',h).attr('rx',14).style('fill',C.panel).style('stroke',C.line);svg.append('text').attr('x',x+w/2).attr('y',y+22).attr('text-anchor','middle').style('fill',C.fg).style('font-weight',600).text(l1);svg.append('text').attr('x',x+w/2).attr('y',y+39).attr('text-anchor','middle').text(l2||'')};
  [[140,190],[330,400],[550,620]].forEach(p=>svg.append('line').attr('x1',p[0]).attr('y1',145).attr('x2',p[1]).attr('y2',145).style('stroke',C.line).attr('stroke-width',2));
  svg.append('path').attr('d','M475 205 C475 262 260 262 260 190').style('fill','none').style('stroke',C.line).attr('stroke-width',2).attr('stroke-dasharray','4 5');
  svg.append('text').attr('x',370).attr('y',268).attr('text-anchor','middle').style('font-size','11px').text('reject: revise once');
  box(20,110,120,70,'Task');box(190,110,140,70,'Advisor',opts.advisorLabel||'gemini-3.6-flash');
  const verBox=svg.append('rect').attr('x',400).attr('y',95).attr('width',150).attr('height',100).attr('rx',14).style('fill',C.panel).style('stroke',C.line);
  svg.append('text').attr('x',475).attr('y',122).attr('text-anchor','middle').style('fill',C.fg).style('font-weight',600).text(nm[0]);
  svg.append('text').attr('x',475).attr('y',140).attr('text-anchor','middle').text(nm[1]);
  const stamp=svg.append('text').attr('x',475).attr('y',178).attr('text-anchor','middle').style('font-size','17px').style('font-weight',800).attr('opacity',0);
  const finBox=svg.append('rect').attr('x',620).attr('y',110).attr('width',120).attr('height',70).attr('rx',14).style('fill',C.panel).style('stroke',C.line);
  svg.append('text').attr('x',680).attr('y',132).attr('text-anchor','middle').style('fill',C.fg).style('font-weight',600).text('Final answer');
  const finMark=svg.append('text').attr('x',680).attr('y',168).attr('text-anchor','middle').style('font-size','26px').style('font-weight',800).attr('opacity',0);
  svg.append('rect').attr('x',590).attr('y',232).attr('width',170).attr('height',56).attr('rx',12).style('fill','none').style('stroke',C.cy).attr('stroke-dasharray','5 4');
  svg.append('text').attr('x',675).attr('y',255).attr('text-anchor','middle').style('fill',C.cy).text('Hidden checker');
  const scan=svg.append('rect').attr('x',592).attr('y',266).attr('width',0).attr('height',3).attr('rx',2).style('fill',C.cy).attr('filter','url(#ar-glow)');
  const chk=svg.append('text').attr('x',675).attr('y',281).attr('text-anchor','middle').style('font-size','11px');
  const ring=svg.append('circle').attr('cx',260).attr('cy',145).attr('r',0).style('fill','none').style('stroke',C.vi).attr('stroke-width',3).attr('opacity',0);
  const flo=svg.append('text').attr('x',680).attr('y',100).attr('text-anchor','middle').style('font-size','15px').style('font-weight',700).attr('opacity',0);
  const pk=svg.append('g').attr('opacity',0),pkc=pk.append('circle').attr('r',17).attr('filter','url(#ar-glow)'),pkt=pk.append('text').attr('text-anchor','middle').attr('y',4).style('fill',C.bg).style('font-weight',800).style('font-size','10px');
  const done=svg.append('text').attr('x',380).attr('y',60).attr('text-anchor','middle').style('font-size','22px').style('font-weight',700).style('fill',C.fg).attr('opacity',0);
  if(opts.measured===false)L.append('p').attr('class','ar-banner').text('Sample outcomes, not study results.');
  const cn=R.append('div').attr('class','ar-cnts'),cv={};
  [['n','Runs'],['acc','Final accuracy'],['dacc','Draft accuracy'],['caught','Errors caught'],['fixed','Fixed'],['broken','Broken'],['fa','False alarms'],['hr','Human review'],['cost','Cost']].forEach(([k,l])=>{const d=cn.append('div').attr('class','ar-cnt');d.append('span').text(l);cv[k]=d.append('b').text('0')});
  const csvg=R.append('svg').attr('class','ar').attr('width','100%').attr('viewBox','0 0 380 150').attr('role','img').attr('aria-label','Running accuracy, draft versus final');
  const log=R.append('div').attr('class','ar-log');

  function paint(){const t=st.tot;cv.n.text(t.n+' / '+N);cv.acc.text(t.n?Math.round(t.fin/t.n*100)+'%':'–').style('color',C.cy);cv.dacc.text(t.n?Math.round(t.dra/t.n*100)+'%':'–');cv.caught.text(t.caught);cv.fixed.text(t.fixed).style('color',C.cy);cv.broken.text(t.broken).style('color',t.broken?C.bad:null);cv.fa.text(t.fa);cv.hr.text(t.hr);cv.cost.text('$'+t.cost.toFixed(2));
    const w=380,h=150,m={l:30,r:8,t:14,b:22},x=d3.scaleLinear([0,N],[m.l,w-m.r]),y=d3.scaleLinear([40,100],[h-m.b,m.t]);csvg.selectAll('*').remove();
    [50,75,100].forEach(v=>{csvg.append('line').attr('x1',m.l).attr('x2',w-m.r).attr('y1',y(v)).attr('y2',y(v)).style('stroke',C.line);csvg.append('text').attr('x',m.l-5).attr('y',y(v)+4).attr('text-anchor','end').text(v)});
    csvg.append('text').attr('x',m.l).attr('y',h-6).text('runs');
    if(st.hist.length>1){const ln=k=>d3.line().x(d=>x(d.n)).y(d=>y(Math.max(40,d[k])))(st.hist);csvg.append('path').attr('d',ln('dra')).style('fill','none').style('stroke',C.vs).attr('stroke-width',2).attr('stroke-dasharray','4 4');csvg.append('path').attr('d',ln('fin')).style('fill','none').style('stroke',C.cy).attr('stroke-width',2.5);const l=st.hist[st.hist.length-1];csvg.append('circle').attr('cx',x(l.n)).attr('cy',y(Math.max(40,l.fin))).attr('r',4).style('fill',C.cy)}
    csvg.append('text').attr('x',w-m.r).attr('y',11).attr('text-anchor','end').style('fill',C.vs).text('draft');csvg.append('text').attr('x',w-m.r-38).attr('y',11).attr('text-anchor','end').style('fill',C.cy).text('final')}
  function reset(){st.idx=0;st.pt=0;st.wait=0;st.hist=[];st.tot={n:0,fin:0,dra:0,caught:0,fixed:0,broken:0,fa:0,hr:0,cost:0};done.attr('opacity',0);log.selectAll('*').remove();pk.attr('opacity',0);finMark.attr('opacity',0);stamp.attr('opacity',0);paint()}
  function commit(r){const t=st.tot;t.n++;if(r.finalCorrect)t.fin++;if(r.draftCorrect)t.dra++;const rej=r.verdict==='reject';if(rej&&!r.draftCorrect)t.caught++;if(rej&&r.draftCorrect)t.fa++;if(!r.draftCorrect&&r.finalCorrect)t.fixed++;if(r.draftCorrect&&!r.finalCorrect)t.broken++;if(r.humanReview)t.hr++;t.cost+=r.cost;st.hist.push({n:t.n,fin:t.fin/t.n*100,dra:t.dra/t.n*100});
    let msg,col=r.finalCorrect?C.cy:C.bad;
    if(r.humanReview){msg=r.route==='revision_not_accepted'?'human review: revision not accepted':r.route==='unreadable'?'human review: verdict unreadable':r.route==='abstain'?'human review: verifier abstained':'sent to human review';col=C.amber}else if(r.verdict==='none')msg=r.finalCorrect?'correct':'wrong';else if(r.verdict==='accept')msg=r.finalCorrect?'accepted, correct':'accepted, missed the error';else msg=r.draftCorrect?'rejected a right answer':(r.finalCorrect?'rejected, fixed':'rejected, still wrong');
    const row=log.insert('div',':first-child');row.append('b').style('color',col).text(r.task);row.append('span').text(msg);const kids=log.selectAll('div').nodes();if(kids.length>7)kids[kids.length-1].remove();paint()}
  function frame(r,p){
    const rej=r.verdict==='reject',dcol=r.draftCorrect?C.cy:C.bad,fcol=r.finalCorrect?C.cy:C.bad;let x=80,col=C.vs;
    if(p<.18)x=lerp(80,260,seg(p,0,.18));else if(p<.36)x=260;else if(p<.52)x=lerp(260,475,seg(p,.36,.52));else if(p<.68)x=475;else if(rej){if(p<.78)x=lerp(475,260,seg(p,.68,.78));else if(p<.86)x=260;else x=lerp(260,680,seg(p,.86,.96))}else x=lerp(475,680,seg(p,.68,.9));
    if(p>=.36)col=dcol;if(rej&&p>=.82)col=fcol;
    const y=(rej&&p>.68&&p<.86)?145+26*Math.sin(seg(p,.68,.78)*Math.PI):145;
    pk.attr('opacity',p<.96&&!(p>=.9&&!rej)?1:Math.max(0,1-seg(p,.9,1))).attr('transform','translate('+x+','+y+')');pkc.style('fill',col);pkt.text((r.task||'').slice(-8));
    const pulse=(p>=.18&&p<.36)||(rej&&p>=.78&&p<.86);ring.attr('opacity',pulse?.9:0).attr('r',pulse?12+((p*40)%1)*34:0);
    const showS=r.verdict!=='none'&&p>=.56&&p<.9,odd=r.verdict==='abstain'||r.verdict==='unreadable',sc=rej?C.bad:(odd?C.amber:C.cy);
    stamp.attr('opacity',showS?1:0).text(rej?'REJECT':r.verdict==='abstain'?'ABSTAIN':r.verdict==='unreadable'?'UNREADABLE':'ACCEPT').style('fill',sc).style('font-size',r.verdict==='unreadable'?'14px':'17px');verBox.style('stroke',showS?sc:C.line);
    const fin=p>=.93;finMark.attr('opacity',fin?1:0).text(r.humanReview?'?':(r.finalCorrect?'✓':'✗')).style('fill',r.humanReview?C.amber:fcol);finBox.style('stroke',fin?(r.humanReview?C.amber:fcol):C.line);
    scan.attr('width',fin?lerp(0,166,seg(p,.93,1)):0);chk.text(fin?(r.finalCorrect?'all checks pass':'failed a hidden check'):'').style('fill',r.finalCorrect?C.cy:C.bad);
    const ev=!r.draftCorrect&&r.finalCorrect?['+1 fixed',C.cy]:(r.draftCorrect&&!r.finalCorrect?['broken',C.bad]:null);flo.attr('opacity',fin&&ev?1:0).attr('y',100-(fin?seg(p,.93,1)*14:0)).text(ev?ev[0]:'').style('fill',ev?ev[1]:C.fg)}
  function tick(){const now=performance.now(),dt=Math.min(80,now-(st.last||now));st.last=now;
    root.selectAll('.ar-spd').classed('on',function(){return +this.getAttribute('data-v')===st.speed}).attr('aria-pressed',function(){return String(+this.getAttribute('data-v')===st.speed)});bPlay.text(st.paused?'Play':'Pause').attr('aria-pressed',String(st.paused)).attr('aria-label',st.paused?'Play the replay':'Pause the replay');
    if(st.paused)return;
    if(st.idx>=N){done.attr('opacity',1).text(N?'Replay complete · '+Math.round(st.tot.fin/N*100)+'% final answers right':'No runs to replay');pk.attr('opacity',0);st.wait+=dt;if(st.wait>4500)reset();return}
    st.pt+=dt*st.speed;const p=st.pt/2600,r=runs[st.idx];if(p>=1){frame(r,1);commit(r);st.idx++;st.pt=0}else frame(r,p)}
  bPlay.on('click',()=>{st.paused=!st.paused});
  reset();
  let timer=null;
  if(reduce()){runs.forEach(commit);st.idx=N;done.attr('opacity',1).text(N?'Replay complete · '+Math.round(st.tot.fin/N*100)+'% final answers right':'No runs to replay');st.paused=true;bPlay.text('Play').attr('aria-pressed','true')}
  timer=d3.timer(tick);
  const api={destroy(){timer.stop();root.selectAll('*').remove()},restart(){reset();st.paused=false},setSpeed(v){st.speed=v},toggle(){st.paused=!st.paused}};
  return api}
window.AdvisoryReplay={mount,normalize};
})();
