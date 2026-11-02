'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const KINDS = ['rear_end', 'cut_in', 'side', 'head_on'];
  const NAME = {rear_end:'追尾', cut_in:'切入', side:'侧碰', head_on:'迎头'};
  const COLOR = {rear_end:'#f4bb5e', cut_in:'#66b9ff', side:'#b995ff', head_on:'#ff7c87'};
  const RELATION = {same_lane:'同一车道', adjacent_same_dir:'同向邻接', merging:'汇入', crossing:'交叉', opposing:'对向'};
  const catalog = window.CATALOG;
  const cache = new Map();
  const pending = new Map();
  const state = {dataset:'waymo', entry:null, payload:null, frame:0, selected:-1, enabled:new Set(KINDS), playing:false, radius:54, pan:[0,0], request:0, ready:false};
  let renderedPoints = [], lastDetection = -1, tickAnchor = 0, tickStart = 0;
  window.registerScene = (key, data) => {
    cache.set(key, data);
    if (pending.has(key)) { pending.get(key).resolve(data); pending.delete(key); }
  };
  window.registerCompressedScene = async (key, encoded) => {
    try {
      const bytes = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
      const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
      const payload = JSON.parse(await new Response(stream).text());
      window.registerScene(key, payload);
    } catch (error) {
      if (pending.has(key)) { pending.get(key).reject(error); pending.delete(key); }
    }
  };
  function load(entry) {
    if (cache.has(entry.key)) return Promise.resolve(cache.get(entry.key));
    if (pending.has(entry.key)) return pending.get(entry.key).promise;
    let resolve, reject;
    const promise = new Promise((yes,no) => {resolve=yes;reject=no;});
    pending.set(entry.key, {promise,resolve,reject});
    const script = document.createElement('script');
    script.src = entry.file;
    script.onload = () => {script.remove();};
    script.onerror = () => {pending.delete(entry.key);script.remove();reject(new Error('场景文件读取失败，请检查离线包是否完整'));};
    document.body.append(script);
    return promise;
  }
  function visibleScenes() {
    const text = $('search').value.trim().toLowerCase(), filter = $('sceneFilter').value;
    return catalog.scenes.filter(x => x.dataset === state.dataset && x.id.toLowerCase().includes(text) && (!filter || (filter === 'unmatched' ? x.unmatched > 0 : x.types.includes(filter))));
  }
  function renderScenes() {
    const list = visibleScenes();
    $('sceneCount').textContent = `${list.length} / ${catalog.scenes.filter(x=>x.dataset===state.dataset).length} 场景 · 保存示例`;
    $('sceneList').replaceChildren();
    for (const entry of list) {
      const item = document.createElement('button');
      item.className = 'scene-item' + (entry.key === state.entry?.key ? ' active' : '');
      item.dataset.key = entry.key;
      item.setAttribute('aria-pressed', String(entry.key === state.entry?.key));
      const id = document.createElement('div');id.className='scene-id';
      const no = document.createElement('span');no.className='scene-no';no.textContent=String(entry.number).padStart(3,'0');
      id.append(no, document.createTextNode(entry.id));
      const mini = document.createElement('div');mini.className='mini-types';
      for (const kind of KINDS) {const dot=document.createElement('span');dot.style.background=entry.types.includes(kind)?COLOR[kind]:'#253747';dot.title=NAME[kind];mini.append(dot);}
      item.append(id,mini);item.onclick=()=>selectScene(entry.key);$('sceneList').append(item);
    }
    if (!list.length) {const item=document.createElement('p');item.className='empty';item.textContent='没有匹配的场景';$('sceneList').append(item);}
  }
  function syncDataset() {
    document.querySelectorAll('[data-dataset]').forEach(b=>{b.classList.toggle('active',b.dataset.dataset===state.dataset);b.setAttribute('aria-pressed',String(b.dataset.dataset===state.dataset));});
    renderScenes();
  }
  async function setDataset(dataset) {
    if (!['waymo','nuplan'].includes(dataset)) return;
    state.dataset=dataset;$('search').value='';$('sceneFilter').value='';syncDataset();
    return selectScene(catalog.scenes.find(x=>x.dataset===dataset).key);
  }
  async function selectScene(key, initialFrame=0) {
    const entry=catalog.scenes.find(x=>x.key===key);if(!entry) return;
    const request=++state.request;pause();state.ready=false;state.payload=null;state.entry=entry;state.dataset=entry.dataset;state.selected=-1;state.pan=[0,0];lastDetection=-1;
    $('sceneTitle').textContent=entry.id;$('datasetLabel').textContent=`${entry.dataset==='waymo'?'WAYMO':'NUPLAN'} / ${String(entry.number).padStart(3,'0')}`;
    $('loadError').hidden=true;$('mapStatus').textContent='载入原始场景…';$('opList').replaceChildren();$('roadTags').replaceChildren();
    syncDataset();updateNavigation();draw();
    try {
      const payload=await load(entry);if(request!==state.request)return;
      state.payload=payload;state.ready=true;buildTimeline();setFrame(initialFrame);
    } catch(error) {
      if(request!==state.request)return;
      $('loadError').textContent=error.message;$('loadError').hidden=false;$('mapStatus').textContent='场景载入失败';
    }
  }
  function updateNavigation() {
    const list=visibleScenes(),index=list.findIndex(x=>x.key===state.entry?.key);
    $('prevScene').disabled=index<=0;$('nextScene').disabled=index<0||index>=list.length-1;
  }
  function changeScene(delta) {const list=visibleScenes(),i=list.findIndex(x=>x.key===state.entry?.key);if(list[i+delta])selectScene(list[i+delta].key);}
  function detectionIndex() {return state.payload ? Math.max(0,state.payload.scan.frames.findLastIndex(f=>f.source_frame<=state.frame)) : 0;}
  function currentDetection() {return state.payload?.scan.frames[detectionIndex()];}
  function visibleOps() {return (currentDetection()?.opportunities||[]).map((op,i)=>({op,i})).filter(({op})=>op.collision_types.some(k=>state.enabled.has(k)));}
  function buildTimeline() {
    $('timelineGrid').replaceChildren();
    for(const kind of KINDS){
      const row=document.createElement('div');row.className='timeline-row';row.style.setProperty('--c',COLOR[kind]);
      const name=document.createElement('span');name.className='timeline-name';name.textContent=NAME[kind];row.append(name);
      state.payload.scan.frames.forEach((frame,i)=>{const button=document.createElement('button');button.className='timeline-cell'+(frame.potential_collision_types.includes(kind)?' on':'');button.dataset.tick=i;button.title=`${frame.time_s.toFixed(1)} s · ${NAME[kind]} · ${frame.potential_collision_types.includes(kind)?'有机会':'未检出此类'}`;button.setAttribute('aria-label',button.title);button.onclick=()=>{pause();setFrame(frame.source_frame);};row.append(button);});
      $('timelineGrid').append(row);
    }
  }
  function setFrame(frame) {
    if(!state.payload)return;
    state.frame=Math.min(state.payload.scan.source_times_s.length-1,Math.max(0,Math.round(Number(frame)||0)));
    $('time').value=state.frame;$('timeLabel').textContent=(state.frame/10).toFixed(1);
    $('frameLabel').textContent=`SOURCE ${String(state.frame).padStart(2,'0')} / 90`;
    const index=detectionIndex();if(index!==lastDetection){state.selected=-1;lastDetection=index;renderDetails();}
    $('mapStatus').textContent=`原始帧 ${state.frame} · 检测 ${currentDetection().time_s.toFixed(1)} s`;
    document.querySelectorAll('.timeline-cell').forEach(b=>b.classList.toggle('current',Number(b.dataset.tick)===index));
    updateURL();draw();
  }
  function updateURL(){try{const url=new URL(location.href);url.searchParams.set('dataset',state.dataset);url.searchParams.set('scene',state.entry.id);url.searchParams.set('time',(state.frame/10).toFixed(1));history.replaceState(null,'',url);}catch{ /* file URLs may not allow history updates */ }}
  function renderDetails() {
    const f=currentDetection();if(!f)return;
    $('detectionTime').textContent=f.time_s.toFixed(1)+' s';
    const status=$('frameStatus');status.classList.toggle('warn',f.status!=='ok');
    status.textContent={ok:'已匹配 Ego 车道',no_ego_lane:'未匹配 Ego 车道 · 不能据此判定无风险',invalid_ego:'Ego 状态无效 · 未完成该时刻检测'}[f.status]||f.status;
    $('roadTags').replaceChildren();
    for(const id of f.road_type_ids){const tag=document.createElement('span');tag.className='road-tag';tag.textContent=state.payload.scan.road_catalog[id].name;$('roadTags').append(tag);}
    if(!f.road_type_ids.length)$('roadTags').textContent='未检测到有效道路标签';
    const ops=visibleOps();$('opCount').textContent=`${ops.length} / ${f.opportunities.length}`;$('opList').replaceChildren();
    for(const {op,i} of ops){
      const kind=op.collision_types.find(k=>state.enabled.has(k));const item=document.createElement('button');item.className='op-item'+(state.selected===i?' selected':'');item.style.setProperty('--c',COLOR[kind]);item.dataset.op=i;
      const heading=document.createElement('strong');heading.textContent=`${String(i+1).padStart(2,'0')}  ${op.collision_types.filter(k=>state.enabled.has(k)).map(k=>NAME[k]).join(' / ')}`;
      const detail=document.createElement('small');detail.textContent=`${RELATION[op.relation_detail]||RELATION[op.relation]||op.relation} · ${op.angle_deg.toFixed(1)}°`;
      const lanes=document.createElement('small');lanes.textContent=`车道 ${op.ego_lane_id} → ${op.other_lane_id}`;
      item.append(heading,detail,lanes);item.onclick=()=>{state.selected=state.selected===i?-1:i;renderDetails();draw();};$('opList').append(item);
    }
    if(!ops.length){const msg=document.createElement('p');msg.className='empty';msg.textContent=f.status!=='ok'?'该时刻未完成有效匹配':f.opportunities.length?'当前类型筛选下无机会':'该时刻未检出碰撞机会';$('opList').append(msg);}
  }
  const canvas=$('map'),ctx=canvas.getContext('2d');
  function scalar(value,index,fallback=0){const v=Array.isArray(value)?value[index]:value;const n=Number(Array.isArray(v)?v[0]:v);return Number.isFinite(n)?n:fallback;}
  function draw(){
    const bounds=canvas.getBoundingClientRect(),w=bounds.width,h=bounds.height;if(!w||!h)return;
    const dpr=Math.min(window.devicePixelRatio||1,2);if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
    ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle='#0d1722';ctx.fillRect(0,0,w,h);renderedPoints=[];
    if(!state.payload)return;
    const scan=state.payload.scan,f=currentDetection(),ego=scan.ego_track,center=ego.position[state.frame];
    if(!center||!center.slice(0,2).every(Number.isFinite))return;
    const camera=[center[0]+state.pan[0],center[1]+state.pan[1]],scale=Math.min(w,h)/(2*state.radius);
    const xy=p=>[(p[0]-camera[0])*scale+w/2,h/2-(p[1]-camera[1])*scale];
    function path(points,color,width=1,dash=[]){if(points.length<2)return;ctx.beginPath();for(let i=0;i<points.length;i++){const p=xy(points[i]);i?ctx.lineTo(...p):ctx.moveTo(...p);}ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.stroke();ctx.setLineDash([]);}
    const step=10;ctx.strokeStyle='#192b3b';ctx.lineWidth=.6;
    for(let x=Math.floor((camera[0]-w/scale/2)/step)*step;x<camera[0]+w/scale/2;x+=step){const p=xy([x,camera[1]]);ctx.beginPath();ctx.moveTo(p[0],0);ctx.lineTo(p[0],h);ctx.stroke();}
    for(let y=Math.floor((camera[1]-h/scale/2)/step)*step;y<camera[1]+h/scale/2;y+=step){const p=xy([camera[0],y]);ctx.beginPath();ctx.moveTo(0,p[1]);ctx.lineTo(w,p[1]);ctx.stroke();}
    const cp=xy(center);ctx.beginPath();ctx.arc(...cp,12*scale,0,Math.PI*2);ctx.strokeStyle='#66dbb52b';ctx.setLineDash([3,5]);ctx.stroke();ctx.setLineDash([]);
    for(const id of f.nearby_lane_ids){const lane=scan.lanes[id];if(!lane)continue;path(lane.polyline,'#3b5166',1.3);
      let travelled=0;for(let i=1;i<lane.polyline.length;i++){const a=xy(lane.polyline[i-1]),b=xy(lane.polyline[i]);const length=Math.hypot(b[0]-a[0],b[1]-a[1]);travelled+=length;if(travelled<35||length<.3)continue;travelled=0;if(b[0]<-10||b[0]>w+10||b[1]<-10||b[1]>h+10)continue;const angle=Math.atan2(b[1]-a[1],b[0]-a[0]);ctx.save();ctx.translate(...b);ctx.rotate(angle);ctx.beginPath();ctx.moveTo(-5,-2.5);ctx.lineTo(0,0);ctx.lineTo(-5,2.5);ctx.strokeStyle='#668098';ctx.lineWidth=1;ctx.stroke();ctx.restore();}}
    for(const id of f.ego_lane_ids){if(scan.lanes[id])path(scan.lanes[id].polyline,'#68e5bb99',2.5);}
    if($('showTrail').checked){path(ego.position.filter((p,i)=>scalar(ego.valid,i)>0&&p.slice(0,2).every(Number.isFinite)),'#68e5bb38',1.5,[4,5]);path(ego.position.slice(0,state.frame+1).filter((p,i)=>scalar(ego.valid,i)>0&&p.slice(0,2).every(Number.isFinite)),'#68e5bb99',2);}
    const ops=visibleOps();
    for(const {op,i} of ops){const kind=op.collision_types.find(k=>state.enabled.has(k));if(i===state.selected){for(const id of [op.ego_lane_id,op.other_lane_id])if(scan.lanes[id])path(scan.lanes[id].polyline,COLOR[kind],3);}}
    function actor(track,isEgo=false,type='VEHICLE'){
      const i=state.frame;if(!scalar(track.valid,i))return;const position=track.position[i];if(!position||!position.slice(0,2).every(Number.isFinite))return;const p=xy(position);if(p[0]<-30||p[0]>w+30||p[1]<-30||p[1]>h+30)return;
      ctx.save();ctx.translate(...p);ctx.rotate(-scalar(track.heading,i));const length=Math.max(2,scalar(track.length,i,4.5)*scale),width=Math.max(2,scalar(track.width,i,1.8)*scale);
      ctx.fillStyle=isEgo?'#68e5bb':type==='PEDESTRIAN'?'#c1b4a4':'#536b80';ctx.strokeStyle=isEgo?'#b9ffe5':'#8ca1b3';ctx.lineWidth=isEgo?1.5:.7;
      ctx.beginPath();ctx.rect(-length/2,-width/2,length,width);ctx.fill();ctx.stroke();ctx.beginPath();ctx.moveTo(length/2,0);ctx.lineTo(length/2-3,-2);ctx.lineTo(length/2-3,2);ctx.closePath();ctx.fillStyle=isEgo?'#103d31':'#b0becc';ctx.fill();ctx.restore();
      if(isEgo){ctx.fillStyle='#a6edcf';ctx.font='10px system-ui';ctx.fillText('EGO',p[0]+8,p[1]-10);}
    }
    if($('showActors').checked)for(const a of state.payload.actors)actor(a.state,false,a.type);
    actor(ego,true);
    // Draw opportunities as rings rather than physical vehicles; coordinates come directly from the scan.
    const ordered=[...ops.filter(x=>x.i!==state.selected),...ops.filter(x=>x.i===state.selected)];
    for(const {op,i} of ordered){const p=xy(op.point),kind=op.collision_types.find(k=>state.enabled.has(k));if(p[0]<-20||p[0]>w+20||p[1]<-20||p[1]>h+20)continue;const r=i===state.selected?12:8;
      ctx.beginPath();ctx.arc(...p,r,0,Math.PI*2);ctx.fillStyle=COLOR[kind]+'24';ctx.fill();ctx.strokeStyle=COLOR[kind];ctx.lineWidth=i===state.selected?2.5:1.7;ctx.stroke();ctx.beginPath();ctx.moveTo(p[0]-3,p[1]);ctx.lineTo(p[0]+3,p[1]);ctx.moveTo(p[0],p[1]-3);ctx.lineTo(p[0],p[1]+3);ctx.stroke();
      ctx.font='bold 10px ui-monospace,monospace';ctx.fillStyle='#0d1722';ctx.fillRect(p[0]+r-1,p[1]-r-7,19,13);ctx.fillStyle=COLOR[kind];ctx.fillText(String(i+1).padStart(2,'0'),p[0]+r+1,p[1]-r+3);renderedPoints.push({i,p});}
    $('scale').textContent=`10 m ${'━'.repeat(Math.max(2,Math.round(10*scale/8)))}`;
    canvas.dataset.sourceFrame=state.frame;canvas.dataset.detectionFrame=f.source_frame;canvas.dataset.visibleOpCount=ops.length;
  }
  let drag=null;
  canvas.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,pan:[...state.pan],moved:false};canvas.setPointerCapture(e.pointerId);});
  canvas.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.hypot(dx,dy)>4)drag.moved=true;const s=Math.min(canvas.clientWidth,canvas.clientHeight)/(2*state.radius);state.pan=[drag.pan[0]-dx/s,drag.pan[1]+dy/s];draw();});
  canvas.addEventListener('pointerup',e=>{if(!drag)return;if(!drag.moved){const rect=canvas.getBoundingClientRect(),p=[e.clientX-rect.left,e.clientY-rect.top];const point=renderedPoints.filter(x=>Math.hypot(x.p[0]-p[0],x.p[1]-p[1])<17).sort((a,b)=>Math.hypot(a.p[0]-p[0],a.p[1]-p[1])-Math.hypot(b.p[0]-p[0],b.p[1]-p[1]))[0];if(point){state.selected=point.i;renderDetails();draw();$('opList').querySelector('.selected')?.scrollIntoView({block:'nearest'});}}drag=null;});
  canvas.addEventListener('pointercancel',()=>drag=null);
  canvas.addEventListener('wheel',e=>{e.preventDefault();state.radius=Math.min(150,Math.max(15,state.radius*Math.exp(e.deltaY*.001)));draw();},{passive:false});
  function pause(){state.playing=false;$('play').textContent='▶ 播放';$('play').setAttribute('aria-label','播放');}
  function animate(now){if(!state.playing)return;const elapsed=(now-tickStart)*Number($('speed').value)/100;const next=Math.min(90,tickAnchor+Math.floor(elapsed));setFrame(next);if(next>=90){pause();return;}requestAnimationFrame(animate);}
  $('play').onclick=()=>{if(state.playing){pause();return;}if(!state.ready)return;if(state.frame>=90)setFrame(0);state.playing=true;tickAnchor=state.frame;tickStart=performance.now();$('play').textContent='Ⅱ 暂停';$('play').setAttribute('aria-label','暂停');requestAnimationFrame(animate);};
  $('speed').onchange=()=>{if(state.playing){tickAnchor=state.frame;tickStart=performance.now();}};
  $('time').oninput=e=>{pause();setFrame(e.target.value);};
  $('prevTick').onclick=()=>{pause();setFrame(Math.max(0,(Math.ceil(state.frame/5)-1)*5));};
  $('nextTick').onclick=()=>{pause();setFrame(Math.min(90,(Math.floor(state.frame/5)+1)*5));};
  $('prevScene').onclick=()=>changeScene(-1);$('nextScene').onclick=()=>changeScene(1);
  $('resetView').onclick=()=>{state.pan=[0,0];state.radius=54;draw();};
  $('showActors').onchange=draw;$('showTrail').onchange=draw;
  $('search').oninput=()=>{renderScenes();updateNavigation();};$('sceneFilter').onchange=()=>{renderScenes();updateNavigation();};
  document.querySelectorAll('[data-dataset]').forEach(b=>b.onclick=()=>setDataset(b.dataset.dataset));
  for(const kind of KINDS){const label=document.createElement('label');label.style.setProperty('--c',COLOR[kind]);const input=document.createElement('input');input.type='checkbox';input.checked=true;input.dataset.kind=kind;input.onchange=()=>{input.checked?state.enabled.add(kind):state.enabled.delete(kind);state.selected=-1;renderDetails();draw();};const ring=document.createElement('i');ring.className='ring';label.append(input,ring,document.createTextNode(NAME[kind]));$('legend').append(label);}
  new ResizeObserver(draw).observe(canvas);
  document.addEventListener('keydown',e=>{if(['INPUT','SELECT','BUTTON'].includes(document.activeElement.tagName))return;if(e.key==='ArrowRight'){e.preventDefault();pause();setFrame(state.frame+5);}if(e.key==='ArrowLeft'){e.preventDefault();pause();setFrame(state.frame-5);}if(e.code==='Space'){e.preventDefault();$('play').click();}});
  window.gallery={state,selectScene,setDataset,setFrame,currentDetection,visibleOps,catalog,draw};
  const params=new URLSearchParams(location.search);state.dataset=params.get('dataset')==='nuplan'?'nuplan':'waymo';
  const entry=catalog.scenes.find(x=>x.dataset===state.dataset&&x.id===params.get('scene'))||catalog.scenes.find(x=>x.dataset===state.dataset);
  selectScene(entry.key,Number(params.get('time')||0)*10);
})();
