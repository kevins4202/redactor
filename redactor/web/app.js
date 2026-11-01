'use strict';
const $ = id => document.getElementById(id);
const canvas = $('canvas'), context = canvas.getContext('2d');
const state = {video:null, frame:0, loadedFrame:-1, image:null, prompts:[], drawing:null, overlay:null, request:0, job:null, busy:false, ready:false, flags:[], flagIndex:0};
async function api(url, options={}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {const body = await response.json(); detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);} catch (_) {}
    const error=new Error(detail);error.status=response.status;throw error;
  }
  return response.json();
}
function post(url, body) {return api(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});}
function message(text='') {$('message').textContent=text; $('message').hidden=!text;}
function controls() {
  $('process').disabled = !state.ready || state.busy || !state.prompts.length;
  $('preview').disabled = !state.ready || state.busy || state.loadedFrame!==state.frame || !state.prompts.some(p=>p.frame===state.frame);
  $('clear').disabled = state.busy;
  $('upload').disabled = state.busy;
  $('seek').disabled = state.busy;
  $('previous').disabled = state.busy;
  $('next').disabled = state.busy;
}
function drawBox(box) {
  context.strokeStyle='#69e17d'; context.lineWidth=3*canvas.width/Math.max(1,canvas.clientWidth);
  context.strokeRect(box[0],box[1],box[2]-box[0],box[3]-box[1]);
}
function render() {
  context.clearRect(0,0,canvas.width,canvas.height);
  if (state.image) context.drawImage(state.image,0,0);
  if (state.overlay) {context.save();context.globalAlpha=.55;context.drawImage(state.overlay,0,0);context.restore();}
  else state.prompts.filter(p=>p.frame===state.frame).forEach(p=>drawBox(p.box));
  if (state.drawing) drawBox(state.drawing.box);
}
async function seek(index) {
  if (!state.video || state.busy) return;
  if(state.overlay) message();
  state.frame=Math.max(0,Math.min(state.video.frames-1,Number(index))); state.overlay=null; state.drawing=null;
  $('seek').value=state.frame; $('frame-label').textContent=`Frame ${state.frame+1} / ${state.video.frames}`; controls();
  const request=++state.request;
  try {
    const image=new Image();
    await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=()=>reject(new Error('Could not load frame'));image.src=`/api/videos/${state.video.id}/frames/${state.frame}`;});
    if (request!==state.request) return;
    state.image=image;state.loadedFrame=state.frame;render();controls();
  } catch(error) {message(error.message);}
}
function selections() {
  $('selections').replaceChildren();
  for (const prompt of state.prompts.slice().sort((a,b)=>a.frame-b.frame)) {
    const button=document.createElement('button');button.textContent=`Selection at frame ${prompt.frame+1} ×`;
    button.title='Remove this selection';button.disabled=state.busy;
    button.onclick=()=>{state.prompts=state.prompts.filter(p=>p!==prompt);state.overlay=null;invalidateResult();selections();render();controls();};
    $('selections').append(button);
  }
}
function invalidateResult() {$('result').hidden=true;state.job=null;state.flags=[];}
async function load(video) {
  state.video=video;state.prompts=[];state.image=null;state.loadedFrame=-1;invalidateResult();
  canvas.width=video.width;canvas.height=video.height;$('seek').max=video.frames-1;$('video-name').textContent=video.name;
  selections();await seek(0);
}
function point(event) {
  const r=canvas.getBoundingClientRect();
  return [Math.round(Math.max(0,Math.min(canvas.width-1,(event.clientX-r.left)*canvas.width/r.width))),Math.round(Math.max(0,Math.min(canvas.height-1,(event.clientY-r.top)*canvas.height/r.height)))];
}
canvas.onpointerdown=event=>{
  if (!state.video || state.busy || state.loadedFrame!==state.frame || event.button!==0)return;
  canvas.setPointerCapture(event.pointerId);const start=point(event);state.overlay=null;state.drawing={start,box:[...start,...start]};render();
};
canvas.onpointermove=event=>{if(!state.drawing)return;const [x,y]=point(event),[a,b]=state.drawing.start;state.drawing.box=[Math.min(a,x),Math.min(b,y),Math.max(a,x),Math.max(b,y)];render();};
canvas.onpointerup=()=>{
  if(!state.drawing)return;const box=state.drawing.box;state.drawing=null;
  if(box[2]-box[0]>3 && box[3]-box[1]>3){state.prompts=state.prompts.filter(p=>p.frame!==state.frame);state.prompts.push({id:'target',frame:state.frame,box});invalidateResult();selections();controls();}
  render();
};
canvas.onpointercancel=()=>{state.drawing=null;render();};
$('seek').oninput=event=>seek(event.target.value);$('previous').onclick=()=>seek(state.frame-1);$('next').onclick=()=>seek(state.frame+1);
document.onkeydown=event=>{if(event.key==='Escape'){state.drawing=null;render();}if(event.target.tagName==='INPUT')return;if(['ArrowLeft','ArrowRight'].includes(event.key)){event.preventDefault();seek(state.frame+(event.key==='ArrowRight'?1:-1));}};
$('clear').onclick=()=>{state.prompts=[];state.overlay=null;invalidateResult();selections();render();controls();};
$('upload').onchange=async event=>{const file=event.target.files[0];if(!file)return;try{const form=new FormData();form.append('file',file);await load(await api('/api/videos',{method:'POST',body:form}));message();}catch(error){message(error.message);}event.target.value='';};
$('preview').onclick=async()=>{
  const frame=state.frame,request=state.request;state.busy=true;controls();selections();message('Checking selection…');
  try{
    const response=await post('/api/previews',{video_id:state.video.id,frame,prompts:state.prompts.filter(p=>p.frame===frame)});
    const image=new Image();await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=()=>reject(new Error('Could not display mask'));image.src=response.mask;});
    if(request===state.request){state.overlay=image;render();}message('Green shows the initial mask and its coverage margin. Adjust the box if it selects the wrong object.');
  }catch(error){message(error.message);}finally{state.busy=false;controls();selections();}
};
$('process').onclick=async()=>{
  state.busy=true;controls();selections();invalidateResult();message();
  try{const job=await post('/api/exports',{video_id:state.video.id,prompts:state.prompts});state.job=job.id;$('job').hidden=false;$('cancel').hidden=false;$('progress').value=0;poll();}
  catch(error){state.busy=false;controls();selections();message(error.message);}
};
async function poll(){
  try{
    const job=await api(`/api/exports/${state.job}`);
    if(['queued','running'].includes(job.status)){
      const names={prepare:'Preparing frames',track:'Following the selected object',render:'Rendering the preview'};
      $('job-title').textContent=names[job.stage]||'Preparing';$('job-detail').textContent=`${job.done} / ${job.total} frames`;$('progress').value=100*job.done/Math.max(1,job.total);setTimeout(poll,1000);return;
    }
    state.busy=false;controls();selections();$('cancel').hidden=true;
    if(!['complete','review'].includes(job.status)){$('job-title').textContent=job.status==='cancelled'?'Cancelled':'Processing failed';$('job-detail').textContent=job.error||'Try again';return;}
    $('job').hidden=true;$('result').hidden=false;state.flags=job.review_frames;state.flagIndex=0;state.reviewReasons=job.review_reasons||{};
    const base=`/api/exports/${state.job}/file/`;
    $('result-video').src=base+'preview';$('report-link').href=base+'report';$('prompts-link').href=base+'prompts';$('download').href=base+'video';
    $('download').hidden=job.status!=='complete';$('approve').hidden=job.status!=='review';$('review-next').hidden=!state.flags.length;
    $('review-summary').textContent=job.status==='review'?`${state.flags.length} frames need review. Check them, add a corrective box and process again, or approve if the visible target is properly covered.`:'Processing finished. Watch the full result to check coverage and identity.';
  }catch(error){
    if(error.status===404){state.busy=false;controls();selections();$('cancel').hidden=true;message('The processing session ended. Open the video again after restarting the app.');}
    else{message(`Could not check progress: ${error.message}. Retrying…`);setTimeout(poll,2000);}
  }
}
$('cancel').onclick=async()=>{try{await post(`/api/exports/${state.job}/cancel`,{});$('job-detail').textContent='Stopping after the current model step…';}catch(error){message(error.message);}};
$('review-next').onclick=async()=>{const frame=state.flags[state.flagIndex++%state.flags.length];await seek(frame);message((state.reviewReasons[String(frame)]||[]).join('. '));$('result-video').currentTime=frame/state.video.fps;canvas.scrollIntoView({block:'center'});};
$('approve').onclick=async()=>{try{await post(`/api/exports/${state.job}/approve`,{});$('approve').hidden=true;$('download').hidden=false;$('review-summary').textContent='Flagged frames marked as reviewed. The video is ready to download.';}catch(error){message(error.message);}};
(async()=>{try{const [videos,capabilities]=await Promise.all([api('/api/videos'),api('/api/capabilities')]);state.ready=capabilities.ready;if(!state.ready)message('The tracking model is not installed. Run python scripts/setup_models.py, then refresh.');if(videos.length)await load(videos[0]);controls();}catch(error){message(error.message);}})();
