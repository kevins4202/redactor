"use strict";
const $ = (id) => document.getElementById(id);
const state = { video: null, frame: 0, image: null, prompts: [], mode: "manual", target: "target-1", targetCount: 1, drawing: null, imageRequest: 0, job: null, busy: false, reviewFrames: [], reviewTargets: {}, reviewIndex: 0, exportMode: "manual" };
const canvas = $("canvas"), context = canvas.getContext("2d");

async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = "The request could not be completed.";
    try { const body = await response.json(); detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail); } catch (_) { /* preserve readable fallback */ }
    const error = new Error(detail); error.status = response.status; throw error;
  }
  return response.json();
}
function message(text = "") { $("message").textContent = text; $("message").hidden = !text; }
function ready() { $("export").disabled = !state.video || state.busy || (state.mode === "manual" && !state.prompts.length); }
function targetLabel(id) { return id.startsWith("detected-") ? id.replace("detected-", "Detected target ") : id.replace("target-", "Target "); }
function formatTime(seconds) { return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${(seconds % 60).toFixed(2).padStart(5, "0")}`; }

function drawPolygon(points, color, label) {
  if (!points.length) return;
  const factor = canvas.width / Math.max(1, canvas.clientWidth);
  context.beginPath(); context.moveTo(...points[0]);
  points.slice(1).forEach((point) => context.lineTo(...point));
  context.closePath(); context.fillStyle = color + "28"; context.fill();
  context.strokeStyle = color; context.lineWidth = 2 * factor; context.stroke();
  if (label) {
    const x = Math.min(...points.map((point) => point[0])), y = Math.max(18 * factor, Math.min(...points.map((point) => point[1])) - 6 * factor);
    context.font = `600 ${11 * factor}px sans-serif`; const width = context.measureText(label).width;
    context.fillStyle = color; context.fillRect(x, y - 17 * factor, width + 12 * factor, 20 * factor);
    context.fillStyle = "#fff"; context.fillText(label, x + 6 * factor, y - 3 * factor);
  }
}
function geometry(prompt) { if (prompt.polygon) return prompt.polygon; const [x1, y1, x2, y2] = prompt.box; return [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]; }
function render() {
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (state.image) context.drawImage(state.image, 0, 0, canvas.width, canvas.height);
  if (state.mode === "manual") state.prompts.filter((prompt) => prompt.frame === state.frame).forEach((prompt) => drawPolygon(geometry(prompt), prompt.id === state.target ? "#359566" : "#e7a64a", targetLabel(prompt.id)));
  if (state.drawing) drawPolygon(state.drawing.points, "#359566");
}
async function seek(frame) {
  if (!state.video) return;
  frame = Math.max(0, Math.min(state.video.frames - 1, Number(frame)));
  state.frame = frame; state.drawing = null;
  $("seek").value = frame; $("timecode").textContent = formatTime(frame / state.video.fps);
  $("frame-label").textContent = `Frame ${frame + 1} of ${state.video.frames}`;
  const request = ++state.imageRequest;
  try {
    const image = new Image();
    await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = () => reject(new Error("Could not load this frame.")); image.src = `/api/videos/${encodeURIComponent(state.video.id)}/frames/${frame}`; });
    if (request !== state.imageRequest) return;
    state.image = image; render();
  } catch (error) { message(error.message); }
}
function selectionList() {
  const list = $("selections"); list.replaceChildren();
  if (!state.prompts.length) { const empty = document.createElement("p"); empty.className = "empty-hint"; empty.textContent = "Your selections will appear here."; list.append(empty); }
  state.prompts.slice().sort((a,b) => a.frame - b.frame).forEach((prompt) => {
    const row = document.createElement("div"); row.className = "selection-item";
    const jump = document.createElement("button"); jump.className = "jump"; jump.textContent = `${targetLabel(prompt.id)} · frame ${prompt.frame + 1}`;
    jump.onclick = () => { state.target = prompt.id; $("target").value = prompt.id; seek(prompt.frame); };
    const remove = document.createElement("button"); remove.className = "remove"; remove.textContent = "×"; remove.setAttribute("aria-label", `Remove ${targetLabel(prompt.id)} selection at frame ${prompt.frame + 1}`);
    remove.onclick = () => { state.prompts = state.prompts.filter((item) => item !== prompt); selectionList(); render(); };
    row.append(jump, remove); list.append(row);
  });
  ready();
}
async function loadVideo(video) {
  state.video = video; state.prompts = []; state.target = "target-1"; state.targetCount = 1; state.image = null; state.reviewFrames = [];
  $("target").replaceChildren(new Option("Target 1", "target-1"));
  $("video-name").textContent = video.name; $("video-detail").textContent = `${video.width} × ${video.height} · ${video.fps.toFixed(2).replace(/\.00$/, "")} fps · ${(video.frames / video.fps).toFixed(1)} seconds`;
  canvas.width = video.width; canvas.height = video.height; $("stage").style.aspectRatio = `${video.width} / ${video.height}`;
  $("empty-state").hidden = true; $("seek").max = video.frames - 1; $("result").hidden = true; $("job-panel").hidden = true;
  selectionList(); message(); await seek(0);
}

function point(event) {
  const rect = canvas.getBoundingClientRect();
  return [Math.max(0, Math.min(canvas.width - 1, (event.clientX - rect.left) * canvas.width / rect.width)), Math.max(0, Math.min(canvas.height - 1, (event.clientY - rect.top) * canvas.height / rect.height))].map(Math.round);
}
canvas.addEventListener("pointerdown", (event) => {
  if (!state.video || state.mode !== "manual" || state.busy || event.button !== 0) return;
  canvas.setPointerCapture(event.pointerId); canvas.focus();
  const start = point(event); state.drawing = { start, points: [start], tool: $("tool").value };
  message(); render();
});
canvas.addEventListener("pointermove", (event) => {
  if (!state.drawing) return;
  const current = point(event), drawing = state.drawing;
  if (drawing.tool === "box") { const [x1,y1] = drawing.start, [x2,y2] = current; drawing.points = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]; }
  else { const last = drawing.points[drawing.points.length - 1]; if (Math.hypot(current[0] - last[0], current[1] - last[1]) > 2) drawing.points.push(current); }
  render();
});
canvas.addEventListener("pointerup", () => {
  if (!state.drawing) return;
  const drawing = state.drawing; state.drawing = null;
  const points = drawing.points;
  if (points.length < 3 || Math.max(...points.map((p) => p[0])) - Math.min(...points.map((p) => p[0])) < 3 || Math.max(...points.map((p) => p[1])) - Math.min(...points.map((p) => p[1])) < 3) { render(); return; }
  const prompt = { id: state.target, frame: state.frame, label: targetLabel(state.target), source: "manual" };
  if (drawing.tool === "box") prompt.box = [Math.min(...points.map((p) => p[0])), Math.min(...points.map((p) => p[1])), Math.max(...points.map((p) => p[0])), Math.max(...points.map((p) => p[1]))];
  else prompt.polygon = points;
  state.prompts = state.prompts.filter((p) => !(p.id === prompt.id && (p.frame === prompt.frame || (p.source === "detector" && p.frame >= prompt.frame)))); state.prompts.push(prompt);
  selectionList(); render();
});
canvas.addEventListener("pointercancel", () => { state.drawing = null; render(); });
$("seek").addEventListener("input", (event) => seek(event.target.value));
$("previous").onclick = () => seek(state.frame - 1); $("next").onclick = () => seek(state.frame + 1);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") { state.drawing = null; render(); }
  if (["INPUT","SELECT","TEXTAREA"].includes(event.target.tagName)) return;
  if (event.key === "ArrowLeft" || event.key === "ArrowRight") { event.preventDefault(); seek(state.frame + (event.key === "ArrowLeft" ? -1 : 1)); }
});
$("new-target").onclick = () => { state.target = `target-${++state.targetCount}`; $("target").add(new Option(targetLabel(state.target), state.target)); $("target").value = state.target; render(); };
$("target").onchange = (event) => { state.target = event.target.value; render(); };
function mode(value) {
  state.mode = value; $("manual-settings").hidden = value !== "manual"; $("auto-settings").hidden = value === "manual";
  for (const name of ["manual","auto"]) { const active = name === value; $(name + "-mode").classList.toggle("selected", active); $(name + "-mode").setAttribute("aria-pressed", String(active)); }
  $("canvas-help").textContent = value === "manual" ? "Drag an outline around the object you want to hide" : "Matching objects will be checked for throughout the clip";
  ready(); render();
}
$("manual-mode").onclick = () => mode("manual"); $("auto-mode").onclick = () => mode("auto");
$("find").onchange = () => { $("description-label").hidden = $("find").value !== "text"; };
$("upload").onchange = async (event) => {
  const file = event.target.files[0]; if (!file) return;
  if (state.busy) { message("Finish or cancel the current export before opening another video."); event.target.value = ""; return; }
  message("Opening your video locally…");
  const form = new FormData(); form.append("file", file);
  try { await loadVideo(await api("/api/videos", { method: "POST", body: form })); } catch (error) { message(error.message); }
  event.target.value = "";
};

$("export").onclick = async () => {
  if (!state.video || state.busy) return;
  message(); state.busy = true; ready(); $("result").hidden = true;
  try {
    const body = { video_id: state.video.id, prompts: state.prompts, backend: $("backend").value, style: $("style").value, mode: state.mode === "manual" ? "manual" : $("find").value, text: $("description").value, audio: $("audio").checked };
    state.exportMode = body.mode;
    const response = await api("/api/exports", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    state.job = response.id; $("job-panel").hidden = false; $("job-title").textContent = "Preparing your video…"; $("job-detail").textContent = "Loading models and preparing frames. This may take a moment."; $("progress").value = 0; $("cancel").hidden = false;
    poll();
  } catch (error) { state.busy = false; ready(); message(error.message); }
};
async function poll() {
  try {
    const job = await api(`/api/exports/${state.job}`);
    if (["queued","running"].includes(job.status)) {
      if (job.done > 0) { $("job-title").textContent = "Following your targets…"; $("job-detail").textContent = `${job.done} of ${job.total} frames processed. Your original video stays unchanged.`; }
      $("progress").value = 100 * job.done / Math.max(1, job.total); setTimeout(poll, 1200); return;
    }
    state.busy = false; ready(); $("cancel").hidden = true;
    if (job.status !== "complete") { $("job-title").textContent = job.status === "cancelled" ? "Export cancelled" : "Export needs attention"; $("job-detail").textContent = job.error || "No output was published."; return; }
    $("job-panel").hidden = true; $("result").hidden = false; state.reviewFrames = job.review_frames; state.reviewTargets = job.review_targets || {}; state.reviewIndex = 0;
    const base = `/api/exports/${state.job}/file/`;
    if (state.exportMode !== "manual") {
      const saved = await api(base + "prompts"); state.prompts = saved.prompts;
      const ids = [...new Set(state.prompts.map((prompt) => prompt.id))];
      $("target").replaceChildren(...ids.map((id) => new Option(targetLabel(id), id)));
      state.target = ids[0] || "target-1"; $("target").value = state.target; selectionList();
    }
    $("result-video").src = base + "video"; $("download").href = base + "video"; $("report-link").href = base + "report"; $("prompts-link").href = base + "prompts";
    $("review-summary").textContent = job.review_frames.length ? `${job.review_frames.length} frames were covered because tracking was uncertain. Add corrections on those frames and export again.` : "No uncertain frames were flagged. Watch the full clip to check the result.";
    $("review-next").disabled = !job.review_frames.length;
  } catch (error) {
    if (error.status === 404) { state.busy = false; ready(); $("job-title").textContent = "Export session ended"; $("job-detail").textContent = "The server may have restarted. Start a new export."; $("cancel").hidden = true; }
    else { message(`Could not check export progress: ${error.message} Retrying…`); setTimeout(poll, 4000); }
  }
}
$("cancel").onclick = async () => { if (!state.job) return; try { await api(`/api/exports/${state.job}/cancel`, { method: "POST" }); $("job-detail").textContent = "Stopping after the current frame…"; } catch (error) { message(error.message); } };
$("review-next").onclick = () => { if (!state.reviewFrames.length) return; const frame = state.reviewFrames[state.reviewIndex++ % state.reviewFrames.length]; const target = state.reviewTargets[String(frame)]?.[0]; if (target) { state.target = target; $("target").value = target; } mode("manual"); seek(frame); $("stage").scrollIntoView({ block: "center", behavior: "auto" }); };

(async () => {
  try {
    const [videos, capabilities] = await Promise.all([api("/api/videos"), api("/api/capabilities")]);
    $("backend").querySelector('[value="sam2"]').disabled = !capabilities.sam2;
    if (capabilities.sam2) $("backend").value = "sam2";
    if (!capabilities.grounding) { $("find").querySelector('[value="text"]').disabled = true; $("find").querySelector('[value="plates"]').disabled = true; $("find").value = "faces"; $("description-label").hidden = true; }
    if (videos.length) await loadVideo(videos[0]);
  } catch (error) { message(error.message); }
})();
