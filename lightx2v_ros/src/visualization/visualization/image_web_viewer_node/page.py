"""Dynamic HTML for the image web viewer.

The camera grid is generated from the active environment contract so the same
viewer renders LIBERO and RoboTwin without code changes. The page also exposes
an evaluation control panel (start/pause/resume/restart, task + scenario + seed
selection), a state banner with success/failure notification, camera visibility
toggles and an episode history table -- all backed by the `/status.json` and
`POST /control` endpoints, which the viewer node bridges to the simulator's
`{namespace}/status` / `{namespace}/control` ROS topics.
"""

import html
import json

_STYLE = """
    :root {
      color-scheme: dark;
      --bg: #0b0d10;
      --panel: #15191f;
      --panel-border: #2a313b;
      --text: #f4f7fb;
      --muted: #95a1af;
      --accent: #5eead4;
      --green: #34d399;
      --red: #f87171;
      --yellow: #fbbf24;
      --blue: #60a5fa;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      height: 56px;
      padding: 0 18px;
      border-bottom: 1px solid var(--panel-border);
      background: #101319;
    }
    h1 { margin: 0; font-size: 16px; font-weight: 650; }
    #banner {
      margin: 14px 14px 0;
      padding: 12px 16px;
      border-radius: 8px;
      border: 1px solid var(--panel-border);
      background: var(--panel);
      font-size: 15px;
      font-weight: 650;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    #banner::before { content: ""; width: 10px; height: 10px; border-radius: 50%; background: var(--muted); flex: none; }
    #banner.running { border-color: rgba(96,165,250,.5); } #banner.running::before { background: var(--blue); box-shadow: 0 0 12px var(--blue); }
    #banner.paused { border-color: rgba(251,191,36,.5); } #banner.paused::before { background: var(--yellow); box-shadow: 0 0 12px var(--yellow); }
    #banner.success { border-color: rgba(52,211,153,.6); background: rgba(52,211,153,.08); } #banner.success::before { background: var(--green); box-shadow: 0 0 14px var(--green); }
    #banner.failure { border-color: rgba(248,113,113,.6); background: rgba(248,113,113,.08); } #banner.failure::before { background: var(--red); box-shadow: 0 0 14px var(--red); }
    #banner.switching { border-color: rgba(251,191,36,.5); } #banner.switching::before { background: var(--yellow); animation: blink 1s infinite; }
    #banner.ready::before { background: var(--accent); }
    @keyframes blink { 50% { opacity: .2; } }
    .panel {
      margin: 14px;
      padding: 14px 16px;
      border: 1px solid var(--panel-border);
      border-radius: 8px;
      background: var(--panel);
    }
    .controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
    .controls label { color: var(--muted); font-size: 13px; }
    select, input[type=number] {
      background: #0e1116;
      color: var(--text);
      border: 1px solid var(--panel-border);
      border-radius: 6px;
      padding: 7px 10px;
      font-size: 13px;
    }
    input[type=number] { width: 90px; }
    button {
      border: 1px solid var(--panel-border);
      border-radius: 6px;
      background: #1b212a;
      color: var(--text);
      padding: 8px 14px;
      font-size: 13px;
      font-weight: 650;
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); }
    button:disabled { opacity: .4; cursor: not-allowed; }
    button.primary { background: rgba(94,234,212,.14); border-color: rgba(94,234,212,.5); }
    button.danger { background: rgba(248,113,113,.1); border-color: rgba(248,113,113,.4); }
    .cam-toggles { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 10px; color: var(--muted); font-size: 13px; }
    .cam-toggles label { display: inline-flex; gap: 6px; align-items: center; cursor: pointer; }
    main {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
      padding: 14px;
    }
    .view {
      position: relative;
      min-width: 0;
      overflow: hidden;
      border: 1px solid var(--panel-border);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: 0 18px 50px rgba(0, 0, 0, 0.22);
    }
    .view h2 {
      position: absolute;
      top: 10px;
      left: 10px;
      z-index: 1;
      margin: 0;
      padding: 5px 8px;
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 6px;
      background: rgba(11, 13, 16, 0.72);
      color: var(--text);
      font-size: 12px;
      font-weight: 650;
      line-height: 1;
      backdrop-filter: blur(8px);
    }
    .view img {
      display: block;
      width: 100%;
      height: 100%;
      min-height: 240px;
      object-fit: contain;
      background: #080a0d;
    }
    .view.hidden { display: none; }
    .task {
      margin: 0 14px 14px;
      padding: 14px 16px;
      border: 1px solid var(--panel-border);
      border-radius: 8px;
      background: var(--panel);
      font-size: 15px;
      line-height: 1.5;
    }
    .task span { color: var(--muted); }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--panel-border); }
    th { color: var(--muted); font-weight: 600; }
    td.outcome-success { color: var(--green); font-weight: 650; }
    td.outcome-failure { color: var(--red); font-weight: 650; }
    .stats { color: var(--muted); font-size: 13px; margin-bottom: 8px; }
    .meta { color: var(--muted); font-size: 13px; margin-left: auto; }
"""

_VIEW_TEMPLATE = """    <section class="view" data-cam="{name}">
      <h2>{label}</h2>
      <img src="/{name}.mjpg" alt="{label}">
    </section>"""

_SCRIPT = """
    const POLICY_CAMS = __POLICY_CAMS__;
    let lastState = null;
    let statusData = {};

    function el(id) { return document.getElementById(id); }

    function stateLabel(s) {
      const m = {
        ready: "就绪 — 点击「开始」运行评测",
        running: "评测运行中",
        paused: "已暂停",
        success: "✔ 评测成功",
        failure: "✘ 评测失败（超过步数上限）",
        switching: "切换/重建场景中，请稍候…",
      };
      return m[s] || s || "等待模拟器状态…";
    }

    async function post(cmd, extra) {
      const body = Object.assign({ cmd: cmd }, extra || {});
      try {
        await fetch("/control", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      } catch (e) { console.error(e); }
      setTimeout(refreshStatus, 300);
    }

    function applyTask() {
      const task = el("task-select").value;
      const config = el("config-select").value;
      const seed = el("seed-input").value;
      const extra = { task_name: task, task_config: config };
      if (seed !== "") extra.seed = parseInt(seed, 10);
      post("set_task", extra);
    }

    function fillSelect(select, options, current) {
      if (!options || !options.length) return;
      if (select.dataset.filled === "1") { select.dataset.current = current || ""; return; }
      select.innerHTML = "";
      for (const opt of options) {
        const o = document.createElement("option");
        o.value = opt; o.textContent = opt;
        select.appendChild(o);
      }
      if (current && options.includes(current)) select.value = current;
      select.dataset.filled = "1";
    }

    function renderHistory(history, stats) {
      const tbody = el("history-body");
      if (!history) return;
      tbody.innerHTML = "";
      for (const h of history.slice().reverse()) {
        const tr = document.createElement("tr");
        const outcome = h.outcome === "success" ? "成功" : "失败";
        tr.innerHTML = `<td>${h.episode}</td><td>${h.task}</td><td>${h.config || "-"}</td><td>${h.seed ?? "-"}</td>` +
          `<td class="outcome-${h.outcome}">${outcome}</td><td>${h.steps}</td>`;
        tbody.appendChild(tr);
      }
      if (stats && stats.episodes > 0) {
        const rate = (100 * stats.successes / stats.episodes).toFixed(0);
        el("stats").textContent = `共 ${stats.episodes} 个 episode，成功 ${stats.successes} 个，成功率 ${rate}%`;
      } else {
        el("stats").textContent = "暂无已完成的 episode";
      }
    }

    function updateButtons(s) {
      el("btn-start").disabled = (s === "running" || s === "switching");
      el("btn-pause").disabled = (s !== "running");
      el("btn-restart").disabled = (s === "switching");
      el("btn-apply").disabled = (s === "switching");
      el("btn-start").textContent = (s === "paused") ? "继续" : "开始";
    }

    async function refreshStatus() {
      try {
        const r = await fetch("/status.json", { cache: "no-store" });
        statusData = await r.json();
      } catch (e) { return; }
      const s = statusData.state;
      const banner = el("banner");
      banner.className = s || "";
      let text = stateLabel(s);
      if (s === "running" && statusData.max_episode_steps) {
        text += `（第 ${statusData.episode} 轮 · ${statusData.episode_step}/${statusData.max_episode_steps} 步）`;
      }
      banner.querySelector("span").textContent = text;
      el("meta").textContent = statusData.task_name
        ? `${statusData.task_name} · ${statusData.task_config || ""} · seed ${statusData.seed ?? "-"}`
        : "";
      if (s !== lastState && (s === "success" || s === "failure") &&
          (lastState === "running" || lastState === "paused")) {
        // Episode just finished: make it unmissable.
        banner.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
      lastState = s;

      if (statusData.supports_task_switch) {
        el("task-panel").style.display = "";
        fillSelect(el("task-select"), statusData.available_tasks, statusData.task_name);
        fillSelect(el("config-select"), statusData.available_task_configs, statusData.task_config);
      }
      renderHistory(statusData.history, statusData.stats);
      updateButtons(s);
    }

    async function refreshTask() {
      try {
        const response = await fetch("/task.txt", { cache: "no-store" });
        const text = (await response.text()).trim();
        el("task").textContent = text || "waiting for task description";
      } catch {}
    }

    function toggleCam(name, visible) {
      const view = document.querySelector(`.view[data-cam="${name}"]`);
      if (!view) return;
      const img = view.querySelector("img");
      if (visible) {
        view.classList.remove("hidden");
        if (!img.getAttribute("src")) img.setAttribute("src", `/${name}.mjpg`);
      } else {
        view.classList.add("hidden");
        img.removeAttribute("src");  // close the MJPEG stream to save bandwidth
      }
    }

    document.addEventListener("DOMContentLoaded", () => {
      el("btn-start").addEventListener("click", () => post(lastState === "paused" ? "resume" : "start"));
      el("btn-pause").addEventListener("click", () => post("pause"));
      el("btn-restart").addEventListener("click", () => post("restart"));
      el("btn-apply").addEventListener("click", applyTask);
      document.querySelectorAll(".cam-toggles input").forEach((box) => {
        box.addEventListener("change", () => toggleCam(box.value, box.checked));
      });
      refreshStatus();
      refreshTask();
      setInterval(refreshStatus, 1000);
      setInterval(refreshTask, 1500);
    });
"""


def render_index(cameras, title="LightX2V ROS", policy_cameras=None):
    views = "\n".join(_VIEW_TEMPLATE.format(name=html.escape(str(cam)), label=html.escape(str(cam))) for cam in cameras)
    toggles = "\n".join(f'        <label><input type="checkbox" value="{html.escape(str(cam))}" checked> {html.escape(str(cam))}</label>' for cam in cameras)
    safe_title = html.escape(str(title))
    script = _SCRIPT.replace("__POLICY_CAMS__", json.dumps(list(policy_cameras or [])))
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>{_STYLE}</style>
</head>
<body>
  <header>
    <h1>{safe_title}</h1>
    <div class="meta" id="meta"></div>
  </header>

  <div id="banner"><span>等待模拟器状态…</span></div>

  <section class="panel">
    <div class="controls">
      <button id="btn-start" class="primary">开始</button>
      <button id="btn-pause">暂停</button>
      <button id="btn-restart" class="danger">重启 (新场景)</button>
    </div>
    <div class="controls" id="task-panel" style="margin-top: 10px; display: none;">
      <label>任务</label>
      <select id="task-select"></select>
      <label>场景</label>
      <select id="config-select"></select>
      <label>seed</label>
      <input id="seed-input" type="number" placeholder="自动">
      <button id="btn-apply">切换任务</button>
      <label style="opacity:.7">（切换需重建场景，约 10~30 秒）</label>
    </div>
    <div class="cam-toggles">
{toggles}
    </div>
  </section>

  <main>
{views}
  </main>

  <section class="task" id="task"><span>waiting for task description</span></section>

  <section class="panel">
    <div class="stats" id="stats">暂无已完成的 episode</div>
    <table>
      <thead><tr><th>Episode</th><th>任务</th><th>场景</th><th>Seed</th><th>结果</th><th>步数</th></tr></thead>
      <tbody id="history-body"></tbody>
    </table>
  </section>

  <script>{script}</script>
</body>
</html>
"""
