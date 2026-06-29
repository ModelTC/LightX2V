INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LightX2V ROS</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b0d10;
      --panel: #15191f;
      --panel-border: #2a313b;
      --text: #f4f7fb;
      --muted: #95a1af;
      --accent: #5eead4;
    }
    * {
      box-sizing: border-box;
    }
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
    h1 {
      margin: 0;
      font-size: 16px;
      font-weight: 650;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .status::before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 14px rgba(94, 234, 212, 0.75);
    }
    main {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
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
      min-height: 260px;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      background: #080a0d;
    }
    .task {
      margin: 0 14px 14px;
      padding: 14px 16px;
      border: 1px solid var(--panel-border);
      border-radius: 8px;
      background: var(--panel);
      color: var(--text);
      font-size: 15px;
      line-height: 1.5;
    }
    .task span {
      color: var(--muted);
    }
  </style>
</head>
<body>
  <header>
    <h1>LightX2V ROS</h1>
    <div class="status">live streams</div>
  </header>
  <main>
    <section class="view">
      <h2>agentview</h2>
      <img src="/agentview.mjpg" alt="agentview">
    </section>
    <section class="view">
      <h2>wrist</h2>
      <img src="/wrist.mjpg" alt="wrist">
    </section>
    <section class="view">
      <h2>frontview</h2>
      <img src="/frontview.mjpg" alt="frontview">
    </section>
    <section class="view">
      <h2>galleryview</h2>
      <img src="/galleryview.mjpg" alt="galleryview">
    </section>
  </main>
  <section class="task" id="task"><span>waiting for task description</span></section>
  <script>
    async function refreshTask() {
      try {
        const response = await fetch("/task.txt", { cache: "no-store" });
        const text = (await response.text()).trim();
        document.getElementById("task").textContent = text || "waiting for task description";
      } catch {
        document.getElementById("task").textContent = "waiting for task description";
      }
    }
    refreshTask();
    setInterval(refreshTask, 1000);
  </script>
</body>
</html>
"""
