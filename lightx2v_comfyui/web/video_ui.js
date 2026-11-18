import { api } from "../../scripts/api.js";

const style = document.createElement("style");
style.textContent = `
.lx2v-videos { height: 100%; overflow: auto; box-sizing: border-box; padding: 8px; color: #eee; background: #202020; font: 14px sans-serif; }
.lx2v-videos .lx-status { min-height: 36px; line-height: 18px; margin-bottom: 8px; }
.lx2v-videos .lx-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.lx2v-videos .lx-draft { border: 2px solid #555; border-radius: 6px; overflow: hidden; cursor: pointer; }
.lx2v-videos .lx-draft[data-selected=true] { border-color: #77bfff; background: #234c70; }
.lx2v-videos video { display: block; width: 100%; aspect-ratio: 16 / 9; object-fit: contain; background: #111; }
.lx2v-videos button { padding: 8px; color: inherit; background: #343434; border: 1px solid #777; border-radius: 4px; cursor: pointer; }
.lx2v-videos .lx-draft button { width: 100%; border: 0; border-radius: 0; background: transparent; }
.lx2v-videos .lx-clear { margin-top: 10px; }
.lx2v-videos button:focus-visible { outline: 2px solid #77bfff; outline-offset: -2px; }
`;
document.head.appendChild(style);

function element(tag, className, text = "") {
    const value = document.createElement(tag);
    value.className = className;
    value.textContent = text;
    return value;
}

function setWidget(node, widget, value) {
    const previous = widget.value;
    widget.value = value;
    widget.callback?.(value);
    node.onWidgetChanged?.(widget.name, value, previous, widget);
}

export function syncRegenerationFields(selector, preview) {
    for (const node of selector.getOutputNodes(0) || []) {
        if (node.comfyClass !== "LightX2VRegenerateVideo") continue;
        for (const name of ["prompt", "seed"]) {
            const widget = node.widgets?.find(w => w.name === name);
            const slot = node.inputs?.findIndex(input => input.name === name) ?? -1;
            if (!widget || preview[name] === undefined || (slot >= 0 && node.isInputConnected(slot))) continue;
            setWidget(node, widget, name === "seed" ? Number(preview[name]) : preview[name]);
        }
        node.setDirtyCanvas(true, true);
    }
}

export function configureVideoSelector(node) {
    if (node.__lxVideoSelector) return node.__lxVideoSelector.restore();
    const index = node.widgets.find(w => w.name === "index");
    const selection = node.widgets.find(w => w.name === "selection_batch");
    // Keep serialized API inputs while replacing their editors with the picker.
    index.hidden = selection.hidden = true;
    const root = element("div", "lx2v-videos");
    const status = element("div", "lx-status");
    status.setAttribute("role", "status");
    const grid = element("div", "lx-grid");
    const clear = element("button", "lx-clear", "Clear selection");
    root.append(status, grid, clear);
    root.addEventListener("pointerdown", event => event.stopPropagation());
    root.addEventListener("wheel", event => event.stopPropagation());
    node.addDOMWidget("draft_previews", "lightx2v_videos", root, {
        serialize: false, hideOnZoom: false, getMinHeight: () => 490, getHeight: () => 490,
    });
    node.setSize([Math.max(node.size[0], 680), Math.max(node.size[1], 640)]);
    let batch = "", previews = [], cards = [];

    function refresh() {
        const selected = selection.value === batch ? Number(index.value) : 0;
        status.textContent = !previews.length ? "Run to generate draft previews."
            : selected ? `Draft ${selected} selected · seed ${previews[selected - 1]?.seed}. Click Run to regenerate.`
                : "Click a draft to select it, then click Run to regenerate.";
        clear.hidden = !selected;
        cards.forEach(({card, button}, i) => {
            const active = selected === i + 1;
            card.dataset.selected = String(active);
            button.setAttribute("aria-pressed", String(active));
            button.textContent = `${active ? "Selected" : "Select"} draft ${i + 1} · seed ${previews[i].seed}`;
        });
        node.setDirtyCanvas(true, true);
    }

    function choose(value) {
        const graph = node.graph;
        graph?.beforeChange();
        try {
            for (const [widget, next] of [[index, value], [selection, value ? batch : ""]]) {
                setWidget(node, widget, next);
            }
            if (value) syncRegenerationFields(node, previews[value - 1]);
            refresh();
        } finally {
            graph?.afterChange();
        }
    }
    clear.onclick = () => choose(0);

    function dispose() {
        for (const {video} of cards) {
            video.pause();
            video.removeAttribute("src");
            video.load();
        }
        cards = [];
        grid.replaceChildren();
    }

    function restore() {
        const saved = node.properties?.lightx2v_previews;
        if (saved?.batch === batch) return refresh();
        dispose();
        batch = saved?.batch || "";
        previews = saved?.videos || [];
        previews.forEach((preview, i) => {
            const card = element("div", "lx-draft");
            const video = element("video", "");
            const button = element("button", "");
            video.controls = true;
            video.playsInline = true;
            video.preload = "metadata";
            video.src = api.apiURL(`/view?${new URLSearchParams({filename: preview.filename, subfolder: preview.subfolder, type: preview.type})}`);
            video.setAttribute("aria-label", `Draft ${i + 1}`);
            video.addEventListener("play", () => cards.forEach(item => { if (item.video !== video) item.video.pause(); }));
            video.addEventListener("error", () => { status.textContent = "Preview unavailable. Run to generate drafts again."; });
            card.onclick = () => choose(i + 1);
            card.append(video, button);
            grid.append(card);
            cards.push({card, video, button});
        });
        refresh();
    }

    const executed = node.onExecuted;
    node.onExecuted = function(message) {
        const result = executed?.apply(this, arguments);
        if (message.lightx2v_candidates && message.lightx2v_batch_id?.[0]) {
            const nextBatch = message.lightx2v_batch_id[0];
            node.properties ||= {};
            node.properties.lightx2v_previews = {batch: nextBatch, videos: message.lightx2v_candidates};
            if (selection.value !== nextBatch) choose(0);
            restore();
        }
        return result;
    };
    const removed = node.onRemoved;
    node.onRemoved = function() {
        dispose();
        return removed?.apply(this, arguments);
    };
    node.__lxVideoSelector = {restore};
    restore();
}
