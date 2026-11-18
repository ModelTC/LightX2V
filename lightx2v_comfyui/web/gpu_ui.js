import { api } from "../../scripts/api.js";

const selectors = new Set();
const MULTIPLE = "Select multiple GPUs…";
let snapshot = null;
let failure = "";
let pending = null;

const style = document.createElement("style");
style.textContent = `
.lx2v-files .lx-gpu-row { display: flex; align-items: center; gap: 12px; padding: 10px; cursor: pointer; }
.lx2v-files .lx-gpu-row:hover { background: #333; }
.lx2v-files .lx-gpu-row input { flex: none; width: 18px; height: 18px; }
.lx2v-files .lx-gpu-row span { white-space: pre-line; }
`;
document.head.appendChild(style);

function deviceTokens(value) {
    return String(value ?? "").split(",").map(part => part.trim()).filter(Boolean);
}

export function selectedGpuCount(node) {
    return deviceTokens(node.widgets?.find(widget => widget.name === "CUDA_VISIBLE_DEVICES")?.value).length;
}

export function setParallelConfig(config, count) {
    config.parallel = count > 1 ? {seq_p_size: count, seq_p_attn_type: "ulysses"} : false;
    if (count <= 1) {
        for (const key of ["text_encoder_tensor_parallel", "vae_encode_parallel", "vae_decode_parallel"]) {
            if (key in config) config[key] = false;
        }
    }
}

async function configRequest(route, options) {
    const response = await api.fetchApi(route, options);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
}

async function saveGpuConfig(node, count) {
    const path = node.widgets.find(widget => widget.name === "config_json")?.value;
    const base = node.widgets.find(widget => widget.name === "lightx2v_path")?.value || "";
    if (!path) throw new Error("Choose a config JSON file before selecting GPUs.");
    if (node.inputs?.some(input => ["config_json", "lightx2v_path"].includes(input.name) && input.link != null)) {
        throw new Error("Automatic GPU configuration needs config_json and lightx2v_path on this Service, not connected inputs.");
    }
    const original = await configRequest(`/lightx2v/configs/content?${new URLSearchParams({path, base})}`);
    const config = JSON.parse(original.content), before = JSON.stringify(config);
    setParallelConfig(config, count);
    if (JSON.stringify(config) === before) return;
    await configRequest("/lightx2v/configs/save", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({path: original.path, base,
            content: JSON.stringify(config, null, 2) + "\n", expected_sha256: original.sha256}),
    });
}

async function applyGpuSelection(selector, value) {
    if (selector.saving) return false;
    const {node, widget} = selector;
    selector.saving = true;
    selector.updateDialog?.();
    try {
        await saveGpuConfig(node, deviceTokens(value).length);
        const graph = node.graph;
        graph?.beforeChange();
        graph?.canvasAction?.(canvas => canvas.emitBeforeChange?.());
        try {
            const previous = widget.value;
            widget.value = value;
            selector.callback?.call(widget, value);
            node.onWidgetChanged?.(widget.name, value, previous, widget);
        } finally {
            graph?.afterChange();
            graph?.canvasAction?.(canvas => canvas.emitAfterChange?.());
        }
        return true;
    } finally {
        selector.saving = false;
        render(selector);
    }
}

function selectedDevice(value) {
    const text = String(value ?? "").trim();
    const devices = snapshot?.devices || [];
    if (/^\d+$/.test(text)) return devices.find(device => device.index === Number(text));
    if (!text.startsWith("GPU-")) return null;
    const matches = devices.filter(device => device.uuid.startsWith(text));
    return matches.length === 1 ? matches[0] : null;
}

function deviceLabel(value) {
    const device = selectedDevice(value);
    if (!device) return `${value || "No GPU selected"} · ${failure ? "status unavailable" : snapshot ? "not visible" : "loading…"}`;
    const gib = amount => amount == null ? "?" : (amount / 1024).toFixed(1);
    const utilization = device.utilization_gpu_percent == null ? "?" : `${device.utilization_gpu_percent}%`;
    return `GPU ${device.index} · VRAM ${gib(device.memory_used_mib)}/${gib(device.memory_total_mib)} GiB · GPU ${utilization}${failure ? " · stale" : ""}`;
}

function gpuLabel(value) {
    if (value === MULTIPLE) return MULTIPLE;
    const devices = deviceTokens(value);
    if (devices.length < 2) return deviceLabel(value);
    const names = devices.map(value => selectedDevice(value)?.index ?? value);
    return `GPUs ${names.join(", ")} · ${devices.length} processes${failure ? " · status unavailable" : ""}`;
}

function render(selector) {
    const {node, widget} = selector;
    const value = String(widget.value ?? "");
    const selected = selectedDevice(value);
    const values = (snapshot?.devices || []).map(device => selected === device ? value : String(device.index));
    // Preserve old UUIDs, multi-device strings, and unavailable saved selections.
    // Refreshing usage must never change the workflow's device or its cache key.
    if (!values.includes(value)) values.unshift(value);
    values.push(MULTIPLE);
    widget.options.values = values;
    widget.options.getOptionLabel = gpuLabel;
    selector.value = value;
    const stamp = snapshot ? new Date(snapshot.sampled_at * 1000).toLocaleTimeString() : "not yet sampled";
    const identity = selected ? `${selected.name}\n${selected.uuid}\n` : "";
    widget.tooltip = `${identity}${deviceTokens(value).map(deviceLabel).join("\n")}\nUpdated ${stamp}. Selecting GPUs saves Ulysses + SP in the current config; one GPU disables parallelism.\n${failure || "Refresh every 5 seconds while visible. Usage does not reserve a GPU."}`;
    selector.updateDialog?.();
    node.setDirtyCanvas(true, true);
}

async function openGpuPicker(selector) {
    if (selector.dialog) return;
    const {widget} = selector;
    const dialog = document.createElement("dialog");
    selector.dialog = dialog;
    dialog.className = "lx2v-files";
    dialog.setAttribute("aria-label", "Select GPUs for LightX2V");
    const title = document.createElement("h2");
    title.textContent = "Select GPUs";
    const hint = document.createElement("p");
    hint.textContent = "Apply saves Ulysses + SP for the checked GPUs in this Service's config file. One GPU disables parallelism. Usage updates automatically.";
    const list = document.createElement("div");
    list.className = "lx-list";
    const status = document.createElement("div");
    status.className = "lx-message";
    status.setAttribute("role", "status");
    const footer = document.createElement("div");
    footer.className = "lx-row lx-footer";
    const cancel = document.createElement("button");
    cancel.textContent = "Cancel";
    const apply = document.createElement("button");
    apply.textContent = "Apply";
    footer.append(cancel, apply);
    dialog.append(title, hint, list, status, footer);
    const selected = new Set(deviceTokens(widget.value).map(value => selectedDevice(value)?.uuid ?? value));
    const rows = new Map();
    let saveError = "";
    selector.updateDialog = () => {
        const resolved = [...selected].map(value => selectedDevice(value)?.uuid ?? value);
        selected.clear();
        for (const value of resolved) selected.add(value);
        const choices = [...new Set([...(snapshot?.devices || []).map(device => device.uuid), ...selected])];
        for (const value of choices) {
            if (!rows.has(value)) {
                const row = document.createElement("label");
                row.className = "lx-gpu-row";
                const input = document.createElement("input");
                input.type = "checkbox";
                const label = document.createElement("span");
                input.onchange = () => {
                    saveError = "";
                    if (input.checked) selected.add(value);
                    else selected.delete(value);
                    selector.updateDialog();
                };
                row.append(input, label); list.append(row);
                rows.set(value, {row, input, label});
            }
            const {input, label} = rows.get(value);
            const device = selectedDevice(value);
            input.checked = selected.has(value);
            input.disabled = Boolean(selector.saving);
            input.setAttribute("aria-label", device ? `GPU ${device.index}` : value);
            label.textContent = `${deviceLabel(value)}\n${device?.name || value}`;
        }
        for (const [value, row] of rows) {
            if (!choices.includes(value)) { row.row.remove(); rows.delete(value); }
        }
        const names = [...selected].map(value => selectedDevice(value)?.index ?? value);
        status.textContent = `${selected.size} GPU(s) selected${names.length ? `: ${names.join(", ")} (process order)` : ""}.${selector.saving ? "\nSaving config…" : saveError ? `\n${saveError}` : failure ? `\n${failure}` : ""}`;
        apply.disabled = !selected.size || selector.saving;
        cancel.disabled = Boolean(selector.saving);
    };
    cancel.onclick = () => dialog.close();
    apply.onclick = async () => {
        saveError = "";
        try {
            if (await applyGpuSelection(selector, [...selected].join(","))) dialog.close();
        } catch (error) {
            saveError = `GPU selection unchanged. ${error.message}`;
            selector.updateDialog?.();
        }
    };
    dialog.addEventListener("cancel", event => { if (selector.saving) event.preventDefault(); });
    dialog.addEventListener("close", () => {
        selector.dialog = null;
        selector.updateDialog = null;
        dialog.remove();
    }, {once: true});
    document.body.appendChild(dialog);
    selector.updateDialog();
    dialog.showModal();
    await refreshGpus();
}

function refreshGpus() {
    if (pending) return pending;
    pending = (async () => {
        try {
            const response = await api.fetchApi("/lightx2v/gpus", {cache: "no-store"});
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
            if (!Array.isArray(data.devices)) throw new Error("Invalid GPU status response");
            snapshot = data;
            failure = "";
        } catch (error) {
            failure = error.message || "GPU status unavailable";
        } finally {
            pending = null;
            for (const selector of selectors) render(selector);
        }
    })();
    return pending;
}

export function configureGpuChoices(node) {
    if (node.__lxGpuSelector) {
        render(node.__lxGpuSelector);
        return;
    }
    const widget = node.widgets?.find(widget => widget.name === "CUDA_VISIBLE_DEVICES");
    if (!widget) return;
    const selector = {node, widget, callback: widget.callback};
    node.__lxGpuSelector = selector;
    selectors.add(selector);
    widget.callback = function() {
        const value = String(widget.value ?? "");
        widget.value = selector.value;
        if (value === MULTIPLE) {
            openGpuPicker(selector);
            return;
        }
        return applyGpuSelection(selector, value).catch(error => {
            window.alert(`GPU selection unchanged. ${error.message}`);
        });
    };
    const originalRemoved = node.onRemoved;
    node.onRemoved = function(...args) {
        selector.dialog?.close();
        selectors.delete(selector);
        return originalRemoved?.apply(this, args);
    };
    const originalAdded = node.onAdded;
    node.onAdded = function(...args) {
        selectors.add(selector);
        const result = originalAdded?.apply(this, args);
        refreshGpus();
        return result;
    };
    render(selector);
    refreshGpus();
}

setInterval(() => {
    if (selectors.size && document.visibilityState === "visible") refreshGpus();
}, 5000);

document.addEventListener("visibilitychange", () => {
    if (selectors.size && document.visibilityState === "visible") refreshGpus();
});
