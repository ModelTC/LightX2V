import { api } from "../../scripts/api.js";

const selectors = new Set();
let snapshot = null;
let failure = "";
let pending = null;

function selectedDevice(value) {
    const text = String(value ?? "").trim();
    const devices = snapshot?.devices || [];
    if (/^\d+$/.test(text)) return devices.find(device => device.index === Number(text));
    if (!text.startsWith("GPU-")) return null;
    const matches = devices.filter(device => device.uuid.startsWith(text));
    return matches.length === 1 ? matches[0] : null;
}

function gpuLabel(value) {
    const device = selectedDevice(value);
    if (!device) return `${value || "No GPU selected"} · ${failure ? "status unavailable" : snapshot ? "not visible" : "loading…"}`;
    const gib = amount => amount == null ? "?" : (amount / 1024).toFixed(1);
    const utilization = device.utilization_gpu_percent == null ? "?" : `${device.utilization_gpu_percent}%`;
    return `GPU ${device.index} · VRAM ${gib(device.memory_used_mib)}/${gib(device.memory_total_mib)} GiB · GPU ${utilization}${failure ? " · stale" : ""}`;
}

function render(selector) {
    const {node, widget} = selector;
    const value = String(widget.value ?? "");
    const selected = selectedDevice(value);
    const values = (snapshot?.devices || []).map(device => selected === device ? value : String(device.index));
    // Preserve old UUIDs, multi-device strings, and unavailable saved selections.
    // Refreshing usage must never change the workflow's device or its cache key.
    if (!values.includes(value)) values.unshift(value);
    widget.options.values = values;
    widget.options.getOptionLabel = gpuLabel;
    const stamp = snapshot ? new Date(snapshot.sampled_at * 1000).toLocaleTimeString() : "not yet sampled";
    const identity = selected ? `${selected.name}\n${selected.uuid}\n` : "";
    widget.tooltip = `${identity}VRAM: used/total GiB; GPU: utilization. Updated ${stamp}.\n${failure || "Refresh every 5 seconds while this page is visible. Usage does not reserve a GPU."}`;
    node.setDirtyCanvas(true, true);
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
    const selector = {node, widget};
    node.__lxGpuSelector = selector;
    selectors.add(selector);
    const originalCallback = widget.callback;
    widget.callback = function(...args) {
        const result = originalCallback?.apply(this, args);
        render(selector);
        return result;
    };
    const originalRemoved = node.onRemoved;
    node.onRemoved = function(...args) {
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
