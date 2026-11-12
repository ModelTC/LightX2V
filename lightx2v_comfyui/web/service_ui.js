import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { openConfigEditor } from "./config_ui.js";
import { fetchJSON, openFileBrowser } from "./files_ui.js";
import { configureGpuChoices } from "./gpu_ui.js";

const nodeTypes = {
    LightX2VServer: "Service",
    LightX2VGenerateVideo: "Generate Video",
    LightX2VGenerateImage: "Generate Image",
    LightX2VUpscaleVideo: "Upscale Video",
};
const ownNodes = new Set(Object.keys(nodeTypes));

function configureTypeBadge(node) {
    const type = nodeTypes[node.comfyClass];
    if (!type || typeof window.LGraphBadge !== "function") return;
    node.__lxTypeBadge ||= new window.LGraphBadge({
        text: `LightX2V-Server: ${type}`, bgColor: "#234c70", fgColor: "#deefff",
        fontSize: 14, height: 24,
    });
    node.badges ||= [];
    if (!node.badges.includes(node.__lxTypeBadge)) node.badges.push(node.__lxTypeBadge);
    node.setDirtyCanvas(true, true);
}

function element(tag, text, className) {
    const el = document.createElement(tag);
    if (text) el.textContent = text;
    if (className) el.className = className;
    return el;
}

let executionErrorDialog = null;
let lastExecutionError = "";

function showExecutionError(detail) {
    if (!ownNodes.has(detail?.node_type)) return;
    const message = String(detail.exception_message || "").trim();
    if (!message) return;
    const identity = detail.prompt_id ? JSON.stringify([detail.prompt_id, detail.node_id, message]) : "";
    if (identity && identity === lastExecutionError) return;
    lastExecutionError = identity;
    executionErrorDialog?.close();
    const dialog = element("dialog", "", "lx2v-files");
    dialog.setAttribute("aria-label", "LightX2V execution error");
    dialog.style.maxHeight = "85vh";
    dialog.style.overflowY = "auto";
    const node = app.graph?.getNodeById(detail.node_id);
    const title = node?.comfyClass === detail.node_type ? node.title || detail.node_type : detail.node_type;
    const reason = element("div", message, "lx-message");
    reason.setAttribute("role", "alert");
    const close = element("button", "Close");
    close.autofocus = true;
    close.onclick = () => dialog.close();
    const footer = element("div", "", "lx-row lx-footer");
    footer.append(close);
    dialog.append(element("h2", `${title} failed`),
        element("p", `Node ${detail.node_id ?? "?"} · ${detail.exception_type || "Error"}`), reason, footer);
    dialog.addEventListener("close", () => {
        dialog.remove();
        if (executionErrorDialog === dialog) executionErrorDialog = null;
    }, {once: true});
    document.body.appendChild(dialog);
    executionErrorDialog = dialog;
    dialog.showModal();
}

function browsePath(node, field) {
    const widget = node.widgets.find(w => w.name === field);
    const source = node.widgets.find(w => w.name === "lightx2v_path");
    openFileBrowser({
        kind: "directory",
        title: field === "lightx2v_path" ? "Choose LightX2V directory" : "Choose model directory",
        initialPath: widget.value || source?.value || "",
        onSelect(value) {
            const graph = node.graph;
            graph?.beforeChange();
            graph?.canvasAction?.(canvas => canvas.emitBeforeChange?.());
            try {
                const previous = widget.value;
                widget.value = value;
                widget.callback?.(widget.value);
                node.onWidgetChanged?.(widget.name, widget.value, previous, widget);
            } finally {
                graph?.afterChange();
                graph?.canvasAction?.(canvas => canvas.emitAfterChange?.());
                node.setDirtyCanvas(true, true);
            }
        },
    });
}

let startupModels = {};
const catalogRequests = new Map();

function configureModelChoices(node) {
    const model = node.widgets?.find(w => w.name === "model_cls");
    const task = node.widgets?.find(w => w.name === "task");
    const source = node.widgets?.find(w => w.name === "lightx2v_path");
    if (!model || !task || !source) return;
    node.__lxModels ||= startupModels;
    function updateTasks(resetInvalid = false) {
        const values = node.__lxModels[model.value] || [];
        task.options.values = [...values];
        // Loading a saved graph must never silently change its task.
        if (resetInvalid && values.length && !values.includes(task.value)) {
            task.value = values[0];
            task.callback?.(task.value);
        }
        node.setDirtyCanvas(true, true);
    }
    async function refreshCatalog() {
        const path = source.value;
        if (!catalogRequests.has(path)) {
            catalogRequests.set(path, fetchJSON("/lightx2v/models", path).finally(() => {
                catalogRequests.delete(path);
            }));
        }
        try {
            const catalog = await catalogRequests.get(path);
            if (source.value !== path) return;
            node.__lxModels = catalog.models;
            model.options.values = Object.keys(catalog.models);
            updateTasks();
        } catch (error) {
            console.warn("[LightX2V] Could not read model/task choices:", error.message);
        }
    }
    if (!node.__lxChoiceCallbacks) {
        node.__lxChoiceCallbacks = true;
        const originalModelCallback = model.callback;
        model.callback = function(...args) {
            const result = originalModelCallback?.apply(this, args);
            updateTasks(true);
            return result;
        };
        const originalPathCallback = source.callback;
        source.callback = function(...args) {
            const result = originalPathCallback?.apply(this, args);
            refreshCatalog();
            return result;
        };
    }
    updateTasks();
    refreshCatalog();
}

async function releaseService(node) {
    if (node.__lxReleasing) return;
    node.__lxReleasing = true;
    const dialog = element("dialog", "", "lx2v-files");
    dialog.setAttribute("aria-label", "Release LightX2V service");
    const message = element("div", "Release this Service's last executed backend and free its VRAM? Before the first run, the current settings identify the backend. If a request is active, it will finish before release. Other nodes sharing this backend will also need to reload it on their next uncached generation.", "lx-message");
    message.setAttribute("role", "status");
    const close = element("button", "Cancel");
    close.autofocus = true;
    close.onclick = () => dialog.close();
    const confirm = element("button", "Release");
    const footer = element("div", "", "lx-row lx-footer");
    footer.append(close, confirm);
    dialog.append(element("h2", "Release LightX2V service"), message, footer);
    dialog.addEventListener("close", () => dialog.remove(), {once: true});
    document.body.appendChild(dialog);
    dialog.showModal();
    try {
        const confirmed = await new Promise(resolve => {
            confirm.onclick = () => resolve(true);
            dialog.addEventListener("close", () => resolve(false), {once: true});
        });
        if (!confirmed) return;
        confirm.remove();
        close.textContent = "Close";
        message.textContent = "Releasing service…";
        const key = node.properties?.lightx2v_service_key;
        const fields = ["lightx2v_path", "model_path", "model_cls", "config_json", "task", "CUDA_VISIBLE_DEVICES"];
        if (!key && node.inputs?.some(input => fields.includes(input.name) && input.link != null)) {
            throw new Error("Run this Service once to resolve its connected inputs before releasing it.");
        }
        const server = Object.fromEntries(fields.map(name => [name, node.widgets.find(w => w.name === name)?.value]));
        const response = await api.fetchApi("/lightx2v/services/release", {
            method: "POST", headers: {"Content-Type": "application/json"},
            body: JSON.stringify(key ? {service_key: key} : {server}),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
        message.textContent = data.status === "release_pending"
            ? "Release queued. The service will stop after its current request finishes."
            : data.status === "not_running" ? "This service is not running."
            : "Service released. It will load again on the next uncached generation.";
        while (data.status === "release_pending" && dialog.isConnected) {
            await new Promise(resolve => setTimeout(resolve, 1000));
            if (!dialog.isConnected) break;
            const status = await api.fetchApi("/lightx2v/services");
            if (!status.ok) throw new Error(`HTTP ${status.status}`);
            const state = await status.json();
            const instance = state.instances.find(item => item.service_key === data.service_key);
            if (!instance) { message.textContent = "Service released."; break; }
            if (instance.cleanup_failed) throw new Error("Process cleanup could not be confirmed. Inspect the service logs and retry Release.");
        }
    } catch (error) {
        message.textContent = error.message;
    } finally {
        node.__lxReleasing = false;
    }
}

function configureNode(node) {
    if (!ownNodes.has(node.comfyClass)) return;
    configureTypeBadge(node);
    node.properties ||= {};
    node.properties.lightx2v_ui_version = 7;
    if (node.comfyClass === "LightX2VServer") configureModelChoices(node);
    if (node.comfyClass === "LightX2VServer" && !node.__lxBrowse) {
        node.__lxBrowse = true;
        node.addWidget("button", "Browse LightX2V", null, () => browsePath(node, "lightx2v_path"), {serialize: false});
        node.addWidget("button", "Browse model", null, () => browsePath(node, "model_path"), {serialize: false});
        node.addWidget("button", "Browse / Edit config", null, () => openConfigEditor(node), {serialize: false});
        node.addWidget("button", "Release", null, () => releaseService(node), {
            serialize: false, tooltip: "Confirm before releasing the last executed service. An active request finishes first. Services with identical settings share a backend.",
        });
    }
    if (node.__lxBrowse) {
        configureGpuChoices(node);
        node.setSize([node.size[0], Math.max(node.size[1], node.computeSize()[1])]);
    }
    const control = node.widgets?.find(w => w.name === "control_after_generate");
    if (control && !control.__lxEnglish) {
        control.__lxEnglish = true;
        Object.defineProperty(control, "label", {
            configurable: true,
            get: () => app.extensionManager?.setting?.get("Comfy.WidgetControlMode") === "before"
                ? "Control before generate" : "Control after generate",
            set: () => {},
        });
    }
}

app.registerExtension({
    name: "LightX2V.ServiceUI",
    beforeRegisterVueAppNodeDefs(nodeDefs) {
        for (let i = 0; i < nodeDefs.length; i++) {
            if (!ownNodes.has(nodeDefs[i].name)) continue;
            const registered = window.LiteGraph?.registered_node_types?.[nodeDefs[i].name]?.nodeData;
            if (!registered?.nodeSource) continue;
            // Use ComfyUI's own definition class so the store retains this
            // presentation override in both canvas and Vue renderers. Keep
            // module/registry metadata intact; our badge already names the source.
            const definition = new registered.constructor(nodeDefs[i]);
            if (!definition.nodeSource) continue;
            definition.nodeSource.badgeText = "";
            nodeDefs[i] = definition;
        }
    },
    setup() {
        // ComfyUI may summarize arbitrary backend errors in its default error panel.
        // Keep its normal handling and show the actual reason for our nodes directly.
        api.addEventListener("execution_error", ({detail}) => showExecutionError(detail));
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "LightX2VServer") {
            // UI-only combo: the backend and saved workflow keep the STRING contract.
            const device = nodeData.input.optional?.CUDA_VISIBLE_DEVICES;
            if (device?.[0] === "STRING") {
                nodeData.input.optional.CUDA_VISIBLE_DEVICES = [["0"], {...device[1]}];
            }
            startupModels = nodeData.input.required.model_cls[1].lightx2v_tasks;
            const original = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                const result = original?.apply(this, arguments);
                if (message?.lightx2v_service_key?.[0]) {
                    this.properties ||= {};
                    this.properties.lightx2v_service_key = message.lightx2v_service_key[0];
                }
                return result;
            };
        }
    },
    nodeCreated: configureNode,
    loadedGraphNode: configureNode,
});
