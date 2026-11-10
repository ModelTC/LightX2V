import { api } from "../../scripts/api.js";
import { openFileBrowser } from "./files_ui.js";

const style = document.createElement("style");
style.textContent = `
.lx2v-config { max-height: 90vh; overflow: auto; }
.lx2v-config .lx-config-path { flex: 1; align-self: center; overflow-wrap: anywhere; }
.lx2v-config textarea { width: 100%; box-sizing: border-box; min-height: 32vh; resize: vertical; padding: 12px; background: #151515; color: #eee; font: 13px/1.5 monospace; border: 1px solid #777; }
.lx2v-config pre { max-height: 25vh; overflow: auto; white-space: pre-wrap; background: #151515; padding: 12px; }
.lx2v-config table { width: 100%; border-collapse: collapse; margin: 10px 0; }
.lx2v-config td, .lx2v-config th { padding: 4px; text-align: left; }
.lx2v-config td input { width: 100%; box-sizing: border-box; }
.lx2v-config td summary { white-space: nowrap; }
.lx2v-config td:first-child { width: 50%; }
.lx2v-config .lx-lora-path { display: flex; align-items: center; gap: 6px; }
.lx2v-config summary { cursor: pointer; margin: 10px 0; }
`;
document.head.appendChild(style);

function el(tag, text = "", className = "") {
    const item = document.createElement(tag);
    item.textContent = text;
    item.className = className;
    return item;
}

async function request(route, options) {
    const response = await api.fetchApi(route, options);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
}

export function openConfigEditor(node) {
    const widget = node.widgets.find(w => w.name === "config_json");
    const base = node.widgets.find(w => w.name === "lightx2v_path")?.value || "";
    const dialog = el("dialog", "", "lx2v-files lx2v-config");
    dialog.setAttribute("aria-label", "LightX2V config editor");
    const path = el("span", widget.value || "No config selected", "lx-config-path");
    const browse = el("button", "Browse…");
    browse.setAttribute("aria-label", "Browse config JSON");
    const pathRow = el("div", "", "lx-row");
    pathRow.append(path, browse);
    const text = el("textarea");
    text.setAttribute("aria-label", "Config JSON");
    text.spellcheck = false;
    const message = el("div", "Loading config…", "lx-message");
    message.setAttribute("role", "status");
    const loras = el("details");
    loras.open = true;
    loras.append(el("summary", "LoRA — edits update the same JSON"));
    const table = el("table");
    const add = el("button", "Add LoRA");
    loras.append(table, add);
    const loaded = el("details");
    loaded.append(el("summary", "Service startup config (read only)"));
    const loadedText = el("pre", "No running instance recorded for this Service.");
    loaded.append(loadedText);
    const destination = el("input");
    destination.setAttribute("aria-label", "Save as path");
    destination.placeholder = "New JSON path on the ComfyUI filesystem";
    const saveAs = el("button", "Save as");
    const targetRow = el("div", "", "lx-row");
    targetRow.append(destination, saveAs);
    const footer = el("div", "", "lx-row lx-footer");
    const reload = el("button", "Reload");
    const save = el("button", "Save");
    const close = el("button", "Close");
    footer.append(reload, save, close);
    dialog.append(el("h2", "Browse / Edit config"), pathRow,
        el("p", "Save updates the file and keeps a backup. Changes apply on the next run; a running service keeps its loaded config. Use Save as for an independent workflow."),
        text, loras, loaded, targetRow, message, footer);
    let original = null, baseline = "", pending = false, loraError = "";
    function selectPath(value) {
        if (widget.value === value) return;
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
    }
    function parsed() {
        const data = JSON.parse(text.value);
        if (!data || Array.isArray(data) || typeof data !== "object") throw new Error("Config must be a JSON object");
        if (data.lora_configs != null && (!Array.isArray(data.lora_configs) || data.lora_configs.some(l => !l || Array.isArray(l) || typeof l !== "object"))) {
            throw new Error("lora_configs must be an array of objects");
        }
        return data;
    }
    function validate() {
        let valid = false;
        try { if (loraError) throw new Error(loraError); parsed(); valid = true; message.textContent = text.value === baseline ? "File loaded." : "Valid JSON · unsaved changes"; }
        catch (error) { message.textContent = error.message; }
        save.disabled = !valid || !original || pending;
        saveAs.disabled = !valid || !destination.value.trim() || pending;
        add.disabled = !valid || pending;
        browse.disabled = pending;
        reload.disabled = pending || !widget.value;
        text.disabled = pending;
        destination.disabled = pending;
        for (const input of table.querySelectorAll("input, button")) input.disabled = pending;
        return valid;
    }
    function renderLoras() {
        loraError = "";
        table.replaceChildren();
        if (!validate()) return;
        const data = parsed();
        const entries = data.lora_configs || [];
        const header = el("tr");
        for (const name of ["Path", "Strength", "Alpha", "", ""]) header.append(el("th", name));
        table.append(header);
        entries.forEach((entry, index) => {
            const row = el("tr");
            for (const key of ["path", "strength", "alpha", "name"]) {
                const cell = el("td"), input = el("input");
                input.value = entry[key] ?? "";
                input.setAttribute("aria-label", `LoRA ${index + 1} ${key}`);
                input.oninput = () => {
                    try {
                        loraError = "";
                        const updated = parsed(), value = input.value;
                        if (value === "" && key !== "path") delete updated.lora_configs[index][key];
                        else {
                            const result = ["strength", "alpha"].includes(key) ? Number(value) : value;
                            if (typeof result === "number" && !Number.isFinite(result)) throw new Error(`${key} must be finite`);
                            updated.lora_configs[index][key] = result;
                        }
                        text.value = JSON.stringify(updated, null, 2) + "\n";
                        validate();
                    } catch (error) { loraError = error.message; validate(); }
                };
                if (key === "name") {
                    const advanced = el("details");
                    const label = el("label", "Name");
                    label.append(input);
                    advanced.append(el("summary", "Advanced"), label);
                    cell.append(advanced);
                } else if (key === "path") {
                    const pathRow = el("div", "", "lx-lora-path");
                    const browse = el("button", "Browse");
                    browse.setAttribute("aria-label", `Browse LoRA ${index + 1} weights`);
                    browse.onclick = () => {
                        if (pending) return;
                        openFileBrowser({kind: "lora", base,
                            initialPath: input.value || node.widgets.find(w => w.name === "model_path")?.value || base,
                            onSelect: value => { input.value = value; input.oninput(); },
                        });
                    };
                    pathRow.append(input, browse);
                    cell.append(pathRow);
                } else cell.append(input);
                row.append(cell);
            }
            const remove = el("button", "Remove");
            remove.onclick = () => { const data = parsed(); data.lora_configs.splice(index, 1); text.value = JSON.stringify(data, null, 2) + "\n"; renderLoras(); };
            const cell = el("td"); cell.append(remove); row.append(cell); table.append(row);
        });
    }
    function canDiscard(message) {
        return text.value === baseline || window.confirm(message);
    }
    async function load(value = widget.value, select = false) {
        if (pending) return;
        pending = true; validate();
        message.textContent = "Loading config…";
        try {
            const query = new URLSearchParams({path: value, base});
            if (node.properties?.lightx2v_service_key) query.set("service_key", node.properties.lightx2v_service_key);
            const data = await request(`/lightx2v/configs/content?${query}`);
            const formatted = JSON.stringify(JSON.parse(data.content), null, 2) + "\n";
            if (select) selectPath(data.path);
            original = data;
            // Display formatting is clean; retain the file's original hash for save conflicts.
            text.value = baseline = formatted;
            path.textContent = original.path;
            destination.value = original.path.replace(/\.json$/i, ".copy.json");
            loadedText.textContent = original.loaded
                ? `${original.loaded.busy ? "Running" : "Resident"} · ${original.loaded.config_json}\n${original.loaded.content}`
                : "No running instance recorded for this Service.";
            pending = false; renderLoras();
        } catch (error) { pending = false; validate(); message.textContent = error.message; }
    }
    async function persist(createNew) {
        if (pending || !validate()) return;
        pending = true; validate();
        try {
            const data = await request("/lightx2v/configs/save", {
                method: "POST", headers: {"Content-Type": "application/json"},
                body: JSON.stringify({path: createNew ? destination.value : original.path, base,
                    content: text.value, expected_sha256: original?.sha256, create_new: createNew}),
            });
            original = data; baseline = data.content; path.textContent = data.path;
            selectPath(data.path);
            pending = false; validate();
            message.textContent = `Saved. Changes apply on the next run.${data.backup ? `\nBackup: ${data.backup}` : ""}`;
        } catch (error) { pending = false; validate(); message.textContent = error.message; }
    }
    text.oninput = renderLoras;
    destination.oninput = validate;
    add.onclick = () => { const data = parsed(); (data.lora_configs ||= []).push({path: "", strength: 1}); text.value = JSON.stringify(data, null, 2) + "\n"; renderLoras(); };
    save.onclick = () => persist(false);
    saveAs.onclick = () => persist(true);
    browse.onclick = () => {
        if (pending) return;
        openFileBrowser({kind: "config", base,
            initialPath: original?.path || widget.value || (base ? `${base}/configs` : ""),
            onSelect(value) {
                if (canDiscard("Discard unsaved config edits and open another file?")) load(value, true);
            },
        });
    };
    reload.onclick = () => { if (canDiscard("Discard unsaved config edits and reload the file?")) load(); };
    function canClose() { return !pending && canDiscard("Close without saving config edits?"); }
    close.onclick = () => { if (canClose()) dialog.close(); };
    dialog.addEventListener("cancel", event => { if (!canClose()) event.preventDefault(); });
    dialog.addEventListener("close", () => dialog.remove(), {once: true});
    document.body.appendChild(dialog);
    dialog.showModal();
    if (widget.value) load();
    else { validate(); message.textContent = "Browse to choose a config JSON file, or enter JSON and use Save as."; }
}
