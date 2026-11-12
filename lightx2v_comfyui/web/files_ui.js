import { api } from "../../scripts/api.js";

const style = document.createElement("style");
style.textContent = `
.lx2v-files { color: #eee; background: #242424; border: 1px solid #666; border-radius: 10px; padding: 22px; width: min(900px, 88vw); }
.lx2v-files::backdrop { background: #0009; }
.lx2v-files h2 { margin: 0 0 8px; font-size: 20px; }
.lx2v-files p { color: #bbb; margin: 0 0 15px; }
.lx2v-files .lx-row { display: flex; gap: 8px; margin: 10px 0; }
.lx2v-files input { flex: 1; min-width: 0; background: #151515; color: #eee; border: 1px solid #777; border-radius: 4px; padding: 8px; }
.lx2v-files button { background: #393939; color: #eee; border: 1px solid #777; border-radius: 4px; padding: 8px 12px; cursor: pointer; }
.lx2v-files button:disabled { opacity: .45; cursor: default; }
.lx2v-files .lx-list { overflow: auto; height: min(46vh, 430px); border: 1px solid #555; border-radius: 4px; }
.lx2v-files .lx-entry { display: block; border: 0; border-radius: 0; background: transparent; text-align: left; width: 100%; }
.lx2v-files .lx-entry:hover { background: #444; }
.lx2v-files .lx-entry[aria-pressed=true] { background: #315e8c; }
.lx2v-files .lx-message { white-space: pre-wrap; overflow-wrap: anywhere; min-height: 2em; margin: 12px 0; }
.lx2v-files .lx-footer { justify-content: flex-end; }
`;
document.head.appendChild(style);

function element(tag, text, className) {
    const el = document.createElement(tag);
    if (text) el.textContent = text;
    if (className) el.className = className;
    return el;
}

export async function fetchJSON(route, path, base = "", options = {}) {
    const query = new URLSearchParams({path, base, ...options});
    const response = await api.fetchApi(`${route}?${query}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
}

export function openFileBrowser({kind = "config", initialPath = "", base = "", title, onSelect}) {
    const directory = kind === "directory", lora = kind === "lora";
    title ||= directory ? "Choose directory" : lora ? "Choose LoRA weights" : "Choose config JSON";
    const hint = directory ? "Open a folder, then choose Use directory."
        : lora ? "Select an existing .safetensors weight file." : "Select an existing JSON file.";
    const route = directory ? "/lightx2v/directories/resolve"
        : lora ? "/lightx2v/loras/resolve" : "/lightx2v/configs/resolve";
    const resultKey = directory ? "directory" : lora ? "lora_path" : "config_json";
    const dialog = element("dialog", "", "lx2v-files");
    dialog.setAttribute("aria-label", title);
    dialog.append(element("h2", title), element("p", `Files and folders visible to ComfyUI. ${hint}`));
    const navigation = element("div", "", "lx-row");
    const up = element("button", "Up");
    up.disabled = true;
    const path = element("input");
    path.setAttribute("aria-label", "Directory path");
    const go = element("button", "Go");
    navigation.append(up, path, go);
    const filterRow = element("div", "", "lx-row");
    const filter = element("input");
    filter.placeholder = "Filter names";
    filter.setAttribute("aria-label", "Filter names");
    filterRow.append(filter);
    const list = element("div", "", "lx-list");
    list.setAttribute("role", "group");
    list.setAttribute("aria-label", directory ? "Folders" : "Files and folders");
    const message = element("div", "", "lx-message");
    message.setAttribute("role", "status");
    const footer = element("div", "", "lx-row lx-footer");
    const cancel = element("button", "Cancel");
    const use = element("button", directory ? "Use directory" : lora ? "Use weights" : "Use config");
    use.disabled = true;
    footer.append(cancel, use);
    dialog.append(navigation, filterRow, list, message, footer);
    document.body.appendChild(dialog);
    dialog.addEventListener("close", () => dialog.remove(), {once: true});
    cancel.onclick = () => dialog.close();
    let current = null, selected = "", sequence = 0;
    function render() {
        list.replaceChildren();
        for (const entry of current?.entries || []) {
            if (!entry.name.toLowerCase().includes(filter.value.toLowerCase())) continue;
            const button = element("button", `${entry.directory ? "Folder" : lora ? "LoRA" : "JSON"}  ${entry.name}`, "lx-entry");
            button.setAttribute("aria-pressed", String(selected === entry.path));
            button.onclick = () => {
                if (entry.directory) return load(entry.path);
                selected = entry.path;
                use.disabled = false;
                message.textContent = selected;
                render();
            };
            if (!entry.directory) button.ondblclick = () => choose(entry.path);
            list.append(button);
        }
    }
    async function load(value) {
        const request = ++sequence;
        current = null;
        selected = "";
        path.value = value;
        message.textContent = "Loading…";
        up.disabled = true;
        use.disabled = true;
        render();
        try {
            const data = await fetchJSON("/lightx2v/files", value, base,
                {directories_only: directory ? "1" : "0", file_type: lora ? "lora" : "config"});
            if (!dialog.isConnected || request !== sequence) return;
            current = data;
            selected = data.selected;
            path.value = data.directory;
            up.disabled = data.parent === data.directory;
            use.disabled = !selected;
            message.textContent = selected || hint;
            if (data.truncated) message.textContent += "\nLarge directory: enter a more specific path to see other entries.";
            render();
        } catch (error) {
            if (request === sequence) message.textContent = error.message;
        }
    }
    async function choose(value) {
        use.disabled = true;
        try {
            const data = await fetchJSON(route, value);
            if (!dialog.isConnected) return;
            onSelect(data[resultKey]);
            dialog.close();
        } catch (error) {
            message.textContent = error.message;
            use.disabled = false;
        }
    }
    go.onclick = () => load(path.value);
    path.onkeydown = event => { if (event.key === "Enter") { event.preventDefault(); load(path.value); } };
    up.onclick = () => load(current.parent);
    filter.oninput = render;
    use.onclick = () => choose(selected);
    dialog.showModal();
    load(initialPath || base);
}
