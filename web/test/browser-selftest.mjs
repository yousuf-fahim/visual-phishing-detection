// End-to-end browser check: the real browser image decoder + preprocess.js + onnxruntime-web (wasm).
//
// Run from web/ while dist/ is being served:
//   python -m http.server 8765 --bind 127.0.0.1 --directory dist      (in another terminal)
//   node test/browser-selftest.mjs [path-to-chrome-or-edge]
//
// Starts a headless Chromium browser with the DevTools protocol, opens index.html?selftest=1, and waits
// until the page prints "SELFTEST PASS/FAIL". The page compares its probability for the bundled
// synthetic image with Python's value (model_card.json). No extra npm packages needed (Node >= 22).
import { spawn } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const browser = process.argv[2] ||
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
const url = process.env.SELFTEST_URL || "http://127.0.0.1:8765/index.html?selftest=1";
const port = 9300 + Math.floor(Math.random() * 600); // fresh port per run
const profile = mkdtempSync(path.join(tmpdir(), "selftest-"));
const proc = spawn(browser, ["--headless=new", "--disable-gpu", "--no-first-run", "--disable-extensions",
  `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, "about:blank"], { stdio: "ignore" });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let target;
for (let i = 0; i < 50 && !target; i++) {
  await sleep(200);
  try {
    target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((t) => t.type === "page");
  } catch { /* browser still starting */ }
}
if (!target) { console.error("browser did not start"); proc.kill(); process.exit(2); }

const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r));
let nextId = 1;
const pending = new Map();
ws.addEventListener("message", (e) => {
  const msg = JSON.parse(e.data);
  if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg.result); pending.delete(msg.id); }
  if (msg.method === "Runtime.exceptionThrown") console.error("page exception:", msg.params.exceptionDetails.text,
    msg.params.exceptionDetails.exception?.description?.slice(0, 300));
  if (msg.method === "Log.entryAdded" && msg.params.entry.level === "error") console.error("console:", msg.params.entry.text.slice(0, 300));
  if (msg.method === "Runtime.consoleAPICalled" && msg.params.type === "error")
    console.error("console.error:", msg.params.args.map((a) => a.value ?? a.description).join(" ").slice(0, 300));
});
const send = (method, params = {}) => new Promise((resolve) => {
  const id = nextId++;
  pending.set(id, resolve);
  ws.send(JSON.stringify({ id, method, params }));
});

await send("Runtime.enable");
await send("Log.enable");
await send("Page.navigate", { url });
let text = "";
for (let i = 0; i < 120 && !text.startsWith("SELFTEST"); i++) {
  await sleep(500);
  const r = await send("Runtime.evaluate", {
    expression: "(document.getElementById('selftest')||{}).textContent || ''", returnByValue: true });
  text = r?.result?.value || "";
  if (!text) {
    const s = await send("Runtime.evaluate", { expression: "document.getElementById('status')?.textContent", returnByValue: true });
    if (s?.result?.value?.startsWith("Could not")) { text = `ERROR ${s.result.value}`; break; }
  }
}
console.log(text || "TIMEOUT: no self-test result after 60 s");
ws.close();
proc.kill();
process.exit(text.startsWith("SELFTEST PASS") ? 0 : 1);
