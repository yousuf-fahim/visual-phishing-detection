// Minimal static file server for testing web/dist locally (no dependencies).
// Run from web/:  npm run serve       then open http://127.0.0.1:8080/
// (Python's http.server sends large files slowly on Windows, which made Chrome abort the
// 10 MB WebAssembly download, so this streams files with the right MIME types instead.)
import { createReadStream, statSync } from "node:fs";
import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "dist");
const port = Number(process.env.PORT || 8080);
const TYPES = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
  ".css": "text/css", ".json": "application/json", ".wasm": "application/wasm",
  ".onnx": "application/octet-stream", ".png": "image/png",
};

createServer((req, res) => {
  const urlPath = decodeURIComponent(new URL(req.url, "http://x").pathname);
  let file = path.normalize(path.join(root, urlPath));
  if (!file.startsWith(root)) { res.writeHead(403).end(); return; } // no ../ escapes
  try {
    if (statSync(file).isDirectory()) file = path.join(file, "index.html");
    const size = statSync(file).size;
    res.writeHead(200, { "Content-Type": TYPES[path.extname(file)] || "application/octet-stream",
                         "Content-Length": size });
    createReadStream(file).pipe(res);
  } catch {
    res.writeHead(404).end("not found");
  }
}).listen(port, "127.0.0.1", () => console.log(`Serving web/dist at http://127.0.0.1:${port}/`));
