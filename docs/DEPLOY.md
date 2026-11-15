# Deploying the demo (static site)

The demo is a plain static site in `web/dist`: HTML, JS, one ONNX model (6.1 MB) and the onnxruntime-web
WebAssembly runtime (11 MB), all self-hosted with relative paths. No server code, no environment variables,
and no external requests (the page's Content-Security-Policy blocks them). Any static host works. Vercel is not used.

## 1. Build it (once, or after any change)

From the repository root, with the Python environment and the datasets in place:

```bash
.venv/Scripts/python -m src.export_onnx   # only if the model changed; writes web/public/*
cd web
npm install                              # installs onnxruntime-web 1.20.1 (pinned)
npm test                                 # JS vs Python parity: must print "23/23 passed"
npm run build                            # writes web/dist
npm run serve                            # http://127.0.0.1:8080/  (Ctrl+C to stop)
```

`web/public/` (model, model card, self-test image) is committed, so `npm install && npm run build` also works
on a machine without the datasets. Only `npm test` needs the git-ignored parity fixtures.

Optional end-to-end browser check, with `npm run serve` running on port 8765 (`PORT=8765 npm run serve`):
`node test/browser-selftest.mjs` (Edge) or `node test/browser-selftest.mjs "C:/Program Files/Google/Chrome/Application/chrome.exe"`.
It must print `SELFTEST PASS`. You can also open `http://127.0.0.1:8080/?selftest=1` and read the box at the bottom.

## 2a. Cloudflare Pages (direct upload, simplest)

1. Sign in at https://dash.cloudflare.com, then go to **Workers & Pages**, **Create**, **Pages**, **Upload assets**.
2. Project name: e.g. `phish-screenshot-demo`. Click **Create project**.
3. Drag the **contents** of `web/dist` (not the folder itself) into the upload box. Make sure `index.html` is at the top level,
   and include the `ort/` folder and the `_headers` file. Click **Deploy site**.
4. Open the `https://<project>.pages.dev` URL it shows, and check `https://<project>.pages.dev/?selftest=1` says `SELFTEST PASS`.

Command-line alternative (needs a free Cloudflare account; `wrangler login` opens a browser once):

```bash
cd web
npx wrangler@3 pages deploy dist --project-name phish-screenshot-demo
```

To update later, rebuild and upload `web/dist` again (dashboard: **Create new deployment**).

## 2b. GitHub Pages

GitHub Pages on a **private** repository needs a paid GitHub plan. On a free account, either make the repo public
(the code, results and model contain no raw data, see README) or use Cloudflare instead.

Option 1, publish `web/dist` to a `gh-pages` branch:

```bash
cd web
npm run build
npx gh-pages@6 -d dist --dotfiles    # pushes dist/ (including .nojekyll) to the gh-pages branch
```

Then on GitHub go to **Settings**, **Pages**, **Build and deployment**, set **Source: Deploy from a branch**,
**Branch: gh-pages / (root)**, and save. After a minute the site is at
`https://<your-username>.github.io/<repo-name>/`. Relative paths mean the sub-folder works without changes.
Check `.../?selftest=1`.

Option 2, without the gh-pages tool: copy the contents of `web/dist` into a `docs/`-style folder on a branch and
point Pages at it the same way. Remember `.nojekyll`.

## Notes

- The page works offline once loaded, and nothing is sent anywhere: the image is decoded and scored in the tab.
- Threads: the model runs single-threaded WebAssembly, because static hosts don't send the cross-origin-isolation headers
  that multi-threading needs. Scoring one screenshot still takes well under a second on a laptop.
- Do **not** add phishing screenshots as examples. If you want an example gallery, capture your own screenshots of
  well-known *legitimate* pages (e.g. a Wikipedia article, your university's homepage, a news site front page, and,
  to show the main weakness, a real bank or webmail **login** page, which the model will likely score as "phishing"),
  put them in `web/public/examples/`, and add them to `build.mjs` and the page.
