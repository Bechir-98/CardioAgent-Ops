import json
import os
import uuid
import urllib.error
import urllib.request

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

# Where the backend services live. Compose sets these to the service names;
# the defaults work when everything runs on the host.
RAG_URL = os.getenv("RAG_URL", "http://localhost:8003").rstrip("/")
AUDIO_URL = os.getenv("AUDIO_URL", "http://localhost:8001").rstrip("/")
VECTOR_URL = os.getenv("VECTOR_URL", "http://localhost:8004").rstrip("/")


def _post_json(url: str, payload: dict, timeout: int = 300) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _multipart_body(field: str, filename: str, content: bytes, content_type: str) -> tuple[bytes, str]:
    boundary = "----CardioAgent-" + uuid.uuid4().hex
    safe_filename = filename.replace("\\", "\\\\").replace('"', '\\"')
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{safe_filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def _probe(url: str, timeout: int = 3) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            ok = response.status == 200
            return {"ready": ok, "detail": "ok" if ok else f"HTTP {response.status}"}
    except urllib.error.HTTPError as exc:
        return {"ready": False, "detail": f"HTTP {exc.code}"}
    except urllib.error.URLError as exc:
        return {"ready": False, "detail": str(exc.reason)}
    except TimeoutError:
        return {"ready": False, "detail": "timeout"}


html_content = """
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>CardioAgent</title>
        <style>
            :root {
                --ink: #000000;
                --muted: #555555;
                --line: #cccccc;
                --border: #888888;
                --accent: #000000;
                --ok: #1a5200;
                --bad: #8a1f1f;
                --warn: #7a4a00;
                --radius: 0px;
            }
            * { box-sizing: border-box; }
            html, body { margin: 0; padding: 0; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
                background: #ffffff;
                color: var(--ink);
                line-height: 1.5;
                min-height: 100vh;
            }
            a { color: var(--ink); text-decoration: underline; }
            button { font: inherit; cursor: pointer; }

            .topbar {
                border-bottom: 1px solid var(--line);
                background: #ffffff;
                padding: 14px 24px;
                position: sticky;
                top: 0;
                z-index: 10;
            }
            .brand { display: flex; align-items: baseline; gap: 10px; }
            .brand h1 { margin: 0; font-size: 1.3rem; font-weight: 700; letter-spacing: -0.02em; }
            .brand p { margin: 0; font-size: 0.72rem; color: var(--muted); letter-spacing: 0.06em; text-transform: uppercase; }
            .brand span { font-size: 0.72rem; color: var(--muted); }

            .status {
                margin-left: auto; display: flex; align-items: center; gap: 16px; font-size: 0.72rem;
            }
            .status-item { display: flex; align-items: center; gap: 5px; }
            .status-dot { width: 5px; height: 5px; border-radius: 1px; background: var(--muted); }
            .status-dot.ok { background: var(--ok); }
            .status-dot.bad { background: var(--bad); }
            .status-label { font-size: 0.66rem; font-weight: 700; letter-spacing: 0.04em; color: var(--muted); text-transform: uppercase; }

            .wrap { max-width: 700px; margin: 0 auto; padding: 30px 20px 60px; }

            .kicker {
                font-size: 0.68rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
                color: var(--muted); margin: 0 0 6px;
            }
            h2 { font-size: 1.05rem; font-weight: 700; margin: 0 0 10px; }
            h3 { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin: 16px 0 8px; }
            .hint { color: var(--muted); font-size: 0.82rem; margin: 0 0 14px; }

            .section {
                border-top: 1px solid var(--line);
                padding-top: 16px;
                margin-bottom: 26px;
            }

            .input-field {
                width: 100%; border: 1px solid var(--border); border-radius: var(--radius);
                padding: 9px 11px; font-size: 0.95rem; background: #ffffff; color: var(--ink);
            }
            .input-field:focus { outline: none; border-color: var(--ink); }

            .controls { display: flex; gap: 10px; align-items: center; margin-top: 12px; flex-wrap: wrap; }
            .controls label { font-size: 0.78rem; color: var(--muted); font-weight: 600; }

            .segmented { display: flex; gap: 4px; border: 1px solid var(--border); border-radius: var(--radius); background: #f5f5f5; }
            .seg-btn {
                border: 0; background: transparent; border-radius: 0; padding: 5px 9px; cursor: pointer;
                color: var(--muted); font-weight: 600; font-size: 0.78rem;
            }
            .seg-btn.active { background: #ffffff; color: var(--ink); border: 1px solid var(--ink); }

            .btn {
                border: 1px solid var(--border); border-radius: var(--radius);
                padding: 8px 14px; font-size: 0.88rem; font-weight: 600;
                color: var(--ink); background: #ffffff; cursor: pointer;
            }
            .btn:hover { border-color: var(--ink); background: #f0f0f0; }
            .btn:disabled { opacity: 0.55; cursor: default; }
            .btn.primary { border-color: var(--ink); color: #ffffff; background: var(--ink); }
            .btn.primary:hover { background: #1a1a1a; border-color: #1a1a1a; }

            .examples { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
            .examples button {
                border: 1px solid var(--border); border-radius: var(--radius);
                padding: 4px 9px; font-size: 0.75rem; color: var(--ink); background: #ffffff;
                cursor: pointer; line-height: 1.4;
            }
            .examples button:hover { border-color: var(--ink); background: #f0f0f0; }

            .error {
                border: 1px solid var(--bad); border-radius: var(--radius);
                background: #fdf3f3; color: var(--bad); font-size: 0.82rem;
                padding: 9px 11px; margin-top: 12px; display: none;
            }
            .error.visible { display: block; }

            .answer { display: none; margin-top: 16px; padding: 14px 14px 10px 14px; background: #f7f7f7; border: 1px solid var(--line); border-left: 3px solid var(--ink); }
            .answer.visible { display: block; }
            .answer-head {
                display: flex; align-items: baseline; gap: 10px; margin-bottom: 8px;
                border-bottom: 1px solid var(--line); padding-bottom: 6px;
            }
            .status-badge {
                font-size: 0.66rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
                padding: 2px 7px; border: 1px solid;
            }
            .status-ok { color: var(--ok); border-color: var(--ok); background: #f4fbf3; }
            .status-warn { color: var(--warn); border-color: var(--warn); background: #fdf8e8; }
            .status-bad { color: var(--bad); border-color: var(--bad); background: #fdf3f3; }

            .answer-text {
                padding-left: 12px; margin-bottom: 12px;
                white-space: pre-wrap; font-size: 1rem; color: var(--ink); line-height: 1.6;
            }
            .answer-text.refusal { color: #4a3200; }
            .cite { font-weight: 800; color: var(--ink); background: #e8e8e8; padding: 0 4px; }

            .sources { border-top: 1px solid var(--line); padding-top: 10px; }
            .sources h3 { margin-top: 0; }
            .source {
                border: 1px solid var(--border); border-radius: var(--radius);
                padding: 9px 11px; margin-bottom: 7px; background: #ffffff;
            }
            .source.cited { border-color: var(--ink); background: #fffbf0; box-shadow: none; }
            .source-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 5px; }
            .source-rank { font-weight: 800; color: var(--ink); font-size: 0.8rem; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace; }
            .source-meta { color: var(--muted); font-size: 0.72rem; font-weight: 500; }
            .source-text { font-size: 0.82rem; color: var(--ink); white-space: pre-wrap; }

            .dropzone {
                border: 1px solid var(--border); border-radius: var(--radius);
                padding: 18px; background: #f5f5f5; text-align: center;
                cursor: pointer; transition: border-color 0.15s;
            }
            .dropzone:hover { border-color: var(--ink); }
            .dropzone div:first-child { font-size: 1.6rem; color: var(--muted); margin-bottom: 7px; }            .dropzone #fileName { font-weight: 700; color: var(--ink); font-size: 0.88rem; margin-bottom: 5px; }
            .dropzone small { color: var(--muted); font-size: 0.72rem; display: block; margin-top: 4px; }

            .audio-label {
                border: 1px solid var(--border); border-radius: var(--radius);
                padding: 10px 11px; background: #f5f5f5; text-align: center;
                font-weight: 700; font-size: 0.88rem; color: var(--ink);
            }
            .audio-label.abnormal { border-color: var(--bad); color: var(--bad); background: #fdf3f3; }
            .audio-label.normal { border-color: var(--ok); color: var(--ok); background: #f4fbf3; }

            .metric-table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.82rem; }
            .metric-table td { border-bottom: 1px solid var(--line); padding: 7px 0; vertical-align: top; }
            .metric-table td:first-child { color: var(--muted); width: 40%; font-weight: 500; }
            .metric-table td:last-child { text-align: right; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace; font-weight: 700; color: var(--ink); }

            footer {
                border-top: 1px solid var(--line); margin-top: 30px; padding-top: 14px;
                color: var(--muted); font-size: 0.7rem; text-align: center;
            }
            footer span { border: 1px solid var(--border); border-radius: 2px; padding: 2px 7px; }

            @media (max-width: 560px) {
                .topbar { padding: 12px 16px; }
                .status { display: none; }
                .wrap { padding: 20px 16px 48px; }
            }
        </style>
    </head>
    <body>
        <header class="topbar">
            <div class="brand">
                <h1>CardioAgent</h1>
                <p>Guideline Q&amp;A · PCG analysis</p>
                <span>On-premise</span>
            </div>
            <nav class="status" aria-label="Service status">
                <div class="status-item"><span class="status-dot" id="dot-llm"></span><span class="status-label" id="lbl-llm">—</span></div>
                <div class="status-item"><span class="status-dot" id="dot-vec"></span><span class="status-label" id="lbl-vec">—</span></div>
                <div class="status-item"><span class="status-dot" id="dot-audio"></span><span class="status-label" id="lbl-audio">—</span></div>
            </nav>
        </header>

        <div class="wrap">
            <section class="section">
                <p class="kicker">Guidelines</p>
                <h2>Ask a question</h2>
                <p class="hint">Answers come only from the indexed ESC corpus. Each answer cites the excerpts it uses.</p>
                <form id="askForm">
                    <textarea id="question" class="input-field" rows="4" placeholder="e.g. When is surgery indicated in severe aortic regurgitation?" required aria-label="Clinical question"></textarea>
                    <div class="controls">
                        <label for="k">Excerpts</label>
                        <div class="segmented" role="radiogroup" aria-label="Number of excerpts">
                            <button type="button" class="seg-btn" data-k="2">2</button>
                            <button type="button" class="seg-btn active" data-k="4">4</button>
                            <button type="button" class="seg-btn" data-k="6">6</button>
                            <button type="button" class="seg-btn" data-k="8">8</button>
                        </div>
                        <button type="submit" id="askButton" class="btn primary">Ask</button>
                        <button type="button" id="clearButton" class="btn">Clear</button>
                    </div>
                    <div class="examples" aria-label="Example questions">
                        <button type="button" class="examples__chip" data-q="What are the echocardiographic criteria for severe aortic stenosis?">Criteria for severe AS</button>
                        <button type="button" class="examples__chip" data-q="When is aortic valve replacement indicated in asymptomatic severe aortic stenosis?">Asymptomatic severe AS</button>
                        <button type="button" class="examples__chip" data-q="What antithrombotic therapy is recommended after TAVI?">Antithrombotic after TAVI</button>
                        <button type="button" class="examples__chip" data-q="How often should heart failure treatment be uptitrated?">Uptitration interval</button>
                        <button type="button" class="examples__chip" data-q="How is cardiac amyloidosis diagnosed?">Amyloidosis diagnosis</button>
                    </div>
                </form>

                <div id="askError" class="error"></div>

                <div id="answerBox" class="answer" aria-live="polite">
                    <div class="answer-head">
                        <span id="answerBadge" class="status-badge status-ok">Grounded</span>
                        <span id="answerCosine" class="status-badge status-ok">cosine —</span>
                    </div>
                    <div id="answerText" class="answer-text"></div>
                    <button type="button" id="copyButton" class="btn" style="margin-bottom:10px">Copy answer</button>
                    <div class="sources" id="sources">
                        <h3>Retrieved excerpts</h3>
                        <div id="sourceList"></div>
                    </div>
                </div>
            </section>

            <section class="section">
                <p class="kicker">Audio</p>
                <h2>Analyse a PCG recording</h2>
                <p class="hint">Upload a heart-sound WAV file for a normal / abnormal prediction from the trained ONNX model.</p>
                <form id="uploadForm">
                    <label for="pcgFile" id="dropZone" class="dropzone">
                        <div>PCG</div>
                        <div id="fileName">Click to choose a WAV file</div>
                        <small>Accepts .wav only</small>
                        <input type="file" name="file" id="pcgFile" accept=".wav" required hidden>
                    </label>
                    <div class="controls" style="margin-top:14px">
                        <input type="text" name="prompt" placeholder="Optional note…" class="input-field" style="max-width:260px; flex:1">
                        <button type="submit" class="btn primary">Analyse</button>
                    </div>
                </form>

                <div id="audioResult" style="display:none; margin-top:16px;">
                    <span id="audioLabel" class="audio-label normal">—</span>
                    <table class="metric-table">
                        <tbody>
                            <tr><td>Confidence</td><td id="confidencePct">—</td></tr>
                            <tr><td>P(abnormal)</td><td id="confidencePAbnormal">—</td></tr>
                            <tr><td>Model</td><td id="confidenceModel">—</td></tr>
                            <tr><td>File</td><td id="confidenceFile">—</td></tr>
                        </tbody>
                    </table>
                    <div id="audioError" class="error"></div>
                </div>
            </section>

            <footer>
                <span>Research prototype — not a medical device. Verify critical decisions with clinical judgment.</span>
            </footer>
        </div>

        <script>
            const askButton = document.getElementById('askButton');
            const questionInput = document.getElementById('question');
            const answerBox = document.getElementById('answerBox');
            const answerBadge = document.getElementById('answerBadge');
            const answerCosine = document.getElementById('answerCosine');
            const answerText = document.getElementById('answerText');
            const sourceList = document.getElementById('sourceList');
            const askError = document.getElementById('askError');
            const askForm = document.getElementById('askForm');
            let currentK = 4;

            async function refreshStatus() {
                try {
                    const r = await fetch('/status');
                    const s = await r.json();
                    setDot('dot-llm', s.llm.ready, 'lbl-llm');
                    setDot('dot-vec', s.vector.ready, 'lbl-vec');
                    setDot('dot-audio', s.audio.ready, 'lbl-audio');
                } catch (e) {
                    ['dot-llm', 'dot-vec', 'dot-audio'].forEach((id, i) => {
                        setDot(id, false, ['lbl-llm','lbl-vec','lbl-audio'][i]);
                    });
                }
            }
            refreshStatus();
            setInterval(refreshStatus, 30000);

            function setDot(id, state, labelId) {
                const el = document.getElementById(id);
                const lbl = labelId ? document.getElementById(labelId) : null;
                el.classList.remove('ok', 'bad');
                if (state === true) {
                    el.classList.add('ok');
                    if (lbl) { lbl.textContent = 'ok'; lbl.style.color = 'var(--ok)'; }
                } else if (state === false) {
                    el.classList.add('bad');
                    if (lbl) { lbl.textContent = 'down'; lbl.style.color = 'var(--bad)'; }
                } else {
                    if (lbl) { lbl.textContent = '—'; lbl.style.color = 'var(--muted)'; }
                }
            }

            document.querySelectorAll('.seg-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    document.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    currentK = parseInt(btn.dataset.k, 10);
                });
            });

            document.querySelectorAll('.examples__chip').forEach(chip => {
                chip.addEventListener('click', () => {
                    questionInput.value = chip.dataset.q;
                    questionInput.focus();
                });
            });

            document.getElementById('clearButton').addEventListener('click', () => {
                questionInput.value = '';
                answerBox.className = 'answer';
                askError.className = 'error';
                questionInput.focus();
            });

            document.getElementById('copyButton').addEventListener('click', async () => {
                try {
                    await navigator.clipboard.writeText(answerText.textContent || '');
                    const btn = document.getElementById('copyButton');
                    const original = btn.textContent;
                    btn.textContent = 'Copied';
                    btn.classList.add('primary');
                    setTimeout(() => { btn.textContent = original; btn.classList.remove('primary'); }, 1200);
                } catch (e) { /* clipboard unavailable */ }
            });

            function renderAnswerText(text) {
                while (answerText.firstChild) answerText.removeChild(answerText.firstChild);
                const parts = String(text).split(/(\[\d{1,2}\])/g);
                parts.forEach(part => {
                    if (/^\\[\\d{1,2}\\]$/.test(part)) {
                        const span = document.createElement('span');
                        span.className = 'cite';
                        span.textContent = part;
                        answerText.appendChild(span);
                    } else if (part) answerText.appendChild(document.createTextNode(part));
                });
            }

            async function ask() {
                const question = questionInput.value.trim();
                if (!question) return;

                askError.className = 'error';
                answerBox.className = 'answer';
                askButton.disabled = true;
                askButton.textContent = 'Thinking…';

                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question, k: currentK })
                    });
                    const data = await response.json();
                    if (!response.ok) throw new Error(data.detail || data.error || ('HTTP ' + response.status));

                    answerBadge.className = 'status-badge';
                    const refused = (data.answer || '').startsWith("I don't know");
                    if (refused) {
                        answerBadge.classList.add('status-warn');
                        answerBadge.textContent = 'Not covered by corpus';
                    } else if (data.grounded) {
                        answerBadge.classList.add('status-ok');
                        answerBadge.textContent = 'Grounded';
                    } else {
                        answerBadge.classList.add('status-bad');
                        answerBadge.textContent = 'No valid citation';
                    }

                    const cos = Number(data.best_score || 0).toFixed(3);
                    answerCosine.className = 'status-badge status-ok';
                    answerCosine.textContent = 'cosine ' + cos;

                    renderAnswerText(data.answer);
                    answerText.className = 'answer-text' + (refused ? ' refusal' : '');

                    while (sourceList.firstChild) sourceList.removeChild(sourceList.firstChild);
                    const cited = new Set(data.citations || []);
                    (data.sources || []).forEach(s => {
                        const box = document.createElement('div');
                        box.className = 'source' + (cited.has(s.rank) ? ' cited' : '');
                        if (cited.has(s.rank)) { box.style.borderLeft = '3px solid var(--accent)'; }

                        const head = document.createElement('div');
                        head.className = 'source-head';
                        const rank = document.createElement('span'); rank.className = 'source-rank'; rank.textContent = '[' + s.rank + ']';
                        const meta = document.createElement('span'); meta.className = 'source-meta';
                        meta.textContent = ((s.section || '').slice(0, 60)) + ' · cosine ' + Number(s.score || 0).toFixed(3);
                        head.appendChild(rank); head.appendChild(meta);

                        const body = document.createElement('div'); body.className = 'source-text'; body.textContent = s.text;
                        box.appendChild(head); box.appendChild(body); sourceList.appendChild(box);
                    });

                    answerBox.className = 'answer visible';
                } catch (err) {
                    askError.textContent = 'Error: ' + err.message;
                    askError.className = 'error visible';
                } finally {
                    askButton.disabled = false;
                    askButton.textContent = 'Ask';
                }
            }

            askForm.addEventListener('submit', e => { e.preventDefault(); ask(); });
            questionInput.addEventListener('keydown', e => {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); ask(); }
            });

            // Audio
            const fileInput = document.getElementById('pcgFile');
            const fileName = document.getElementById('fileName');
            const dropZone = document.getElementById('dropZone');
            fileInput.addEventListener('change', () => {
                fileName.textContent = fileInput.files.length ? fileInput.files[0].name : 'Click to choose a WAV file';
            });
            ['dragenter', 'dragover'].forEach(evt => dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.style.borderColor = 'var(--accent)'; }));
            ['dragleave', 'drop'].forEach(evt => dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.style.borderColor = 'var(--border)'; }));
            dropZone.addEventListener('drop', e => {
                const files = e.dataTransfer.files;
                if (files.length) { fileInput.files = files; fileName.textContent = files[0].name; }
            });

            const audioResult = document.getElementById('audioResult');
            const audioLabel = document.getElementById('audioLabel');
            const audioError = document.getElementById('audioError');
            const confidencePct = document.getElementById('confidencePct');
            const confidencePAbnormal = document.getElementById('confidencePAbnormal');
            const confidenceModel = document.getElementById('confidenceModel');
            const confidenceFile = document.getElementById('confidenceFile');

            document.getElementById('uploadForm').onsubmit = async (e) => {
                e.preventDefault();
                audioError.className = 'error';
                audioError.textContent = '';
                audioResult.style.display = 'block';
                audioLabel.textContent = 'Analysing…';
                audioLabel.className = 'audio-label';
                confidencePct.textContent = '—';
                confidencePAbnormal.textContent = '—';
                confidenceModel.textContent = '—';
                confidenceFile.textContent = '—';
                try {
                    const response = await fetch('/analyze_pcg', { method: 'POST', body: new FormData(e.target) });
                    const data = await response.json();
                    if (!response.ok) throw new Error(data.detail || data.error || ('HTTP ' + response.status));
                    const label = data.label === 'normal' ? 'normal' : 'abnormal';
                    const confidence = typeof data.confidence === 'number' ? data.confidence : null;
                    const pAbn = typeof data.p_abnormal === 'number' ? data.p_abnormal : null;
                    audioLabel.textContent = label.toUpperCase();
                    audioLabel.className = 'audio-label ' + label;
                    confidencePct.textContent = confidence === null ? 'n/a' : (confidence * 100).toFixed(1) + '%';
                    confidencePAbnormal.textContent = pAbn === null ? 'n/a' : (pAbn * 100).toFixed(1) + '%';
                    confidenceModel.textContent = data.model || '—';
                    confidenceFile.textContent = data.filename || '—';
                } catch (err) {
                    audioError.textContent = 'Error: ' + err.message;
                    audioError.className = 'error visible';
                    audioLabel.textContent = 'FAILED';
                    audioLabel.className = 'audio-label bad';
                }
            };
        </script>
    </body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return html_content


@app.get("/health")
async def health():
    return {"service": "api-gateway", "status": "ok", "rag_url": RAG_URL}


@app.get("/status")
def status():
    """Short health probe for the UI status bar (short timeouts only)."""
    checks = {
        "llm": _probe(f"{RAG_URL}/health"),
        "vector": _probe(f"{VECTOR_URL}/api/v2/heartbeat"),
        "audio": _probe(f"{AUDIO_URL}/health"),
    }
    ready = all(c["ready"] for c in checks.values())
    return {"overall": "ok" if ready else "degraded", **checks}


@app.post("/ask")
def ask(payload: dict):
    """Proxy a question to the RAG service and return answer + sources.

    Runs as a sync endpoint so FastAPI executes the blocking HTTP calls in its
    threadpool instead of stalling the event loop.
    """
    question = (payload or {}).get("question", "").strip()
    if len(question) < 3:
        raise HTTPException(status_code=400, detail="question is required (at least 3 characters)")

    k = int(payload.get("k", 4) or 4)
    k = max(1, min(k, 10))

    try:
        answer = _post_json(f"{RAG_URL}/answer", {"question": question, "k": k})
        hits = _post_json(f"{RAG_URL}/query", {"question": question, "k": k}).get("hits", [])
    except urllib.error.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"RAG service returned HTTP {exc.code}: {exc.read().decode(errors='replace')[:200]}",
        )
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=503, detail=f"RAG service unreachable at {RAG_URL}: {exc.reason}")
    except TimeoutError:
        raise HTTPException(status_code=504, detail="RAG service timed out")

    return {
        "question": question,
        "answer": answer.get("answer"),
        "grounded": answer.get("grounded"),
        "citations": answer.get("citations") or [],
        "best_score": answer.get("best_score"),
        "sources": [
            {
                "rank": hit["rank"],
                "source": hit["source"],
                "section": hit["section"],
                "score": hit["score"],
                "text": hit["text"][:500],
            }
            for hit in hits
        ],
    }


@app.post("/analyze_pcg")
def analyze_pcg(file: UploadFile = File(...), prompt: str = Form(None)):
    """Forward the WAV to the audio engine and mirror its response.

    Runs as a sync endpoint so FastAPI executes the blocking HTTP call in its
    threadpool instead of stalling the event loop.
    """
    raw = file.file.read()
    content_type = file.content_type or "audio/wav"
    body, content_type_header = _multipart_body("file", file.filename or "upload.wav", raw, content_type)
    request = urllib.request.Request(
        f"{AUDIO_URL}/predict",
        data=body,
        headers={"Content-Type": content_type_header},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Audio engine returned HTTP {exc.code}: {exc.read().decode(errors='replace')[:200]}",
        )
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=503, detail=f"Audio engine unreachable at {AUDIO_URL}: {exc.reason}")
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Audio engine timed out")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {exc}")

    label = result.get("label") or "unknown"
    probability = result.get("probability")
    confidence = max(probability, 1 - probability) if isinstance(probability, (int, float)) else None
    pct = f"{confidence:.1%}" if isinstance(confidence, (int, float)) else "n/a"
    p_abnormal = f"{probability:.1%}" if isinstance(probability, (int, float)) else "n/a"
    message = (
        f"Prediction: {label}\n"
        f"Confidence: {pct}\n"
        f"P(abnormal): {p_abnormal}\n"
        f"Model: {result.get('model')}\n"
        f"File: {file.filename}\n"
        f"Note: {prompt or 'No prompt provided'}"
    )
    return {
        "status": "success",
        "message": message,
        "label": label,
        "prediction": result.get("prediction"),
        "probability": probability,
        "confidence": confidence,
        "p_abnormal": probability,
        "model": result.get("model"),
        "filename": file.filename,
        "note": prompt or "",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
