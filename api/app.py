"""FastAPI backend with markdown table rendering, static image serving, and async embedding."""

from pathlib import Path
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from agent.orchestrator import AgentOrchestrator
from embeddings.embedder import Embedder
from embeddings.reranker import BGEReranker
from ingestion.router import IngestionRouter
from storage.faiss_store import FAISSStore
from storage.sqlite_store import SQLiteStore

app = FastAPI(title="VeritasAI RAG Pipeline")

# Static file serving for images & figures
uploads_dir = Path("data/uploads")
uploads_dir.mkdir(parents=True, exist_ok=True)
figures_dir = Path("data/uploads/figures")
figures_dir.mkdir(parents=True, exist_ok=True)
app.mount("/figures", StaticFiles(directory=str(figures_dir)), name="figures")
app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")

sqlite_store = SQLiteStore()
faiss_store = FAISSStore()
embedder = Embedder()
reranker = BGEReranker()
orchestrator = AgentOrchestrator(sqlite_store, faiss_store, embedder, reranker)
router = IngestionRouter()

indexing_state = {"is_indexing": False, "last_error": None}


def async_embed_and_index(child_chunks: list) -> None:
    """Computes embeddings and adds vectors to FAISS in the background."""
    if not child_chunks:
        return
    indexing_state["is_indexing"] = True
    indexing_state["last_error"] = None
    try:
        texts = [c.breadcrumb_text for c in child_chunks]
        ids = [c.child_id for c in child_chunks]
        vectors = embedder.embed_documents(texts)
        faiss_store.add(ids, vectors)
    except Exception as e:
        indexing_state["last_error"] = str(e)
    finally:
        indexing_state["is_indexing"] = False


class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"


HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>VeritasAI - Research Mate</title>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <style>
    * { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; display: flex; flex-direction: column; height: 100vh; }
    .container { max-width: 950px; margin: 0 auto; width: 100%; display: flex; flex-direction: column; flex: 1; }
    header { border-bottom: 1px solid #334155; padding-bottom: 12px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; }
    h1 { margin: 0; font-size: 22px; color: #38bdf8; }
    .status { font-size: 13px; color: #94a3b8; display: flex; align-items: center; gap: 12px; }
    .upload-card { background: #1e293b; border: 1px dashed #475569; border-radius: 8px; padding: 14px; margin-bottom: 16px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
    .upload-card input { color: #94a3b8; font-size: 13px; }
    .btn { background: #0284c7; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 500; font-size: 13px; }
    .btn:hover { background: #0369a1; }
    #chat-box { flex: 1; overflow-y: auto; background: #1e293b; border-radius: 8px; padding: 18px; margin-bottom: 16px; display: flex; flex-direction: column; gap: 14px; }
    .msg { padding: 12px 18px; border-radius: 8px; max-width: 90%; font-size: 14px; line-height: 1.6; }
    .user-msg { background: #0369a1; align-self: flex-end; color: white; }
    .bot-msg { background: #1e293b; border: 1px solid #334155; align-self: flex-start; color: #f1f5f9; border-left: 4px solid #38bdf8; }
    
    /* Tables styling */
    table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; background: #0f172a; border-radius: 6px; overflow: hidden; }
    th, td { border: 1px solid #334155; padding: 8px 12px; text-align: left; }
    th { background: #1e293b; color: #38bdf8; font-weight: 600; }
    tr:nth-child(even) { background: #131e33; }
    
    /* Code & Images styling */
    pre { background: #0f172a; padding: 12px; border-radius: 6px; overflow-x: auto; border: 1px solid #334155; font-size: 13px; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #38bdf8; }
    img { max-width: 100%; border-radius: 6px; border: 1px solid #475569; margin: 10px 0; display: block; }
    
    .sources { margin-top: 10px; font-size: 12px; color: #94a3b8; background: #0f172a; padding: 10px; border-radius: 6px; border: 1px solid #334155; }
    .input-row { display: flex; gap: 10px; }
    .input-row input { flex: 1; background: #1e293b; border: 1px solid #475569; color: white; padding: 12px 14px; border-radius: 6px; font-size: 14px; outline: none; }
    .input-row input:focus { border-color: #38bdf8; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>VeritasAI — Research Mate</h1>
      <div class="status">
        <span>Session: <input id="session-id" value="default" onchange="loadMemory()" style="background:#1e293b;color:#38bdf8;border:1px solid #475569;border-radius:4px;padding:4px 8px;width:110px;font-size:12px;"></span>
        <span id="memory-status" style="color:#a7f3d0;font-size:12px;">🧠 Memory: 0 turns</span>
        <button class="btn" style="background:#475569;padding:4px 10px;font-size:12px;" onclick="clearMemory()">Clear Memory</button>
        <a href="/docs" target="_blank" style="color: #38bdf8; text-decoration: none;">API Docs (/docs)</a>
      </div>
    </header>

    <div class="upload-card">
      <input type="file" id="file-input" multiple>
      <button class="btn" onclick="uploadFiles()">Ingest Documents</button>
      <button class="btn" style="background: #0d9488;" onclick="refreshUploads()">🔄 Refresh Uploads DB</button>
      <button class="btn" style="background: #dc2626;" onclick="resetDatabase()">Clear / Reset DB</button>
      <span id="upload-status" style="font-size: 13px; color: #38bdf8;"></span>
    </div>

    <div id="chat-box">
      <div class="msg bot-msg">Hello! I am your Research Assistant with 3-Tier Week-Long Memory. Ask any question about your documents, or continue a multi-turn discussion across days.</div>
    </div>

    <div class="input-row">
      <input type="text" id="query-input" placeholder="Type your research question..." onkeydown="if(event.key==='Enter') sendQuery()">
      <button class="btn" onclick="sendQuery()">Ask</button>
    </div>
  </div>

  <script>
    mermaid.initialize({ startOnLoad: false, theme: 'dark' });

    function getSessionId() {
      return (document.getElementById('session-id').value || 'default').trim();
    }

    async function loadMemory(restoreChat = true) {
      const sid = getSessionId();
      try {
        const res = await fetch(`/memory?session_id=${encodeURIComponent(sid)}`);
        const mem = await res.json();
        document.getElementById('memory-status').innerText =
          `🧠 Memory: ${mem.turns.length} turns | ${mem.facts.length} facts | ${mem.episodes_count} episodes`;
        if (restoreChat && mem.turns.length > 0) {
          const chat = document.getElementById('chat-box');
          chat.innerHTML = '';
          mem.turns.forEach(t => {
            if (t.role === 'user') {
              chat.innerHTML += `<div class="msg user-msg">${t.content}</div>`;
            } else {
              chat.innerHTML += `<div class="msg bot-msg">${marked.parse(t.content)}</div>`;
            }
          });
          chat.scrollTop = chat.scrollHeight;
        }
      } catch (e) {}
    }

    async function clearMemory() {
      const sid = getSessionId();
      await fetch(`/memory/clear?session_id=${encodeURIComponent(sid)}`, { method: 'POST' });
      document.getElementById('chat-box').innerHTML =
        '<div class="msg bot-msg">Conversation memory cleared for this session.</div>';
      await loadMemory(false);
    }

    async function pollStatus() {
      const status = document.getElementById('upload-status');
      try {
        const res = await fetch('/status');
        const st = await res.json();
        if (st.last_error) {
          status.innerText = '❌ Embedding error: ' + st.last_error;
          return;
        }
        if (st.is_indexing) {
          status.innerText = `⏳ Indexing embeddings in background... (${st.indexed_chunks} chunks in FAISS so far)`;
          setTimeout(pollStatus, 1000);
        } else {
          status.innerText = `✅ Ready! (${st.indexed_chunks} chunks indexed in FAISS)`;
        }
      } catch (e) {}
    }

    window.addEventListener('DOMContentLoaded', () => {
      pollStatus();
      loadMemory(true);
    });

    async function uploadFiles() {
      const input = document.getElementById('file-input');
      const files = input.files;
      if (!files.length) return alert('Please choose one or more files');
      const status = document.getElementById('upload-status');
      status.innerText = `⏳ Parsing ${files.length} document(s)...`;
      const fd = new FormData();
      for (const file of files) {
        fd.append('files', file);
      }
      try {
        const res = await fetch('/ingest', { method: 'POST', body: fd });
        const data = await res.json();
        if (res.ok) {
          if (data.errors && data.errors.length) {
            status.innerText = `⚠️ Error parsing: ${data.errors.join('; ')}`;
          } else {
            status.innerText = `⏳ Parsed ${data.total_files} file(s). Computing embeddings on MPS GPU...`;
            setTimeout(pollStatus, 500);
          }
        } else {
          status.innerText = 'Error: ' + (data.detail || 'Upload failed');
        }
      } catch (e) {
        status.innerText = 'Upload failed';
      }
    }

    async function refreshUploads() {
      const status = document.getElementById('upload-status');
      status.innerText = '⏳ Refreshing DB from data/uploads/ with VLM image descriptions...';
      try {
        const res = await fetch('/refresh_uploads', { method: 'POST' });
        const data = await res.json();
        status.innerText = `⏳ Re-ingesting ${data.queued_files} file(s) with Qwen VLM in background...`;
        setTimeout(pollStatus, 1000);
      } catch (e) {
        status.innerText = 'Refresh failed';
      }
    }

    async function resetDatabase() {
      if (!confirm('Are you sure you want to clear all documents from the database?')) return;
      const status = document.getElementById('upload-status');
      status.innerText = 'Clearing database...';
      try {
        const res = await fetch('/reset', { method: 'POST' });
        const data = await res.json();
        status.innerText = '✅ Database reset! (0 chunks in FAISS)';
        const chat = document.getElementById('chat-box');
        chat.innerHTML = '<div class="msg bot-msg">Database cleared. You can upload fresh documents now.</div>';
      } catch (e) {
        status.innerText = 'Reset failed';
      }
    }

    async function sendQuery() {
      const input = document.getElementById('query-input');
      const q = input.value.trim();
      if (!q) return;
      input.value = '';

      const chat = document.getElementById('chat-box');
      chat.innerHTML += `<div class="msg user-msg">${q}</div>`;
      chat.scrollTop = chat.scrollHeight;

      const botPlaceholder = document.createElement('div');
      botPlaceholder.className = 'msg bot-msg';
      botPlaceholder.innerText = 'Thinking and retrieving...';
      chat.appendChild(botPlaceholder);
      chat.scrollTop = chat.scrollHeight;

      try {
        const res = await fetch('/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: q, session_id: getSessionId() })
        });
        const data = await res.json();
        const answerHtml = marked.parse(data.answer);
        let figuresHtml = '';
        if (data.sources && data.sources.length) {
          const seenImgs = new Set();
          data.sources.forEach(s => {
            const match = (s.content || '').match(/Image:\s*([^\s\]\)]+\.(?:png|jpg|jpeg|webp))/i)
                       || (s.content || '').match(/\/figures\/([^\s\]\)]+\.(?:png|jpg|jpeg|webp))/i);
            if (match && match[1] && !answerHtml.includes(match[1]) && !seenImgs.has(match[1])) {
              seenImgs.add(match[1]);
              figuresHtml += `<div style="margin:12px 0;"><div style="font-size:12px;color:#38bdf8;margin-bottom:4px;">📊 <b>Retrieved Figure:</b> ${s.doc} — ${s.section} (Page ${s.page})</div><img src="/figures/${match[1]}" alt="${s.section}"></div>`;
            }
          });
        }
        let sourcesHtml = '';
        if (data.sources && data.sources.length) {
          sourcesHtml = '<div class="sources"><b>Sources:</b><br>' + 
            data.sources.map(s => `• [${s.doc} | Page ${s.page}] ${s.section}`).join('<br>') + '</div>';
        }
        let latencyHtml = '';
        if (data.latency) {
          const l = data.latency;
          latencyHtml = `<div style="margin-top:8px; font-size:11px; color:#38bdf8;">⚡ <b>MPS GPU Latency:</b> Total ${l.total_ms}ms (Memory: ${l.memory_recall_ms || 0}ms | Embed: ${l.query_embedding_ms}ms | FAISS: ${l.faiss_search_ms}ms | SQLite: ${l.sqlite_lookup_ms}ms | Rerank: ${l.reranking_ms}ms | Groq: ${l.llm_generation_ms}ms)</div>`;
        }
        // Render rich markdown (tables, lists, code, bold, images) via marked.js + retrieved figures
        botPlaceholder.innerHTML = answerHtml + figuresHtml + latencyHtml + sourcesHtml;
        // Convert any ```mermaid blocks into rendered Mermaid SVG diagrams
        botPlaceholder.querySelectorAll('pre code.language-mermaid').forEach((block) => {
          const div = document.createElement('div');
          div.className = 'mermaid';
          div.textContent = block.textContent;
          block.parentElement.replaceWith(div);
        });
        await mermaid.run({ nodes: botPlaceholder.querySelectorAll('.mermaid') });
        await loadMemory(false);
      } catch (e) {
        botPlaceholder.innerText = 'Failed to answer query.';
      }
      chat.scrollTop = chat.scrollHeight;
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Serves the interactive web UI."""
    return HTML_UI


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/status")
async def status() -> dict:
    """Returns live background indexing state and total FAISS vector count."""
    return {
        "is_indexing": indexing_state["is_indexing"],
        "indexed_chunks": len(faiss_store.ids),
        "last_error": indexing_state["last_error"],
    }


@app.get("/memory")
async def get_memory(session_id: str = "default") -> dict:
    """Returns turns, semantic facts, and episodic memory count for a session."""
    turns = orchestrator.memory.get_all_turns(session_id)
    facts = orchestrator.memory.get_facts(session_id)
    episodes_count = len(orchestrator.memory.episode_faiss.ids)
    return {
        "session_id": session_id,
        "turns": turns,
        "facts": facts,
        "episodes_count": episodes_count,
    }


@app.post("/memory/clear")
def clear_memory(session_id: str = "default") -> dict:
    """Clears conversation memory for a given session."""
    orchestrator.memory.clear(session_id)
    return {"status": "success", "session_id": session_id}


@app.post("/ingest")
async def ingest_files(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
) -> dict:
    """Ingests multiple documents, saves to SQLite immediately, and embeds in background."""
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    errors = []
    all_children_to_embed = []

    for file in files:
        file_path = upload_dir / file.filename
        with file_path.open("wb") as f:
            f.write(await file.read())

        try:
            result = router.ingest_file(file_path)
        except Exception as e:
            errors.append(f"{file.filename}: {e}")
            continue

        sqlite_store.save(result.parent_chunks, result.child_chunks, result.extracted_tables)
        all_children_to_embed.extend(result.child_chunks)

        summary.append(
            {
                "filename": result.filename,
                "parent_chunks": len(result.parent_chunks),
                "child_chunks": len(result.child_chunks),
            }
        )

    # Fast return: kick off embedding and FAISS indexing in the background
    if all_children_to_embed:
        indexing_state["is_indexing"] = True
        background_tasks.add_task(async_embed_and_index, all_children_to_embed)

    return {
        "status": "success" if not errors else "partial",
        "message": "Chunks stored in SQLite. Vector embeddings are indexing in the background.",
        "files_ingested": summary,
        "total_files": len(summary),
        "errors": errors,
    }


def _refresh_all_uploads_worker(file_paths: list[Path]) -> None:
    """Clears DB/FAISS and freshly ingests all files in data/uploads with rich VLM figure descriptions."""
    indexing_state["is_indexing"] = True
    indexing_state["last_error"] = None
    try:
        sqlite_store.clear()
        faiss_store.clear()
        for fp in file_paths:
            try:
                result = router.ingest_file(fp)
                sqlite_store.save(result.parent_chunks, result.child_chunks, result.extracted_tables)
                if result.child_chunks:
                    texts = [c.breadcrumb_text for c in result.child_chunks]
                    ids = [c.child_id for c in result.child_chunks]
                    vectors = embedder.embed_documents(texts)
                    faiss_store.add(ids, vectors)
            except Exception as e:
                indexing_state["last_error"] = f"{fp.name}: {e}"
    finally:
        indexing_state["is_indexing"] = False


@app.post("/refresh_uploads")
def refresh_uploads(background_tasks: BackgroundTasks) -> dict:
    """Triggers a full fresh ingestion of all files in data/uploads with VLM image descriptions."""
    upload_dir = Path("data/uploads")
    supported = (
        IngestionRouter.PDF_EXTS
        | IngestionRouter.TABULAR_EXTS
        | IngestionRouter.MARKDOWN_EXTS
        | IngestionRouter.IMAGE_EXTS
    )
    files = sorted(f for f in upload_dir.iterdir() if f.is_file() and f.suffix.lower() in supported)
    indexing_state["is_indexing"] = True
    background_tasks.add_task(_refresh_all_uploads_worker, files)
    return {"status": "started", "queued_files": len(files), "files": [f.name for f in files]}


@app.post("/reset")
def reset_stores() -> dict:
    """Clears all SQLite chunks/tables, FAISS vector embeddings, and memory."""
    sqlite_store.clear()
    faiss_store.clear()
    orchestrator.memory.clear()
    return {"status": "success", "message": "Database and vector index cleared successfully."}


@app.post("/query")
def answer_query(req: QueryRequest) -> dict:
    """Answers a user query using the agent orchestrator with session memory."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    return orchestrator.run(req.query, session_id=req.session_id)
