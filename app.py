"""
PedigreeGraph web UI.

  pip install flask google-genai
  $env:GEMINI_API_KEY = "your-key"        (PowerShell)
  python app.py                           then open http://localhost:5000

Left: the family pedigree read from FalkorDB (squares = male, circles = female,
dark fill = affected). Right: chat with the agent. When the agent answers, the
family path it used is highlighted on the tree.
"""
import os
import threading
import time
from collections import OrderedDict, defaultdict, deque

from flask import Flask, jsonify, request

import agent_tools as at

app = Flask(__name__)
_lock = threading.Lock()
_chats = OrderedDict()  # one chat session per visitor (keyed by a random id from the browser)
CALLS = []

# Abuse protection for a public deployment. Defaults are effectively off for local use;
# on the host set e.g. ASK_PER_HOUR=15 and ASK_PER_DAY=200.
ASK_PER_HOUR = int(os.environ.get("ASK_PER_HOUR", "1000"))
ASK_PER_DAY = int(os.environ.get("ASK_PER_DAY", "10000"))
_hits = defaultdict(deque)
_day_hits = deque()

# record every tool call the agent makes (name, args, output)
_orig_run_tool = at.run_tool


def _recording_run_tool(name, args):
    out = _orig_run_tool(name, args)
    CALLS.append({"tool": name, "args": args, "output": out})
    return out


at.run_tool = _recording_run_tool


def get_chat(cid):
    if cid in _chats:
        _chats.move_to_end(cid)
        return _chats[cid]
    import agent_gemini as ag

    chat = ag.make_chat()
    _chats[cid] = chat
    while len(_chats) > 50:
        _chats.popitem(last=False)
    return chat


def client_ip():
    xff = request.headers.get("X-Forwarded-For", "")
    return (xff.split(",")[0].strip() if xff else request.remote_addr) or "?"


def rate_limited():
    """Return a message if this visitor (or the whole demo) is over the limit, else record the hit."""
    now = time.time()
    q = _hits[client_ip()]
    while q and now - q[0] > 3600:
        q.popleft()
    while _day_hits and now - _day_hits[0] > 86400:
        _day_hits.popleft()
    hint = "You can still click any person on the tree for an instant report."
    if len(q) >= ASK_PER_HOUR:
        return f"Limit reached: {ASK_PER_HOUR} questions per hour per visitor. {hint}"
    if len(_day_hits) >= ASK_PER_DAY:
        return f"The daily question limit for this demo was reached. {hint}"
    q.append(now)
    _day_hits.append(now)
    return None


def pct(x):
    return "0%" if x == 0 else (f"{x * 100:.1f}%" if x < 0.01 else f"{round(x * 100)}%")


def format_report(r):
    """Plain-text version of health_report (no AI involved)."""
    if "error" in r:
        return r["error"]
    lines = [
        f"{r['person']}  |  {r.get('condition') or 'condition'}  ({r.get('inheritance_pattern') or 'pattern unclear'})",
        f"Currently affected: {'yes' if r['currently_affected'] else 'no'}",
    ]
    yr = r.get("your_risk")
    if yr:
        lines.append(f"Chance: {pct(yr['value'])} ({yr['meaning']})")
    fh = r["family_history"]
    lines.append("Affected close relatives: " + (", ".join(fh["first_degree_affected"]) or "none recorded"))
    if fh["second_degree_affected"]:
        lines.append("Affected more distant relatives: " + ", ".join(fh["second_degree_affected"]))
    if r.get("path_to_nearest_affected_relative"):
        lines.append("Family path to nearest affected relative: " + " -> ".join(r["path_to_nearest_affected_relative"]))
    for p in r["partner_planning"]:
        s = f"With partner {p['partner']}: {pct(p['chance_child_affected'])} chance a child is affected"
        if p["shared_ancestors"]:
            s += " (partners share ancestors: " + ", ".join(p["shared_ancestors"]) + ")"
        lines.append(s)
    lines.append("Next steps:")
    lines += ["- " + x for x in r["next_steps"]]
    lines.append(r["disclaimer"])
    return "\n".join(lines)


def collect_highlight(calls):
    """Pull the family paths and key people out of the tool outputs."""
    nodes, paths = set(), []
    path_keys = ("path_to_nearest_affected_relative", "path_to_affected_relative")
    list_keys = ("common_ancestors", "shared_ancestors")
    id_keys = ("partner", "partner_1", "partner_2")

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in path_keys and isinstance(v, list) and v:
                    paths.append(v)
                    nodes.update(v)
                elif k in list_keys and isinstance(v, list):
                    nodes.update(v)
                elif k in id_keys and isinstance(v, str):
                    nodes.add(v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    for c in calls:
        if c["tool"] == "health_report" and isinstance(c["args"].get("person"), str):
            nodes.add(c["args"]["person"])
        walk(c["output"])
    return {"nodes": sorted(nodes), "paths": paths}


@app.get("/api/families")
def api_families():
    try:
        rows = at.graph().query("MATCH (p:Person) RETURN DISTINCT p.family, p.condition").result_set
    except Exception as e:
        return jsonify({"error": f"cannot reach FalkorDB ({type(e).__name__}: {e})"}), 500
    return jsonify(sorted(({"id": r[0], "condition": r[1]} for r in rows), key=lambda x: x["id"]))


@app.get("/api/graph")
def api_graph():
    fam = request.args.get("family", "")
    g = at.graph()
    prm = {"fam": fam}
    nodes = g.query(
        "MATCH (p:Person {family:$fam}) RETURN p.id, p.sex, p.generation, p.affected", prm
    ).result_set
    parent = g.query(
        "MATCH (a:Person {family:$fam})-[:PARENT_OF]->(b:Person) RETURN a.id, b.id", prm
    ).result_set
    partner = g.query(
        "MATCH (a:Person {family:$fam})-[:PARTNER_OF]->(b:Person) RETURN a.id, b.id", prm
    ).result_set
    return jsonify(
        {
            "nodes": [
                {"id": r[0], "sex": r[1], "generation": r[2], "affected": bool(r[3])} for r in nodes
            ],
            "edges": [{"from": r[0], "to": r[1], "type": "PARENT_OF"} for r in parent]
            + [{"from": r[0], "to": r[1], "type": "PARTNER_OF"} for r in partner],
        }
    )


@app.get("/healthz")
def healthz():
    return "ok"


@app.get("/api/report")
def api_report():
    person = request.args.get("person", "")
    try:
        rep = at.health_report(person)
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    if "error" in rep:
        return jsonify({"error": rep["error"]}), 404
    call = {"tool": "health_report", "args": {"person": person}, "output": rep}
    return jsonify({"text": format_report(rep), "highlight": collect_highlight([call])})


@app.post("/api/ask")
def api_ask():
    body = request.get_json(force=True) or {}
    question = body.get("question", "").strip()
    cid = str(body.get("cid", "default"))[:64]
    if not question:
        return jsonify({"error": "Empty question"}), 400
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        return jsonify({"error": "Chat is unavailable: this server has no AI key. Click a person on the tree for their report."}), 503
    limited = rate_limited()
    if limited:
        return jsonify({"error": limited}), 429
    with _lock:
        CALLS.clear()
        try:
            resp = get_chat(cid).send_message(question)
        except Exception as e:
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
        calls = list(CALLS)
    return jsonify(
        {
            "answer": resp.text or "(no text returned)",
            "calls": [{"tool": c["tool"], "args": c["args"]} for c in calls],
            "highlight": collect_highlight(calls),
        }
    )


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>PedigreeGraph</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  body { margin:0; font-family: system-ui, sans-serif; background:#f8fafc; color:#0f172a; }
  header { padding:12px 20px; background:#0f172a; color:#fff; }
  header small { color:#94a3b8; margin-left:10px; }
  html, body { height:100%; overflow:hidden; }
  main { display:flex; height:calc(100vh - 57px); overflow:hidden; }
  #left { flex:3; display:flex; flex-direction:column; border-right:1px solid #e2e8f0; min-width:0; min-height:0; }
  #bar { padding:8px 12px; background:#fff; border-bottom:1px solid #e2e8f0; font-size:14px; }
  #graph { height:calc(100vh - 135px); position:relative; background:#fff; color:#b91c1c; padding:0; font-size:14px; }
  #legend { padding:6px 12px; font-size:12px; color:#475569; background:#fff; border-top:1px solid #e2e8f0; }
  #right { flex:2; display:flex; flex-direction:column; min-width:340px; min-height:0; }
  #log { flex:1; min-height:0; overflow:auto; padding:12px; }
  .msg { margin-bottom:12px; padding:10px; border-radius:8px; white-space:pre-wrap; font-size:14px; line-height:1.45; }
  .you { background:#e0e7ff; }
  .bot { background:#fff; border:1px solid #e2e8f0; }
  .tools { font-size:12px; color:#475569; margin-bottom:6px; font-family:monospace; }
  #form { display:flex; gap:8px; padding:10px; border-top:1px solid #e2e8f0; background:#fff; }
  #q { flex:1; padding:8px; font-size:14px; }
  button { padding:8px 14px; background:#0f172a; color:#fff; border:0; border-radius:6px; cursor:pointer; }
  .sugg { font-size:12px; padding:4px 8px; background:#e2e8f0; color:#0f172a; margin:2px; }
</style></head>
<body>
<header><b>PedigreeGraph</b><small>family health tracker on FalkorDB - synthetic demo data, not medical advice</small></header>
<main>
  <section id="left">
    <div id="bar">Family: <select id="fam"></select></div>
    <div id="graph"></div>
    <div id="legend">Square = male, circle = female, dark = affected. Red = path used. Click a person for an instant report.</div>
  </section>
  <section id="right">
    <div id="log"></div>
    <div style="padding:0 10px 6px">
      <button class="sugg" onclick="useQ(this)">What is the risk picture for FAM1_P14?</button>
      <button class="sugg" onclick="useQ(this)">Which family is dominant and which couples should worry?</button>
      <button class="sugg" onclick="useQ(this)">Who should be tested first in FAM3?</button>
    </div>
    <div id="form"><input id="q" placeholder="Ask about a person or family..."><button onclick="ask()">Ask</button></div>
  </section>
</main>
<script>
let network, nodesDS, edgesDS;
const CID = (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : String(Math.random()).slice(2);

async function loadFamiliesInner() {
  const resp = await fetch('/api/families');
  const fams = await resp.json();
  if (!resp.ok) throw new Error(fams.error || ('server error ' + resp.status));
  if (!fams.length) throw new Error('no families found in the graph. Run: python generate_families.py');
  const sel = document.getElementById('fam');
  sel.innerHTML = '';
  fams.forEach(f => { const o = document.createElement('option'); o.value = f.id; o.textContent = f.id + ' - ' + (f.condition || ''); sel.appendChild(o); });
  sel.onchange = () => loadGraph(sel.value);
  if (fams.length) loadGraph(fams[0].id);
}

async function loadFamilies() {
  try {
    await loadFamiliesInner();
  } catch (e) {
    document.getElementById('graph').textContent = 'Could not load families: ' + e.message;
  }
}

async function drawGraph(fam) {
  document.getElementById('fam').value = fam;
  const g = await (await fetch('/api/graph?family=' + encodeURIComponent(fam))).json();
  nodesDS = new vis.DataSet(g.nodes.map(n => ({
    id: n.id, label: n.id.split('_')[1], level: n.generation,
    shape: n.sex === 'M' ? 'box' : 'circle',
    color: { background: n.affected ? '#334155' : '#ffffff', border: '#334155' },
    font: { color: n.affected ? '#ffffff' : '#0f172a' },
    title: n.id + (n.affected ? ' (affected)' : ''), borderWidth: 1
  })));
  edgesDS = new vis.DataSet(g.edges.map((e, i) => ({
    id: 'e' + i, from: e.from, to: e.to,
    arrows: e.type === 'PARENT_OF' ? 'to' : '', dashes: e.type === 'PARTNER_OF',
    color: { color: '#94a3b8' }, width: 1
  })));
  network = new vis.Network(document.getElementById('graph'), { nodes: nodesDS, edges: edgesDS }, {
    layout: { hierarchical: { enabled: true, direction: 'UD', levelSeparation: 90, nodeSpacing: 70 } },
    physics: false, interaction: { hover: true }
  });
  network.on('click', p => { if (p.nodes && p.nodes.length) showReport(p.nodes[0]); });
}

async function showReport(id) {
  try {
    const r = await fetch('/api/report?person=' + encodeURIComponent(id));
    const d = await r.json();
    if (d.error) { addMsg(d.error, 'bot'); return; }
    addMsg(d.text, 'bot', 'report for ' + id + ' (graph queries + exact genetics, no AI call)');
    applyHighlight(d.highlight);
  } catch (e) { addMsg('Error: ' + e, 'bot'); }
}

async function loadGraph(fam) {
  const box = document.getElementById('graph');
  try {
    if (typeof vis === 'undefined') throw new Error('the drawing library did not load (check your internet connection or ad blocker)');
    box.textContent = '';
    await drawGraph(fam);
  } catch (e) {
    box.textContent = 'Could not draw the tree: ' + e.message;
  }
}

function clearHighlight() {
  if (!nodesDS) return;
  nodesDS.update(nodesDS.getIds().map(id => ({ id: id, borderWidth: 1, shadow: false })));
  edgesDS.update(edgesDS.getIds().map(id => ({ id: id, width: 1, color: { color: '#94a3b8' } })));
}

function applyHighlight(h) {
  clearHighlight();
  const nodes = h.nodes || [];
  if (nodes.length) {
    const fam = nodes[0].split('_')[0];
    if (document.getElementById('fam').value !== fam) {
      loadGraph(fam).then(() => applyHighlight(h));
      return;
    }
  }
  nodesDS.update(nodes.filter(id => nodesDS.get(id)).map(id => ({
    id: id, borderWidth: 4, shadow: { enabled: true, color: '#ef4444', size: 18, x: 0, y: 0 }
  })));
  const pairs = [];
  (h.paths || []).forEach(p => { for (let i = 0; i < p.length - 1; i++) pairs.push([p[i], p[i + 1]]); });
  const upd = [];
  edgesDS.get().forEach(e => {
    if (pairs.some(([a, b]) => (e.from === a && e.to === b) || (e.from === b && e.to === a)))
      upd.push({ id: e.id, width: 4, color: { color: '#ef4444' } });
  });
  edgesDS.update(upd);
}

function addMsg(text, cls, tools) {
  const log = document.getElementById('log');
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  if (tools) { const t = document.createElement('div'); t.className = 'tools'; t.textContent = tools; d.appendChild(t); }
  d.appendChild(document.createTextNode(text));
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
}

function useQ(btn) { document.getElementById('q').value = btn.textContent; ask(); }

async function ask() {
  const input = document.getElementById('q');
  const q = input.value.trim();
  if (!q) return;
  input.value = '';
  addMsg(q, 'you');
  addMsg('Thinking...', 'bot');
  const log = document.getElementById('log');
  try {
    const r = await fetch('/api/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: q, cid: CID }) });
    const d = await r.json();
    log.removeChild(log.lastChild);
    if (d.error) { addMsg(d.error, 'bot'); return; }
    const tools = 'tools: ' + d.calls.map(c => c.tool + '(' + Object.values(c.args).join(', ') + ')').join('  ->  ');
    addMsg(d.answer, 'bot', tools);
    applyHighlight(d.highlight);
  } catch (e) {
    log.removeChild(log.lastChild);
    addMsg('Error: ' + e, 'bot');
  }
}
document.getElementById('q').addEventListener('keydown', e => { if (e.key === 'Enter') ask(); });
loadFamilies();
</script>
</body></html>"""


@app.get("/")
def index():
    return PAGE


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")), debug=False)
