"""
Own-family mode: a visitor enters a small family (nicknames only) and gets the same
report as the demo families (Cypher family history, shortest family path,
consanguinity, exact genetics).

Privacy by design:
  - nicknames only (letters, numbers, spaces; max 12 characters), no free text
  - the family is NOT stored: for each report a temporary graph with a random
    name is created in FalkorDB, queried, and deleted again
  - nothing from the request is logged
  - at most MAX_PEOPLE people, because the exact-genetics engine grows fast
"""
import json
import re
import uuid

import agent_tools as at
import pedigree_tools as pt

MAX_PEOPLE = 20
MIN_PEOPLE = 3
NAME_RE = re.compile(r"^[A-Za-z0-9 ]{1,12}$")
CONDITIONS = [
    "Beta-thalassemia",
    "Sickle cell disease",
    "Cystic fibrosis",
    "Tay-Sachs disease",
    "Familial hypercholesterolemia",
]


class InputError(ValueError):
    """The visitor's input is not usable; the message is safe to show."""


def _ref(v):
    v = "" if v is None else str(v).strip().lower()
    return v or None


def validate(payload):
    """Check the payload and turn it into graph rows. Raises InputError with a friendly message."""
    if not isinstance(payload, dict):
        raise InputError("Bad request.")
    cond = payload.get("condition")
    if cond not in CONDITIONS:
        raise InputError("Pick a condition from the list.")
    rows = payload.get("people")
    if not isinstance(rows, list) or not (MIN_PEOPLE <= len(rows) <= MAX_PEOPLE):
        raise InputError(f"Enter between {MIN_PEOPLE} and {MAX_PEOPLE} people.")

    info = {}
    for r in rows:
        if not isinstance(r, dict):
            raise InputError("Bad person entry.")
        name = str(r.get("name", "")).strip()
        if not NAME_RE.match(name):
            raise InputError("Names must be 1 to 12 letters, numbers or spaces. Use nicknames, not real names.")
        key = name.lower()
        if key in info:
            raise InputError(f'The name "{name}" is used twice.')
        if r.get("sex") not in ("M", "F"):
            raise InputError(f'Choose a sex for "{name}".')
        info[key] = {
            "name": name, "sex": r["sex"], "affected": bool(r.get("affected")),
            "father": _ref(r.get("father")), "mother": _ref(r.get("mother")), "partner": _ref(r.get("partner")),
        }

    for key, p in info.items():
        for role, want in (("father", "M"), ("mother", "F"), ("partner", None)):
            ref = p[role]
            if ref is None:
                continue
            if ref not in info:
                raise InputError(f'"{p["name"]}": the {role} is not in the list.')
            if ref == key:
                raise InputError(f'"{p["name"]}" cannot be their own {role}.')
            if want and info[ref]["sex"] != want:
                raise InputError(f'"{info[ref]["name"]}" must be {"male" if want == "M" else "female"} to be a {role}.')
            if role == "partner" and info[ref]["sex"] == p["sex"]:
                raise InputError("Partners must be one male and one female (the model follows father and mother).")
        if (p["father"] is None) != (p["mother"] is None):
            raise InputError(f'"{p["name"]}": give both parents or neither.')

    parents = {k: [p["father"], p["mother"]] for k, p in info.items() if p["father"]}

    depth = {}

    def dfs(k, stack):
        if k in depth:
            return depth[k]
        if k in stack:
            raise InputError("Someone is their own ancestor. Check the parents.")
        if k not in parents:
            depth[k] = 0
            return 0
        depth[k] = 1 + max(dfs(x, stack | {k}) for x in parents[k])
        return depth[k]

    for k in info:
        dfs(k, frozenset())

    # couples: parents of a child, plus explicit partners
    couples = set()
    for k, p in info.items():
        if p["father"]:
            couples.add((p["father"], p["mother"]))
        if p["partner"]:
            a, b = (k, p["partner"]) if p["sex"] == "M" else (p["partner"], k)
            couples.add((a, b))
    for m, f in couples:
        for x, y in ((m, f), (f, m)):
            if y in parents.get(x, []):
                raise InputError("A person cannot be the partner of their own parent or child.")

    # show founders who marry in on the same row as their partner
    gen = dict(depth)
    for m, f in couples:
        if m not in parents:
            gen[m] = max(gen[m], depth[f])
        if f not in parents:
            gen[f] = max(gen[f], depth[m])

    order = sorted(info, key=lambda k: (gen[k], k))
    ids = {k: f"CUSTOM_P{i + 1}" for i, k in enumerate(order)}
    names = {v: info[k]["name"] for k, v in ids.items()}

    people = [
        {"id": ids[k], "family": "CUSTOM", "condition": cond, "sex": info[k]["sex"],
         "generation": gen[k], "affected": info[k]["affected"]}
        for k in order
    ]
    parent_of = []
    for k, p in info.items():
        if p["father"]:
            parent_of.append({"a": ids[p["father"]], "b": ids[k]})
            parent_of.append({"a": ids[p["mother"]], "b": ids[k]})
    partners = [{"a": ids[m], "b": ids[f]} for m, f in sorted(couples)]
    return {"people": people, "parent_of": parent_of, "partners": partners, "ids": ids, "names": names}


def _rename(obj, names):
    """Swap internal ids (CUSTOM_P3) for the visitor's nicknames in a report."""
    if isinstance(obj, dict):
        return {k: _rename(v, names) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rename(x, names) for x in obj]
    if isinstance(obj, str):
        return re.sub(r"CUSTOM_P\d+", lambda m: names.get(m.group(0), m.group(0)), obj)
    return obj


def analyze(payload, person_name):
    """Build a temporary graph, produce the report for one person, delete the graph."""
    model = validate(payload)
    key = str(person_name).strip().lower()
    ids_by_lower = {n.lower(): i for i, n in model["names"].items()}
    if key not in ids_by_lower:
        raise InputError("Unknown person.")
    person_id = ids_by_lower[key]

    import config

    db = config.connect_db()
    g = db.select_graph("custom_" + uuid.uuid4().hex[:16])
    try:
        g.query(
            "UNWIND $rows AS r CREATE (:Person {id: r.id, family: r.family, condition: r.condition, sex: r.sex, "
            "generation: r.generation, affected: r.affected, tested: false})",
            {"rows": model["people"]},
        )
        if model["parent_of"]:
            g.query(
                "UNWIND $rows AS r MATCH (a:Person {id: r.a}), (b:Person {id: r.b}) CREATE (a)-[:PARENT_OF]->(b)",
                {"rows": model["parent_of"]},
            )
        if model["partners"]:
            g.query(
                "UNWIND $rows AS r MATCH (a:Person {id: r.a}), (b:Person {id: r.b}) CREATE (a)-[:PARTNER_OF]->(b)",
                {"rows": model["partners"]},
            )
        ped = pt.load_family(g, "CUSTOM")
        report = at.health_report(person_id, g=g, ped=ped)
    finally:
        try:
            g.delete()  # nothing is kept
        except Exception:
            pass
    report = _rename(report, model["names"])
    if isinstance(report, dict) and "error" not in report and not any(p["affected"] for p in model["people"]):
        # nobody affected: no pattern can be inferred, and the numbers are only population averages
        report["inheritance_pattern"] = None
        if report.get("your_risk"):
            report["your_risk"]["meaning"] = (
                "average chance of being a carrier if the condition were recessive (population baseline; "
                "nobody in the family was marked as affected)"
            )
        report["next_steps"] = [
            "No affected relatives were entered, so this shows population averages only. It does not mean there is no risk.",
            "Carriers usually have no symptoms, so family history can miss them. If you are planning a child, "
            "ask a doctor or genetic counselor about carrier screening, especially if this condition is common in your community.",
        ]
    if isinstance(report, dict) and "error" not in report:
        report["disclaimer"] = (
            "Based only on the family you entered, with simple Mendelian assumptions. "
            "This is family-history risk awareness, not a diagnosis. Real concerns belong with a doctor or genetic counselor."
        )
    return report


EXAMPLE = {
    "condition": "Beta-thalassemia",
    "people": [
        {"name": "Asha", "sex": "F"}, {"name": "Ravi", "sex": "M"},
        {"name": "Meera", "sex": "F", "father": "Ravi", "mother": "Asha"},
        {"name": "Kiran", "sex": "M", "father": "Ravi", "mother": "Asha"},
        {"name": "Dev", "sex": "M", "partner": "Meera"},
        {"name": "Sita", "sex": "F", "partner": "Kiran"},
        {"name": "Anil", "sex": "M", "father": "Dev", "mother": "Meera"},
        {"name": "Tara", "sex": "F", "father": "Kiran", "mother": "Sita", "partner": "Anil"},
        {"name": "Neel", "sex": "M", "father": "Anil", "mother": "Tara", "affected": True},
    ],
}

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PedigreeGraph - your own family</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, sans-serif; background:#f8fafc; color:#0f172a; }
  header { padding:12px 20px; background:#0f172a; color:#fff; display:flex; align-items:center; gap:14px; flex-wrap:wrap; }
  header small { color:#94a3b8; }
  header a { color:#7dd3fc; margin-left:auto; text-decoration:none; font-size:14px; }
  #warn { background:#fef3c7; color:#78350f; padding:8px 20px; font-size:13px; border-bottom:1px solid #fcd34d; }
  main { display:flex; gap:0; min-height:calc(100vh - 100px); }
  #left { flex:5; padding:14px 16px; border-right:1px solid #e2e8f0; overflow:auto; min-width:0; }
  #right { flex:5; display:flex; flex-direction:column; min-width:0; }
  h3 { margin:0 0 8px; font-size:15px; }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  th { text-align:left; color:#475569; font-weight:600; padding:4px 3px; }
  td { padding:3px; }
  input[type=text], select { width:100%; padding:5px; font-size:13px; border:1px solid #cbd5e1; border-radius:5px; background:#fff; }
  button { padding:7px 12px; background:#0f172a; color:#fff; border:0; border-radius:6px; cursor:pointer; font-size:13px; }
  button.light { background:#e2e8f0; color:#0f172a; }
  button.x { padding:3px 8px; background:#fee2e2; color:#991b1b; }
  .row { display:flex; gap:8px; margin:10px 0; flex-wrap:wrap; align-items:center; }
  #graph { height:340px; background:#fff; border-bottom:1px solid #e2e8f0; }
  #report { flex:1; padding:12px 16px; white-space:pre-wrap; font-size:14px; line-height:1.5; overflow:auto; }
  #msg { color:#b91c1c; font-size:13px; min-height:18px; }
  .hint { font-size:12px; color:#64748b; }
</style></head>
<body>
<header><b>PedigreeGraph</b><small>try your own family (use nicknames, not real names)</small><a href="/">Back to the demo families</a></header>
<div id="warn">Use nicknames, not real names. This app does not save your family: each report builds a temporary graph in FalkorDB and deletes it right away. It is a demo project, not medical advice.</div>
<main>
  <section id="left">
    <h3>1. Describe a family</h3>
    <div class="row">
      <label for="cond">Condition:</label>
      <select id="cond" style="width:auto"></select>
      <button class="light" onclick="loadExample()">Load an example</button>
      <button class="light" onclick="clearAll()">Clear</button>
    </div>
    <table>
      <thead><tr><th>Nickname</th><th>Sex</th><th>Affected</th><th>Father</th><th>Mother</th><th>Partner</th><th></th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
    <div class="row"><button onclick="addRow()">+ Add person</button><span class="hint" id="count"></span></div>
    <div class="hint">Give both parents or neither. Partners who share a child are linked automatically. Up to __MAX__ people. Tick "Affected" for anyone in the family known to have the condition, even a distant relative. If nobody is ticked, you only see population averages. You do not need to add a child: add yourself and your partner as a couple.</div>
    <div id="msg"></div>
  </section>
  <section id="right">
    <div id="graph"><div style="padding:14px;color:#64748b;font-size:14px">The tree appears here. Name people on the left, then click a person for their report.</div></div>
    <div id="report">2. Click a person on the tree to get their family-history report (graph queries + exact genetics).</div>
  </section>
</main>
<script>
const MAXP = __MAX__;
const CONDS = __CONDS__;
const EXAMPLE = __EXAMPLE__;
let rows = [], nextId = 1, network = null, nodesDS = null, edgesDS = null;

const condSel = document.getElementById('cond');
CONDS.forEach(c => { const o = document.createElement('option'); o.value = c; o.textContent = c; condSel.appendChild(o); });
condSel.onchange = () => { hideReport(); };

function newRow(o) {
  return Object.assign({ rid: nextId++, name: '', sex: 'M', affected: false, father: null, mother: null, partner: null }, o || {});
}
function addRow(o) {
  if (rows.length >= MAXP) { setMsg('Up to ' + MAXP + ' people.'); return; }
  rows.push(newRow(o)); renderRows(); redraw();
}
function removeRow(rid) {
  rows = rows.filter(r => r.rid !== rid);
  rows.forEach(r => { ['father', 'mother', 'partner'].forEach(k => { if (r[k] === rid) r[k] = null; }); });
  renderRows(); redraw();
}
function clearAll() { rows = []; setMsg(''); renderRows(); redraw(); hideReport(); }
function loadExample() {
  rows = []; nextId = 1;
  condSel.value = EXAMPLE.condition;
  const byName = {};
  EXAMPLE.people.forEach(p => { const r = newRow({ name: p.name, sex: p.sex, affected: !!p.affected }); rows.push(r); byName[p.name] = r; });
  EXAMPLE.people.forEach(p => {
    const r = byName[p.name];
    if (p.father) r.father = byName[p.father].rid;
    if (p.mother) r.mother = byName[p.mother].rid;
    if (p.partner) r.partner = byName[p.partner].rid;
  });
  setMsg(''); renderRows(); redraw(); hideReport();
}
function setMsg(t) { document.getElementById('msg').textContent = t; }
function hideReport() { document.getElementById('report').textContent = '2. Click a person on the tree to get their family-history report (graph queries + exact genetics).'; }

function opts(list, sel) {
  let h = '<option value="">-</option>';
  list.forEach(r => { h += '<option value="' + r.rid + '"' + (sel === r.rid ? ' selected' : '') + '>' + esc(r.name || '(unnamed)') + '</option>'; });
  return h;
}
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

function renderRows() {
  const tb = document.getElementById('rows'); tb.innerHTML = '';
  rows.forEach(r => {
    const tr = document.createElement('tr');
    const males = rows.filter(x => x.sex === 'M' && x.rid !== r.rid);
    const females = rows.filter(x => x.sex === 'F' && x.rid !== r.rid);
    const others = rows.filter(x => x.sex !== r.sex && x.rid !== r.rid);
    tr.innerHTML =
      '<td><input type="text" maxlength="12" value="' + esc(r.name) + '" data-k="name" placeholder="nickname"></td>' +
      '<td><select data-k="sex"><option value="M"' + (r.sex === 'M' ? ' selected' : '') + '>M</option><option value="F"' + (r.sex === 'F' ? ' selected' : '') + '>F</option></select></td>' +
      '<td style="text-align:center"><input type="checkbox" data-k="affected"' + (r.affected ? ' checked' : '') + '></td>' +
      '<td><select data-k="father">' + opts(males, r.father) + '</select></td>' +
      '<td><select data-k="mother">' + opts(females, r.mother) + '</select></td>' +
      '<td><select data-k="partner">' + opts(others, r.partner) + '</select></td>' +
      '<td><button class="x" title="remove">x</button></td>';
    tr.querySelectorAll('[data-k]').forEach(el => {
      el.addEventListener(el.tagName === 'INPUT' && el.type === 'text' ? 'input' : 'change', () => {
        const k = el.dataset.k;
        if (k === 'affected') r[k] = el.checked;
        else if (k === 'name') r[k] = el.value;
        else if (k === 'sex') { r[k] = el.value; r.partner = null; rows.forEach(o => { if (o.father === r.rid || o.mother === r.rid) { o.father = null; o.mother = null; } }); }
        else r[k] = el.value ? parseInt(el.value, 10) : null;
        if (k === 'sex') renderRows(); else if (k === 'name') renderRowsKeepFocus();
        if (k === 'name') redrawSoon(); else redraw();
      });
    });
    tr.querySelector('button.x').onclick = () => removeRow(r.rid);
    tb.appendChild(tr);
  });
  document.getElementById('count').textContent = rows.length + ' / ' + MAXP + ' people';
}
function renderRowsKeepFocus() {
  {  // refresh the other dropdowns without losing the cursor
    document.querySelectorAll('#rows tr').forEach((tr, i) => {
      const r = rows[i];
      const males = rows.filter(x => x.sex === 'M' && x.rid !== r.rid);
      const females = rows.filter(x => x.sex === 'F' && x.rid !== r.rid);
      const others = rows.filter(x => x.sex !== r.sex && x.rid !== r.rid);
      tr.querySelector('[data-k=father]').innerHTML = opts(males, r.father);
      tr.querySelector('[data-k=mother]').innerHTML = opts(females, r.mother);
      tr.querySelector('[data-k=partner]').innerHTML = opts(others, r.partner);
    });
  }
}
let _t = null;
function redrawSoon() { clearTimeout(_t); _t = setTimeout(redraw, 350); }

function depthMap() {
  const byId = {}; rows.forEach(r => byId[r.rid] = r);
  const memo = {};
  function d(r, stack) {
    if (memo[r.rid] !== undefined) return memo[r.rid];
    if (stack.has(r.rid)) return 0;
    if (!r.father || !r.mother || !byId[r.father] || !byId[r.mother]) { memo[r.rid] = 0; return 0; }
    const s2 = new Set(stack); s2.add(r.rid);
    memo[r.rid] = 1 + Math.max(d(byId[r.father], s2), d(byId[r.mother], s2));
    return memo[r.rid];
  }
  rows.forEach(r => d(r, new Set()));
  const lvl = Object.assign({}, memo);
  rows.forEach(r => {   // in-laws sit on their partner's row
    const pr = couples().filter(c => c[0] === r.rid || c[1] === r.rid);
    pr.forEach(c => { const o = c[0] === r.rid ? c[1] : c[0]; if (!r.father) lvl[r.rid] = Math.max(lvl[r.rid], memo[o] || 0); });
  });
  return lvl;
}
function couples() {
  const seen = new Set(), out = [];
  const add = (m, f) => { const k = m + '-' + f; if (!seen.has(k)) { seen.add(k); out.push([m, f]); } };
  const byId = {}; rows.forEach(r => byId[r.rid] = r);
  rows.forEach(r => {
    if (r.father && r.mother) add(r.father, r.mother);
    if (r.partner && byId[r.partner]) { r.sex === 'M' ? add(r.rid, r.partner) : add(r.partner, r.rid); }
  });
  return out;
}

function redraw() {
  const box = document.getElementById('graph');
  const named = rows.filter(r => r.name.trim());
  if (!named.length) { return; }
  const lvl = depthMap();
  const ids = new Set(named.map(r => r.rid));
  nodesDS = new vis.DataSet(named.map(r => ({
    id: r.rid, label: r.name, level: lvl[r.rid] || 0,
    shape: r.sex === 'M' ? 'box' : 'circle',
    color: { background: r.affected ? '#334155' : '#ffffff', border: '#334155' },
    font: { color: r.affected ? '#ffffff' : '#0f172a' }, borderWidth: 1
  })));
  const edges = []; let n = 0;
  named.forEach(r => {
    if (r.father && r.mother && ids.has(r.father) && ids.has(r.mother)) {
      edges.push({ id: 'e' + (n++), from: r.father, to: r.rid, arrows: 'to', color: { color: '#94a3b8' }, width: 1 });
      edges.push({ id: 'e' + (n++), from: r.mother, to: r.rid, arrows: 'to', color: { color: '#94a3b8' }, width: 1 });
    }
  });
  couples().forEach(c => { if (ids.has(c[0]) && ids.has(c[1])) edges.push({ id: 'e' + (n++), from: c[0], to: c[1], dashes: true, color: { color: '#94a3b8' }, width: 1 }); });
  edgesDS = new vis.DataSet(edges);
  box.innerHTML = '';
  network = new vis.Network(box, { nodes: nodesDS, edges: edgesDS }, {
    layout: { hierarchical: { enabled: true, direction: 'UD', levelSeparation: 90, nodeSpacing: 90 } },
    physics: false, interaction: { hover: true }
  });
  network.on('click', p => { if (p.nodes && p.nodes.length) report(p.nodes[0]); });
}

function payload() {
  const byId = {}; rows.forEach(r => byId[r.rid] = r);
  const nm = id => (id && byId[id]) ? byId[id].name.trim() : '';
  return {
    condition: condSel.value,
    people: rows.map(r => ({ name: r.name.trim(), sex: r.sex, affected: r.affected, father: nm(r.father), mother: nm(r.mother), partner: nm(r.partner) }))
  };
}

async function report(rid) {
  const r = rows.find(x => x.rid === rid); if (!r) return;
  const out = document.getElementById('report');
  setMsg(''); out.textContent = 'Building a temporary graph and asking FalkorDB...';
  try {
    const resp = await fetch('/api/custom/report', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ family: payload(), person: r.name.trim() }) });
    const d = await resp.json();
    if (d.error) { out.textContent = ''; setMsg(d.error); return; }
    out.textContent = d.text; highlight(d.highlight);
  } catch (e) { out.textContent = ''; setMsg('Error: ' + e); }
}

function highlight(h) {
  if (!nodesDS) return;
  nodesDS.update(nodesDS.getIds().map(id => ({ id: id, borderWidth: 1, shadow: false })));
  edgesDS.update(edgesDS.getIds().map(id => ({ id: id, width: 1, color: { color: '#94a3b8' } })));
  const byName = {}; rows.forEach(r => byName[r.name.trim().toLowerCase()] = r.rid);
  const toId = n => byName[String(n).toLowerCase()];
  const ids = (h.nodes || []).map(toId).filter(x => x && nodesDS.get(x));
  nodesDS.update(ids.map(id => ({ id: id, borderWidth: 4, shadow: { enabled: true, color: '#ef4444', size: 18, x: 0, y: 0 } })));
  const pairs = [];
  (h.paths || []).forEach(p => { for (let i = 0; i < p.length - 1; i++) pairs.push([toId(p[i]), toId(p[i + 1])]); });
  const upd = [];
  edgesDS.get().forEach(e => { if (pairs.some(([a, b]) => (e.from === a && e.to === b) || (e.from === b && e.to === a))) upd.push({ id: e.id, width: 4, color: { color: '#ef4444' } }); });
  edgesDS.update(upd);
}

loadExample();
</script>
</body></html>
""".replace("__MAX__", str(MAX_PEOPLE)).replace("__CONDS__", json.dumps(CONDITIONS)).replace("__EXAMPLE__", json.dumps(EXAMPLE))