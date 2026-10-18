"""
Pedigree analysis tools (the future "agent tools").

Two kinds of tools:
  1. Graph tools   - Cypher queries run inside FalkorDB (carriers, consanguinity).
  2. Inference tools - exact Mendelian probability, computed by code (never by the LLM).

Assumptions (always state these in the demo):
  - Single gene, complete penetrance, no new mutations.
  - Population allele frequency: AR = 1%, AD = 0.1% (editable in ALLELE_FREQ).
  - Results are decision support for a genetic counselor, not a diagnosis.
All data is synthetic.
"""
import json
import math
from collections import deque
from itertools import product

ALLELE_FREQ = {"AR": 0.01, "AD": 0.001}
MODE_NAMES = {"AR": "autosomal recessive", "AD": "autosomal dominant"}


# --------------------------------------------------------------------------
# Loading a family from the graph
# --------------------------------------------------------------------------
class Pedigree:
    def __init__(self, people, parent_of, partners):
        self.people = {p["id"]: p for p in people}
        self.parents = {pid: [] for pid in self.people}
        self.children = {pid: [] for pid in self.people}
        for a, b in parent_of:
            self.parents[b].append(a)
            self.children[a].append(b)
        self.partners = list(partners)
        # parents always have a lower generation number, so this is a safe order
        self.order = sorted(self.people, key=lambda i: self.people[i]["generation"])


def load_family(g, family):
    """Read one family out of FalkorDB into a Pedigree."""
    prm = {"fam": family}
    rows = g.query(
        "MATCH (p:Person {family:$fam}) RETURN p.id, p.sex, p.generation, p.affected, p.tested, p.condition",
        prm,
    ).result_set
    people = [
        {"id": r[0], "sex": r[1], "generation": r[2], "affected": bool(r[3]), "tested": bool(r[4]), "condition": r[5]}
        for r in rows
    ]
    parent_of = g.query(
        "MATCH (a:Person {family:$fam})-[:PARENT_OF]->(b:Person) RETURN a.id, b.id", prm
    ).result_set
    partners = g.query(
        "MATCH (a:Person {family:$fam})-[:PARTNER_OF]->(b:Person) RETURN a.id, b.id", prm
    ).result_set
    return Pedigree(people, parent_of, partners)


# --------------------------------------------------------------------------
# Graph tools (Cypher)
# --------------------------------------------------------------------------
def obligate_carriers(g, family):
    """Unaffected parents of an affected child: must carry a recessive allele."""
    rows = g.query(
        """
        MATCH (par:Person {family:$fam})-[:PARENT_OF]->(c:Person {family:$fam, affected:true})
        WHERE NOT par.affected
        RETURN par.id, collect(c.id)
        """,
        {"fam": family},
    ).result_set
    return [{"person": r[0], "affected_children": r[1]} for r in rows]


def find_consanguinity(g, family):
    """Partners who share an ancestor (a loop in the pedigree)."""
    rows = g.query(
        """
        MATCH (a:Person {family:$fam})-[:PARTNER_OF]->(b:Person {family:$fam}),
              (anc:Person)-[:PARENT_OF*1..3]->(a),
              (anc)-[:PARENT_OF*1..3]->(b)
        RETURN a.id, b.id, collect(DISTINCT anc.id)
        """,
        {"fam": family},
    ).result_set
    return [{"partner_1": r[0], "partner_2": r[1], "common_ancestors": r[2]} for r in rows]


# --------------------------------------------------------------------------
# Exact inference engine
# --------------------------------------------------------------------------
def _prior(q):
    return [(1 - q) ** 2, 2 * q * (1 - q), q * q]


def _child_dist(gf, gm):
    tf, tm = gf / 2, gm / 2  # chance each parent passes the allele on
    return [(1 - tf) * (1 - tm), tf * (1 - tm) + (1 - tf) * tm, tf * tm]


def _is_affected(mode, geno):
    return geno == 2 if mode == "AR" else geno >= 1


def posterior_bruteforce(ped, mode):
    """
    Original brute-force engine (3^n work). Kept only to check the fast engine below against.
    Exact posterior over everyone's genotype (0, 1 or 2 disease alleles),
    given the observed affected/unaffected status of the whole family.
    Because every person is one variable, shared ancestors (consanguinity)
    are handled correctly.
    """
    prior = _prior(ALLELE_FREQ[mode])
    order = ped.order
    n = len(order)
    idx = {pid: i for i, pid in enumerate(order)}
    par = [[idx[x] for x in ped.parents[pid]] for pid in order]
    aff = [bool(ped.people[pid]["affected"]) for pid in order]
    pairs = [(idx[a], idx[b]) for a, b in ped.partners]

    geno = [0] * n
    marg = [[0.0] * 3 for _ in range(n)]
    joint = {pr: [[0.0] * 3 for _ in range(3)] for pr in pairs}
    total = [0.0]

    def rec(i, w):
        if i == n:
            total[0] += w
            for k in range(n):
                marg[k][geno[k]] += w
            for (a, b), tab in joint.items():
                tab[geno[a]][geno[b]] += w
            return
        dist = _child_dist(geno[par[i][0]], geno[par[i][1]]) if len(par[i]) == 2 else prior
        for gt in range(3):
            if dist[gt] > 0 and _is_affected(mode, gt) == aff[i]:
                geno[i] = gt
                rec(i + 1, w * dist[gt])

    rec(0, 1.0)
    if total[0] == 0:
        return {"likelihood": 0.0, "marginals": None, "joint": None}
    return {
        "likelihood": total[0],
        "marginals": {order[k]: [v / total[0] for v in marg[k]] for k in range(n)},
        "joint": {
            (order[a], order[b]): [[v / total[0] for v in row] for row in tab]
            for (a, b), tab in joint.items()
        },
    }


# --------------------------------------------------------------------------
# Exact inference by variable elimination (scales to 50+ people)
# --------------------------------------------------------------------------
# The brute-force engine above lists every genotype combination: 3^n work, so it
# freezes around 22 people. This engine gives the SAME exact answers, but it
# eliminates one person at a time (a clique tree over the pedigree), so the work
# grows with how tangled the family is, not with how many people it has.
class PedigreeTooComplex(ValueError):
    """The family has too many tangled loops for the exact engine to finish quickly."""


MAX_CLIQUE = 11  # most people one calculation step may hold at once (3^11 = 177k entries)


def _product(scope, factors):
    """Multiply factors (vars, table) into one table over `scope` (first variable = most significant digit)."""
    k = len(scope)
    pos = {v: i for i, v in enumerate(scope)}
    maps = []
    for fv, ft in factors:
        st = [0] * k
        m = len(fv)
        for j, v in enumerate(fv):
            st[pos[v]] = 3 ** (m - 1 - j)
        maps.append((ft, [(i, s) for i, s in enumerate(st) if s]))
    out = []
    for assign in product((0, 1, 2), repeat=k):
        w = 1.0
        for ft, st in maps:
            idx = 0
            for i, s in st:
                idx += assign[i] * s
            w *= ft[idx]
            if w == 0.0:
                break
        out.append(w)
    return out


def _sum_to(scope, table, keep):
    """Add up the table over every variable not in `keep` (result ordered like `keep`)."""
    pos = {v: i for i, v in enumerate(scope)}
    kp = [pos[v] for v in keep]
    out = [0.0] * (3 ** len(keep))
    for i, assign in enumerate(product((0, 1, 2), repeat=len(scope))):
        idx = 0
        for p in kp:
            idx = idx * 3 + assign[p]
        out[idx] += table[i]
    return out


def _elimination_plan(n, scopes, extra_edges):
    """Greedy min-fill elimination order. Returns (order, neighbours-at-elimination-time)."""
    adj = [set() for _ in range(n)]
    for sc in scopes:
        for a in sc:
            for b in sc:
                if a != b:
                    adj[a].add(b)
    for a, b in extra_edges:
        if a != b:
            adj[a].add(b)
            adj[b].add(a)
    left = set(range(n))
    order, nbrs = [], {}
    while left:
        best, best_key = None, None
        for v in sorted(left):
            nb = adj[v]
            fill = sum(1 for a in nb for b in nb if a < b and b not in adj[a])
            key = (fill, len(nb), v)
            if best_key is None or key < best_key:
                best, best_key = v, key
        v = best
        nb = set(adj[v])
        if len(nb) + 1 > MAX_CLIQUE:
            raise PedigreeTooComplex(
                "This family has too many tangled loops (for example many marriages between relatives) "
                "to calculate exactly."
            )
        for a in nb:
            adj[a].discard(v)
            for b in nb:
                if a != b:
                    adj[a].add(b)
        order.append(v)
        nbrs[v] = nb
        left.discard(v)
    return order, nbrs


def posterior(ped, mode):
    """
    Exact posterior over everyone's genotype (0, 1 or 2 disease alleles), given the observed
    affected/unaffected status of the whole family. Same result as posterior_bruteforce, but computed
    by variable elimination on a clique tree, so a 50-person family takes a fraction of a second.
    Shared ancestors (consanguinity) are handled correctly because every person is one variable.
    """
    cache = ped.__dict__.setdefault("_posterior_cache", {})
    if mode in cache:
        return cache[mode]
    cache[mode] = res = _posterior_ve(ped, mode)
    return res


def _posterior_ve(ped, mode):
    prior = _prior(ALLELE_FREQ[mode])
    order = ped.order
    n = len(order)
    idx = {pid: i for i, pid in enumerate(order)}

    # one factor per person: P(genotype | parents) times "matches the observed status"
    factors = []
    for i, pid in enumerate(order):
        aff = bool(ped.people[pid]["affected"])
        ps = ped.parents[pid]
        if len(ps) == 2:
            f, m = idx[ps[0]], idx[ps[1]]
            t = []
            for gc in range(3):
                ok = 1.0 if _is_affected(mode, gc) == aff else 0.0
                for gf in range(3):
                    for gm in range(3):
                        t.append(_child_dist(gf, gm)[gc] * ok)
            factors.append(((i, f, m), t))
        else:
            factors.append(((i,), [prior[g] if _is_affected(mode, g) == aff else 0.0 for g in range(3)]))

    pairs = []
    for a, b in ped.partners:
        if (idx[a], idx[b]) not in pairs:
            pairs.append((idx[a], idx[b]))

    elim, nbrs = _elimination_plan(n, [f[0] for f in factors], pairs)
    pos = {v: k for k, v in enumerate(elim)}
    scope = {v: tuple([v] + sorted(nbrs[v], key=pos.get)) for v in elim}  # clique of v
    sep = {v: tuple(sorted(nbrs[v], key=pos.get)) for v in elim}          # what v hands to its parent
    parent = {v: (min(nbrs[v], key=pos.get) if nbrs[v] else None) for v in elim}
    kids = {v: [] for v in elim}
    for v in elim:
        if parent[v] is not None:
            kids[parent[v]].append(v)
    own = {v: [] for v in elim}
    for fv, ft in factors:
        own[min(fv, key=pos.get)].append((fv, ft))

    # upward pass (messages are rescaled so tiny probabilities never underflow)
    up, logz = {}, 0.0
    for v in elim:
        fs = own[v] + [(sep[c], up[c]) for c in kids[v]]
        msg = _sum_to(scope[v], _product(scope[v], fs), sep[v])
        top = max(msg)
        if top <= 0.0:
            return {"likelihood": 0.0, "marginals": None, "joint": None}
        logz += math.log(top)
        up[v] = [x / top for x in msg]

    # downward pass and beliefs
    down, belief = {}, {}
    for v in reversed(elim):
        fs = own[v] + [(sep[c], up[c]) for c in kids[v]]
        if parent[v] is not None:
            fs = fs + [(sep[v], down[v])]
        belief[v] = _product(scope[v], fs)
        for c in kids[v]:
            fs_c = own[v] + [(sep[o], up[o]) for o in kids[v] if o != c]
            if parent[v] is not None:
                fs_c = fs_c + [(sep[v], down[v])]
            msg = _sum_to(scope[v], _product(scope[v], fs_c), sep[c])
            top = max(msg)
            down[c] = [x / top for x in msg] if top > 0 else msg

    marg = {}
    for v in elim:
        t = _sum_to(scope[v], belief[v], (v,))
        s = sum(t)
        marg[order[v]] = [x / s for x in t]
    joint = {}
    for a, b in pairs:
        v = a if pos[a] < pos[b] else b  # the one eliminated first holds the other in its clique
        t = _sum_to(scope[v], belief[v], (a, b))
        s = sum(t)
        joint[(order[a], order[b])] = [[t[3 * x + y] / s for y in range(3)] for x in range(3)]
    return {"likelihood": math.exp(logz), "marginals": marg, "joint": joint}


# --------------------------------------------------------------------------
# Inference tools
# --------------------------------------------------------------------------
def infer_inheritance(ped):
    """Which inheritance pattern explains this family better? (equal prior on each)"""
    lik = {m: posterior(ped, m)["likelihood"] for m in ("AR", "AD")}
    tot = sum(lik.values())
    if tot == 0:
        return {"best_pattern": None, "note": "Neither pattern fits; consider X-linked or other causes."}
    probs = {m: lik[m] / tot for m in lik}
    best = max(probs, key=probs.get)
    return {
        "best_pattern": MODE_NAMES[best],
        "mode": best,
        "probabilities": {MODE_NAMES[m]: round(p, 4) for m, p in probs.items()},
        "note": "Compared under the stated allele-frequency assumptions. X-linked not modelled.",
    }


def path_to_affected(ped, pid):
    """Shortest family path (through parent/child links) from a person to the nearest affected relative."""
    seen, queue = {pid: None}, deque([pid])
    while queue:
        cur = queue.popleft()
        if cur != pid and ped.people[cur]["affected"]:
            path = []
            while cur is not None:
                path.append(cur)
                cur = seen[cur]
            return path[::-1]
        for nxt in ped.parents[cur] + ped.children[cur]:
            if nxt not in seen:
                seen[nxt] = cur
                queue.append(nxt)
    return None


def relative_risks(ped, mode):
    """Per-person risk for everyone who is currently unaffected."""
    post = posterior(ped, mode)
    if post["marginals"] is None:
        return None
    out = []
    for pid in ped.order:
        p = ped.people[pid]
        if p["affected"]:
            continue
        g0, g1, g2 = post["marginals"][pid]
        out.append(
            {
                "person": pid,
                "risk": round(g1 if mode == "AR" else g1 + g2, 4),
                "meaning": "chance of being a carrier" if mode == "AR" else "chance of having inherited the allele",
                "tested": p.get("tested", False),
                "path_to_affected_relative": path_to_affected(ped, pid),
            }
        )
    return sorted(out, key=lambda r: -r["risk"])


def couple_risks(ped, mode):
    """For each couple: chance that a future child is affected."""
    post = posterior(ped, mode)
    if post["joint"] is None:
        return None
    out = []
    for (a, b), tab in post["joint"].items():
        p_aff = 0.0
        for ga in range(3):
            for gb in range(3):
                cd = _child_dist(ga, gb)
                child_aff = cd[2] if mode == "AR" else 1 - cd[0]
                p_aff += tab[ga][gb] * child_aff
        out.append({"partners": [a, b], "chance_child_affected": round(p_aff, 4)})
    return sorted(out, key=lambda r: -r["chance_child_affected"])


def testing_priority(ped, mode, top=5):
    """Who to test first: highest-risk relatives who have not been tested."""
    risks = relative_risks(ped, mode) or []
    return [r for r in risks if not r["tested"]][:top]


# --------------------------------------------------------------------------
# Demo: run every tool on every family
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import config

    g = config.connect("pedigree")
    try:
        truth = json.load(open("ground_truth.json"))
    except FileNotFoundError:
        truth = {}

    families = sorted(r[0] for r in g.query("MATCH (p:Person) RETURN DISTINCT p.family").result_set)
    for fam in families:
        ped = load_family(g, fam)
        print("=" * 70)
        print(f"{fam}: {len(ped.people)} people, {sum(p['affected'] for p in ped.people.values())} affected")

        inf = infer_inheritance(ped)
        print("Inferred pattern:", inf["best_pattern"], inf.get("probabilities"))
        real = next((t["mode"] for pid, t in truth.items() if t["family"] == fam), None)
        if real:
            print("  (hidden truth:", MODE_NAMES[real] + ")")

        mode = inf.get("mode")
        if mode is None:
            continue
        if mode == "AR":
            print("Obligate carriers:", json.dumps(obligate_carriers(g, fam)))
        print("Consanguinity:", json.dumps(find_consanguinity(g, fam)))
        print("Couples (chance a child is affected):", json.dumps(couple_risks(ped, mode)[:3]))
        print("Test first:")
        for r in testing_priority(ped, mode, top=3):
            print("  ", r["person"], "risk", r["risk"], "via", " -> ".join(r["path_to_affected_relative"] or []))