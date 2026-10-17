"""
Tools exposed to the agent. Every tool returns plain JSON-friendly data.

The LLM never computes risk itself: it picks a tool, the tool queries the
FalkorDB graph (Cypher) and runs the exact inference in pedigree_tools.py.
All data is synthetic.
"""
import graph_queries as gq
import pedigree_tools as pt

DISCLAIMER = (
    "Synthetic demo data. This is family-history risk awareness, not a diagnosis. "
    "Discuss any concern with a doctor or genetic counselor."
)

CONDITION_TIPS = {
    "Beta-thalassemia": "Ask about carrier screening for beta-thalassemia (a simple blood test), especially before planning children.",
    "Sickle cell disease": "Ask about sickle cell trait (carrier) screening, a simple blood test.",
    "Familial hypercholesterolemia": "Ask about a cholesterol (lipid) test and whether earlier or more frequent screening is advisable.",
}

_G = None
_CACHE = {}


def graph():
    global _G
    if _G is None:
        import config

        _G = config.connect("pedigree")
    return _G


def _family_of(person):
    rows = graph().query("MATCH (p:Person {id:$id}) RETURN p.family", {"id": person}).result_set
    return rows[0][0] if rows else None


def _ped(family):
    if family not in _CACHE:
        _CACHE[family] = pt.load_family(graph(), family)
    return _CACHE[family]


def _mode(family, mode=None):
    return mode or pt.infer_inheritance(_ped(family)).get("mode")


# ---------------------------------------------------------------- tools
def list_families():
    rows = graph().query("MATCH (p:Person) RETURN DISTINCT p.family").result_set
    return sorted(r[0] for r in rows)


def family_overview(family):
    ped = _ped(family)
    affected = [p for p in ped.order if ped.people[p]["affected"]]
    return {
        "family": family,
        "condition": ped.people[ped.order[0]].get("condition"),
        "people": len(ped.people),
        "generations": 1 + max(p["generation"] for p in ped.people.values()),
        "affected": affected,
    }


def infer_inheritance(family):
    return pt.infer_inheritance(_ped(family))


def obligate_carriers(family):
    if _mode(family) != "AR":
        return {"result": [], "note": "Only meaningful for recessive conditions; this family does not look recessive."}
    return pt.obligate_carriers(graph(), family)


def find_consanguinity(family):
    return pt.find_consanguinity(graph(), family)


def couple_risks(family, mode=None):
    return pt.couple_risks(_ped(family), _mode(family, mode))


def testing_priority(family, mode=None, top=5):
    rows = pt.testing_priority(_ped(family), _mode(family, mode), top)
    for r in rows:  # family paths come from a Cypher query inside FalkorDB
        try:
            r["path_to_affected_relative"] = gq.path_to_nearest_affected(graph(), r["person"], family)
        except Exception as e:
            print(f"[warn] Cypher path query failed, keeping Python path: {e}")
    return rows


def _relatives(ped, pid):
    parents = set(ped.parents[pid])
    children = set(ped.children[pid])
    siblings = {c for p in parents for c in ped.children[p]} - {pid}
    first = parents | children | siblings
    grandparents = {gp for p in parents for gp in ped.parents[p]}
    grandchildren = {gc for c in children for gc in ped.children[c]}
    aunts_uncles = {s for p in parents for pp in ped.parents[p] for s in ped.children[pp]} - parents
    nieces_nephews = {c for s in siblings for c in ped.children[s]}
    second = (grandparents | grandchildren | aunts_uncles | nieces_nephews) - first - {pid}
    return first, second


def health_report(person):
    family = _family_of(person)
    if family is None:
        return {"error": f"Unknown person {person}"}
    ped = _ped(family)
    inf = pt.infer_inheritance(ped)
    mode = inf.get("mode")
    me = ped.people[person]

    try:  # family history and path are answered by Cypher queries inside FalkorDB
        fh = gq.family_history(graph(), person)
        first_aff, second_aff = fh["first_degree_affected"], fh["second_degree_affected"]
        path = gq.path_to_nearest_affected(graph(), person, family)
    except Exception as e:
        print(f"[warn] Cypher query failed, using Python fallback: {e}")
        first, second = _relatives(ped, person)
        first_aff = sorted(r for r in first if ped.people[r]["affected"])
        second_aff = sorted(r for r in second if ped.people[r]["affected"])
        path = pt.path_to_affected(ped, person)
    if len(first_aff) >= 2 or (first_aff and second_aff):
        flag = "Condition clearly runs in this family (multiple close relatives affected)."
    elif first_aff:
        flag = "One close relative is affected."
    elif second_aff:
        flag = "Only more distant relatives are affected."
    else:
        flag = "No affected close relatives recorded."

    report = {
        "person": person,
        "family": family,
        "condition": me.get("condition"),
        "currently_affected": me["affected"],
        "inheritance_pattern": inf.get("best_pattern"),
        "family_history": {
            "first_degree_affected": first_aff,
            "second_degree_affected": second_aff,
            "summary": flag,
        },
        "your_risk": None,
        "path_to_nearest_affected_relative": path,
        "partner_planning": [],
        "next_steps": [],
        "disclaimer": DISCLAIMER,
    }
    if mode is None:
        return report

    risks = {r["person"]: r for r in (pt.relative_risks(ped, mode) or [])}
    if person in risks:
        r = risks[person]
        report["your_risk"] = {"value": r["risk"], "meaning": r["meaning"]}

    consang = {
        frozenset((c["partner_1"], c["partner_2"])): c["common_ancestors"]
        for c in pt.find_consanguinity(graph(), family)
    }
    for c in pt.couple_risks(ped, mode) or []:
        if person in c["partners"]:
            other = next(x for x in c["partners"] if x != person)
            report["partner_planning"].append(
                {
                    "partner": other,
                    "chance_child_affected": c["chance_child_affected"],
                    "shared_ancestors": consang.get(frozenset(c["partners"]), []),
                }
            )

    steps = []
    if first_aff:
        steps.append("Share this family history with your doctor.")
    if report["your_risk"] and report["your_risk"]["value"] >= 0.5:
        steps.append("Ask about genetic (carrier) testing for this condition.")
    if any(p["chance_child_affected"] >= 0.1 for p in report["partner_planning"]):
        steps.append("Consider genetic counseling before planning children.")
    tip = CONDITION_TIPS.get(me.get("condition"))
    if tip and (first_aff or (report["your_risk"] and report["your_risk"]["value"] >= 0.5)):
        steps.append(tip)
    if me["affected"]:
        steps.insert(0, "Already affected: discuss management and what this means for relatives with a clinician or genetic counselor.")
    elif not steps:
        steps.append("No elevated risk found under this model; keep family history up to date.")
    report["next_steps"] = steps
    return report


# ---------------------------------------------------------------- schemas
def _schema(name, desc, props, required):
    return {
        "name": name,
        "description": desc,
        "input_schema": {"type": "object", "properties": props, "required": required},
    }


_FAM = {"family": {"type": "string", "description": "Family id, e.g. FAM1"}}
_MODE = {"mode": {"type": "string", "enum": ["AR", "AD"], "description": "Optional; inferred if omitted"}}

TOOLS = [
    _schema("list_families", "List the family ids stored in the graph.", {}, []),
    _schema("family_overview", "Size, generations and affected members of a family.", _FAM, ["family"]),
    _schema(
        "infer_inheritance",
        "Decide whether the family's condition looks autosomal recessive or dominant (exact comparison).",
        _FAM,
        ["family"],
    ),
    _schema(
        "obligate_carriers",
        "Graph query: unaffected parents of affected children (must be recessive carriers).",
        _FAM,
        ["family"],
    ),
    _schema(
        "find_consanguinity",
        "Graph query: partners who share an ancestor, with the common ancestors.",
        _FAM,
        ["family"],
    ),
    _schema(
        "couple_risks",
        "Exact chance that a future child of each couple is affected.",
        {**_FAM, **_MODE},
        ["family"],
    ),
    _schema(
        "testing_priority",
        "Untested relatives ranked by risk, with the family path to the nearest affected relative.",
        {**_FAM, **_MODE, "top": {"type": "integer"}},
        ["family"],
    ),
    _schema(
        "health_report",
        "Full family-history report for ONE person: pattern, personal risk, affected relatives, "
        "partner planning, family path, next steps. Use this for any question about a specific person.",
        {"person": {"type": "string", "description": "Person id, e.g. FAM1_P14"}},
        ["person"],
    ),
]

DISPATCH = {
    "list_families": list_families,
    "family_overview": family_overview,
    "infer_inheritance": infer_inheritance,
    "obligate_carriers": obligate_carriers,
    "find_consanguinity": find_consanguinity,
    "couple_risks": couple_risks,
    "testing_priority": testing_priority,
    "health_report": health_report,
}


def run_tool(name, args):
    try:
        return DISPATCH[name](**args)
    except Exception as e:  # return the error to the model instead of crashing
        return {"error": f"{type(e).__name__}: {e}"}