"""
Synthetic pedigree generator + FalkorDB loader.

- Builds several fictional families with ONE planted condition each
  (autosomal recessive "AR" or autosomal dominant "AD").
- The graph only stores what a counselor could observe:
  who is related to whom, sex, generation, affected or not.
- The hidden truth (true genotypes, inheritance mode) goes to
  ground_truth.json, so we can later check the agent's answers.

All data is synthetic. No real people, no real genomes.
"""
import json
import os
import random


def transmit(geno, rng):
    """Chance a parent passes on the disease allele: 0 copies -> 0, 1 -> 50%, 2 -> 100%."""
    return 1 if rng.random() < geno / 2 else 0


def opposite(sex):
    return "F" if sex == "M" else "M"


def build_family(fam, mode, rng):
    """Build one 4-generation family. Returns None if the result is not interesting enough."""
    people, parent_of, partners = {}, [], []

    def add(sex, geno, gen):
        pid = f"{fam}_P{len(people) + 1}"
        affected = (geno == 2) if mode == "AR" else (geno >= 1)
        people[pid] = {
            "id": pid, "family": fam, "sex": sex,
            "generation": gen, "geno": geno, "affected": affected,
        }
        return pid

    def have_child(father, mother, gen):
        geno = transmit(people[father]["geno"], rng) + transmit(people[mother]["geno"], rng)
        child = add(rng.choice("MF"), geno, gen)
        parent_of.append((father, child))
        parent_of.append((mother, child))
        return child

    # Generation 0: one founder carries the disease allele
    grandfather = add("M", 1, 0)
    grandmother = add("F", 0, 0)
    partners.append((grandfather, grandmother))

    # Generation 1: three children, each marries an unrelated outsider
    # Generation 2: two children per couple
    sibships = []
    for _ in range(3):
        child = have_child(grandfather, grandmother, 1)
        spouse = add(opposite(people[child]["sex"]), 0, 1)
        pair = (child, spouse) if people[child]["sex"] == "M" else (spouse, child)
        partners.append(pair)
        sibships.append([have_child(pair[0], pair[1], 2) for _ in range(2)])

    # Generation 3: two cousins (from different sibships) become partners
    options = [
        (a, b)
        for i in range(3) for j in range(i + 1, 3)
        for a in sibships[i] for b in sibships[j]
        if people[a]["sex"] != people[b]["sex"]
    ]
    if not options:
        return None
    both_carry = [o for o in options if people[o[0]]["geno"] >= 1 and people[o[1]]["geno"] >= 1]
    a, b = rng.choice(both_carry or options)
    father, mother = (a, b) if people[a]["sex"] == "M" else (b, a)
    partners.append((father, mother))
    for _ in range(4):
        have_child(father, mother, 3)

    # Keep only families where the condition actually shows up enough
    n_affected = sum(p["affected"] for p in people.values())
    if n_affected < (1 if mode == "AR" else 3):
        return None

    return {"people": people, "parent_of": parent_of, "partners": partners}


def generate(seed=42):
    rng = random.Random(seed)
    # the first three must stay exactly as they are (README demo questions use them)
    plan = [
        ("FAM1", "AR", "Beta-thalassemia"),
        ("FAM2", "AD", "Familial hypercholesterolemia"),
        ("FAM3", "AR", "Sickle cell disease"),
    ]
    # extra families, appended AFTER the originals so the originals do not change
    extra_ar = ["Cystic fibrosis", "Beta-thalassemia", "Sickle cell disease", "Tay-Sachs disease"]
    extra_ad = ["Familial hypercholesterolemia", "Huntington's disease", "Marfan syndrome"]
    for i in range(4, 31):  # FAM4 .. FAM30
        if i % 3 == 0:
            plan.append((f"FAM{i}", "AD", extra_ad[i % len(extra_ad)]))
        else:
            plan.append((f"FAM{i}", "AR", extra_ar[i % len(extra_ar)]))

    people, parent_of, partners, truth = [], [], [], {}
    for fam, mode, cond in plan:
        result = None
        while result is None:
            result = build_family(fam, mode, rng)
        for p in result["people"].values():
            people.append({k: p[k] for k in ("id", "family", "sex", "generation", "affected")} | {"condition": cond})
            truth[p["id"]] = {"geno": p["geno"], "mode": mode, "family": fam, "condition": cond}
        parent_of += [{"a": a, "b": b} for a, b in result["parent_of"]]
        partners += [{"a": a, "b": b} for a, b in result["partners"]]
    return people, parent_of, partners, truth


def load(people, parent_of, partners):
    import config

    graph_name = os.environ.get("PEDIGREE_GRAPH", "pedigree_big")
    db = config.connect_db()
    try:
        db.select_graph(graph_name).delete()  # start clean
    except Exception:
        pass
    g = db.select_graph(graph_name)

    g.query(
        "UNWIND $rows AS r CREATE (:Person {id: r.id, family: r.family, condition: r.condition, sex: r.sex, "
        "generation: r.generation, affected: r.affected, tested: false})",
        {"rows": people},
    )
    try:
        g.query("CREATE INDEX FOR (p:Person) ON (p.id)")
    except Exception:
        pass
    g.query(
        "UNWIND $rows AS r MATCH (a:Person {id: r.a}), (b:Person {id: r.b}) "
        "CREATE (a)-[:PARENT_OF]->(b)",
        {"rows": parent_of},
    )
    g.query(
        "UNWIND $rows AS r MATCH (a:Person {id: r.a}), (b:Person {id: r.b}) "
        "CREATE (a)-[:PARTNER_OF]->(b)",
        {"rows": partners},
    )
    return g


if __name__ == "__main__":
    people, parent_of, partners, truth = generate()
    with open("ground_truth.json", "w") as f:
        json.dump(truth, f, indent=2)
    print(f"Generated {len(people)} people, {len(parent_of)} parent links, {len(partners)} partnerships")

    g = load(people, parent_of, partners)

    n = g.query("MATCH (p:Person) RETURN p.family, count(p), sum(CASE WHEN p.affected THEN 1 ELSE 0 END)")
    print("\nfamily, people, affected:")
    for row in n.result_set:
        print(row)

    carriers = g.query(
        "MATCH (par:Person)-[:PARENT_OF]->(c:Person {affected: true}) "
        "WHERE NOT par.affected "
        "RETURN DISTINCT par.id, collect(c.id)"
    )
    print("\nunaffected parents of affected children (possible recessive carriers):")
    for row in carriers.result_set:
        print(row)