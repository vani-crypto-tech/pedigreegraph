"""
Checks that the fast engine (variable elimination) gives the same answers as the original
brute-force engine.

  python compare_engines.py            random families (no database needed) + a timing test
  python compare_engines.py --demo     also all demo families from FalkorDB (set the FALKORDB_* variables first)
"""
import random
import sys
import time

import pedigree_tools as pt

TOL = 1e-9


def random_family(n, rng, cousin_marriage=False):
    """A random valid pedigree with n people (some founders, some children, extra partners, optional loops)."""
    people, parent_of, partners = [], [], []
    sex, gen = {}, {}

    def add(sx, g):
        pid = f"P{len(people) + 1}"
        people.append({"id": pid, "sex": sx, "generation": g, "affected": rng.random() < 0.25,
                       "tested": False, "condition": "X"})
        sex[pid], gen[pid] = sx, g
        return pid

    for _ in range(max(2, n // 6)):
        add(rng.choice("MF"), 0)
    while len(people) < n:
        males = [p for p in sex if sex[p] == "M"]
        females = [p for p in sex if sex[p] == "F"]
        r = rng.random()
        if r < 0.25 or not males or not females:
            add(rng.choice("MF"), 0)  # marries in
            continue
        f, m = rng.choice(males), rng.choice(females)
        if cousin_marriage and rng.random() < 0.5:
            kids = [p for p in sex if any(b == p for _, b in parent_of)]
            if len(kids) >= 2:
                a, b = rng.sample(kids, 2)
                if sex[a] != sex[b]:
                    f, m = (a, b) if sex[a] == "M" else (b, a)
        g = max(gen[f], gen[m]) + 1
        c = add(rng.choice("MF"), g)
        parent_of += [(f, c), (m, c)]
        partners.append((f, m))
    # a few couples with no children
    males = [p for p in sex if sex[p] == "M"]
    females = [p for p in sex if sex[p] == "F"]
    for _ in range(2):
        if males and females:
            a, b = rng.choice(males), rng.choice(females)
            if (a, b) not in partners:
                partners.append((a, b))
    for p in people:
        if p["generation"] > 0:
            pass
    return pt.Pedigree(people, parent_of, list(dict.fromkeys(partners)))


def same(a, b):
    if a["marginals"] is None or b["marginals"] is None:
        return a["marginals"] is None and b["marginals"] is None and a["likelihood"] == b["likelihood"] == 0.0
    worst = abs(a["likelihood"] - b["likelihood"]) / max(a["likelihood"], 1e-300)
    for k in a["marginals"]:
        worst = max(worst, max(abs(x - y) for x, y in zip(a["marginals"][k], b["marginals"][k])))
    for k in a["joint"]:
        for ra, rb in zip(a["joint"][k], b["joint"][k]):
            worst = max(worst, max(abs(x - y) for x, y in zip(ra, rb)))
    return worst


def check(ped, label):
    worst = 0.0
    for mode in ("AR", "AD"):
        old, new = pt.posterior_bruteforce(ped, mode), pt.posterior(ped, mode)
        d = same(old, new)
        if d is False or (d is not True and d > TOL):
            print(f"  FAIL {label} {mode}: difference {d}")
            return False
        worst = max(worst, d if d is not True else 0.0)
    return worst


if __name__ == "__main__":
    rng = random.Random(7)
    bad, worst, runs = 0, 0.0, 0
    for n in list(range(3, 13)) * 12:
        ped = random_family(n, rng, cousin_marriage=rng.random() < 0.5)
        r = check(ped, f"random n={n}")
        runs += 1
        if r is False:
            bad += 1
        else:
            worst = max(worst, r)
    print(f"random families: {runs - bad}/{runs} identical, largest difference {worst:.2e}")

    if "--demo" in sys.argv:
        import config

        g = config.connect("pedigree")
        fams = sorted(r[0] for r in g.query("MATCH (p:Person) RETURN DISTINCT p.family").result_set)
        ok = 0
        for fam in fams:
            ped = pt.load_family(g, fam)
            r = check(ped, fam)
            ok += r is not False
            print(f"  {fam}: {'same' if r is not False else 'DIFFERENT'}")
        print(f"demo families: {ok}/{len(fams)} identical")

    print("\nTiming of the fast engine (both modes, one report's worth of work):")
    for n in (20, 30, 40, 50):
        ped = random_family(n, random.Random(n))
        t = time.time()
        for m in ("AR", "AD"):
            pt.posterior(ped, m)
        print(f"  {n} people: {time.time() - t:.2f} s")
    sys.exit(1 if bad else 0)