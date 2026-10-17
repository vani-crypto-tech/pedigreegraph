"""
Cross-check: every graph query (Cypher, inside FalkorDB) must agree with an independent
pure-Python implementation over the same family. Run:  python test_cypher.py
"""
import time

import agent_tools as at
import graph_queries as gq
import pedigree_tools as pt


def closure(start, step, depth):
    seen, frontier = set(), {start}
    for _ in range(depth):
        frontier = {y for x in frontier for y in step[x]}
        seen |= frontier
    return sorted(seen)


def main():
    import config

    g = config.connect("pedigree")
    fams = sorted(r[0] for r in g.query("MATCH (p:Person) RETURN DISTINCT p.family").result_set)
    checks = bad = 0
    t0 = time.time()

    def check(name, got, want, pid):
        nonlocal checks, bad
        checks += 1
        if got != want:
            bad += 1
            print(f"MISMATCH {name} for {pid}: cypher={got} python={want}")

    for fam in fams:
        ped = pt.load_family(g, fam)
        for pid in ped.order:
            check("ancestors", gq.ancestors(g, pid, 4), closure(pid, ped.parents, 4), pid)
            check("descendants", gq.descendants(g, pid, 4), closure(pid, ped.children, 4), pid)

            fh = gq.family_history(g, pid)
            first, second = at._relatives(ped, pid)
            check("first_degree_affected", fh["first_degree_affected"],
                  sorted(r for r in first if ped.people[r]["affected"]), pid)
            check("second_degree_affected", fh["second_degree_affected"],
                  sorted(r for r in second if ped.people[r]["affected"]), pid)

            c_path = gq.path_to_nearest_affected(g, pid, fam)
            p_path = pt.path_to_affected(ped, pid)
            check("path_length", None if c_path is None else len(c_path),
                  None if p_path is None else len(p_path), pid)
            if c_path:
                check("path_endpoints", (c_path[0], ped.people[c_path[-1]]["affected"]), (pid, True), pid)

    print(f"\n{checks} checks in {time.time() - t0:.1f}s, {bad} mismatches")
    print("ALL CHECKS PASSED" if bad == 0 else "SOME CHECKS FAILED: send me the lines above")


if __name__ == "__main__":
    main()