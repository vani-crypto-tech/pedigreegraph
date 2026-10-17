"""
Graph-native queries: the family structure questions are answered INSIDE FalkorDB with Cypher.

  ancestors / descendants   multi-hop traversals over PARENT_OF
  relatives_within          blood relatives within N hops (undirected traversal)
  family_history            affected first- and second-degree relatives of a person
  path_to_nearest_affected  shortest family path from a person to an affected relative

test_cypher.py checks every one of these against an independent Python implementation.
"""


def _hops(n):
    return int(n)  # literal, validated integer (variable-length bounds cannot be parameters)


def ancestors(g, person, depth=4):
    q = f"MATCH (anc:Person)-[:PARENT_OF*1..{_hops(depth)}]->(p:Person {{id:$id}}) RETURN DISTINCT anc.id"
    return sorted(r[0] for r in g.query(q, {"id": person}).result_set)


def descendants(g, person, depth=4):
    q = f"MATCH (p:Person {{id:$id}})-[:PARENT_OF*1..{_hops(depth)}]->(d:Person) RETURN DISTINCT d.id"
    return sorted(r[0] for r in g.query(q, {"id": person}).result_set)


def relatives_within(g, person, hops=3):
    q = (
        f"MATCH (p:Person {{id:$id}})-[:PARENT_OF*1..{_hops(hops)}]-(r:Person) "
        "WHERE r.id <> $id RETURN DISTINCT r.id"
    )
    return sorted(r[0] for r in g.query(q, {"id": person}).result_set)


_Q_PARENTS = "MATCH (x:Person)-[:PARENT_OF]->(p:Person {id:$id}) RETURN DISTINCT x.id, x.affected"
_Q_CHILDREN = "MATCH (p:Person {id:$id})-[:PARENT_OF]->(x:Person) RETURN DISTINCT x.id, x.affected"
_Q_SIBLINGS = (
    "MATCH (par:Person)-[:PARENT_OF]->(p:Person {id:$id}), (par)-[:PARENT_OF]->(x:Person) "
    "WHERE x.id <> $id RETURN DISTINCT x.id, x.affected"
)
_Q_GRANDPARENTS = (
    "MATCH (x:Person)-[:PARENT_OF]->(:Person)-[:PARENT_OF]->(p:Person {id:$id}) RETURN DISTINCT x.id, x.affected"
)
_Q_GRANDCHILDREN = (
    "MATCH (p:Person {id:$id})-[:PARENT_OF]->(:Person)-[:PARENT_OF]->(x:Person) RETURN DISTINCT x.id, x.affected"
)
_Q_AUNTS_UNCLES = (
    "MATCH (gp:Person)-[:PARENT_OF]->(par:Person)-[:PARENT_OF]->(p:Person {id:$id}), (gp)-[:PARENT_OF]->(x:Person) "
    "WHERE x.id <> par.id RETURN DISTINCT x.id, x.affected"
)
_Q_NIECES_NEPHEWS = (
    "MATCH (par:Person)-[:PARENT_OF]->(p:Person {id:$id}), (par)-[:PARENT_OF]->(s:Person)-[:PARENT_OF]->(x:Person) "
    "WHERE s.id <> $id RETURN DISTINCT x.id, x.affected"
)


def family_history(g, person):
    """First-degree (parents, children, siblings) and second-degree relatives, with who is affected."""

    def rows(q):
        return {r[0]: bool(r[1]) for r in g.query(q, {"id": person}).result_set}

    first = {}
    for q in (_Q_PARENTS, _Q_CHILDREN, _Q_SIBLINGS):
        first.update(rows(q))
    second = {}
    for q in (_Q_GRANDPARENTS, _Q_GRANDCHILDREN, _Q_AUNTS_UNCLES, _Q_NIECES_NEPHEWS):
        second.update(rows(q))
    for k in list(second):
        if k in first or k == person:
            del second[k]
    return {
        "first_degree_affected": sorted(k for k, v in first.items() if v),
        "second_degree_affected": sorted(k for k, v in second.items() if v),
    }


def path_to_nearest_affected(g, person, family, max_hops=6):
    """Shortest path of people (through parent/child links) to the nearest other affected relative."""
    q = (
        f"MATCH p = (a:Person {{id:$id}})-[:PARENT_OF*1..{_hops(max_hops)}]-(b:Person {{family:$fam, affected:true}}) "
        "WHERE b.id <> $id "
        "RETURN [n IN nodes(p) | n.id] AS path, length(p) AS hops "
        "ORDER BY hops LIMIT 1"
    )
    rows = g.query(q, {"id": person, "fam": family}).result_set
    return rows[0][0] if rows else None