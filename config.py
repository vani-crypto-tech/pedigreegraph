"""
FalkorDB connection settings, read from environment variables.
Defaults point at the local Docker container. For FalkorDB Cloud set:

  FALKORDB_HOST, FALKORDB_PORT, FALKORDB_USER (default "falkordb"), FALKORDB_PASSWORD
  FALKORDB_SSL=true   (only if your instance uses TLS; the Free tier has none)
  PEDIGREE_GRAPH      (optional) graph name to use instead of "pedigree"
"""
import os


def connect_db():
    from falkordb import FalkorDB

    kwargs = {
        "host": os.environ.get("FALKORDB_HOST", "localhost"),
        "port": int(os.environ.get("FALKORDB_PORT", "6379")),
    }
    password = os.environ.get("FALKORDB_PASSWORD")
    if password:
        kwargs["username"] = os.environ.get("FALKORDB_USER", "falkordb")
        kwargs["password"] = password
    if os.environ.get("FALKORDB_SSL", "").lower() in ("1", "true", "yes"):
        kwargs["ssl"] = True
    return FalkorDB(**kwargs)


def connect(graph_name="pedigree"):
    graph_name = os.environ.get("PEDIGREE_GRAPH", graph_name)
    return connect_db().select_graph(graph_name)