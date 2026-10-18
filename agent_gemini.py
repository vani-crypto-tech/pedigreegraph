"""
PedigreeGraph agent, Gemini version.

Setup (PowerShell):
    pip install google-genai
    $env:GEMINI_API_KEY = "your-key-here"
    python agent_gemini.py

If you get a "model not found" error, set a current model name from AI Studio:
    $env:GEMINI_MODEL = "gemini-3.5-flash-lite"
"""
import json
import os

import agent_tools as at

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

SYSTEM = """You are PedigreeGraph, a family-history risk assistant. All families are SYNTHETIC demo data stored in a FalkorDB graph.

Rules:
1. Always use the tools. Never invent, estimate or calculate probabilities yourself; quote the tool results.
2. For a question about one person, call health_report. For a family-level question, call infer_inheritance first, then other tools as needed.
3. Back every claim with the family path or the person ids it came from (for example FAM1_P14 -> FAM1_P16).
4. Use cautious wording: "chance", "higher than average". Never say someone will or will not get a disease. This is not a diagnosis; suggest discussing results with a doctor or genetic counselor.
5. If asked, state the assumptions: single gene, complete penetrance, no new mutations, assumed population allele frequencies.
6. Name the condition (from the tool output) and phrase risks as conditions the family history puts someone at higher risk for.
7. Be concise: short answer first, then the evidence."""


def make_tools(on_tool=None):
    """Plain Python functions with type hints and docstrings: Gemini reads these as the tool definitions."""

    def log(name, **args):
        if on_tool:
            on_tool(name, args)

    def list_families() -> dict:
        """List the family ids stored in the graph."""
        log("list_families")
        return {"families": at.list_families()}

    def family_overview(family: str) -> dict:
        """Size, generations and affected members of a family.

        Args:
            family: Family id, for example FAM1.
        """
        log("family_overview", family=family)
        return at.run_tool("family_overview", {"family": family})

    def infer_inheritance(family: str) -> dict:
        """Decide whether the family's condition looks autosomal recessive or dominant (exact comparison).

        Args:
            family: Family id, for example FAM1.
        """
        log("infer_inheritance", family=family)
        return at.run_tool("infer_inheritance", {"family": family})

    def obligate_carriers(family: str) -> dict:
        """Graph query: unaffected parents of affected children, who must be recessive carriers.

        Args:
            family: Family id, for example FAM1.
        """
        log("obligate_carriers", family=family)
        return {"result": at.run_tool("obligate_carriers", {"family": family})}

    def find_consanguinity(family: str) -> dict:
        """Graph query: partners who share an ancestor, with the common ancestors.

        Args:
            family: Family id, for example FAM1.
        """
        log("find_consanguinity", family=family)
        return {"result": at.run_tool("find_consanguinity", {"family": family})}

    def couple_risks(family: str, mode: str = "") -> dict:
        """Exact chance that a future child of each couple is affected.

        Args:
            family: Family id, for example FAM1.
            mode: Optional, AR (recessive) or AD (dominant). Leave empty to infer it.
        """
        log("couple_risks", family=family, mode=mode)
        args = {"family": family}
        if mode:
            args["mode"] = mode
        return {"result": at.run_tool("couple_risks", args)}

    def testing_priority(family: str, mode: str = "", top: int = 5) -> dict:
        """Untested relatives ranked by risk, with the family path to the nearest affected relative.

        Args:
            family: Family id, for example FAM1.
            mode: Optional, AR (recessive) or AD (dominant). Leave empty to infer it.
            top: How many people to return.
        """
        log("testing_priority", family=family, mode=mode, top=top)
        args = {"family": family, "top": top}
        if mode:
            args["mode"] = mode
        return {"result": at.run_tool("testing_priority", args)}

    def health_report(person: str) -> dict:
        """Full family-history report for ONE person: pattern, personal risk, affected relatives,
        partner planning, family path and next steps. Use this for any question about a specific person.

        Args:
            person: Person id, for example FAM1_P14.
        """
        log("health_report", person=person)
        return at.run_tool("health_report", {"person": person})

    return [
        list_families, family_overview, infer_inheritance, obligate_carriers,
        find_consanguinity, couple_risks, testing_priority, health_report,
    ]


_CLIENT = None  # module-level, so the connection is not closed when make_chat returns


def make_chat(on_tool=None):
    global _CLIENT
    from google import genai
    from google.genai import types

    if _CLIENT is None:
        _CLIENT = genai.Client()  # reads GEMINI_API_KEY
    client = _CLIENT
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        tools=make_tools(on_tool),
    )
    return client.chats.create(model=MODEL, config=config)


if __name__ == "__main__":
    chat = make_chat(on_tool=lambda n, a: print(f"   [tool] {n} {json.dumps(a)}"))
    print("PedigreeGraph agent (Gemini, synthetic data). Type a question, or 'quit'.")
    print("Try: What is the risk picture for FAM1_P14?")
    print("Try: Which family has a dominant condition and which couples should worry?\n")
    while True:
        q = input("you> ").strip()
        if q.lower() in ("quit", "exit", ""):
            break
        resp = chat.send_message(q)
        print("\n" + (resp.text or "(no text returned)") + "\n")