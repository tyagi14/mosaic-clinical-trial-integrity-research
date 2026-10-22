##############################################################################
# graph/graph_builder.py
#
# PURPOSE:
#   Wires the supervisor and the six specialist agents into one LangGraph
#   graph, compiles it once, and exposes it as `mosaic_graph`.
#
#   This file is the shape of the whole system. Everything else is a node.
#
# THE GRAPH:
#
#                            START
#                              │
#                       ┌──────▼───────┐
#                       │  supervisor  │   stamps the run, clears state
#                       └──────┬───────┘
#         ┌───────────┬────────┼────────┬───────────┬───────────┐
#         ▼           ▼        ▼        ▼           ▼           ▼
#   missing_      broken_    track_   pattern_   side_      timeline
#   results      promises    record    finder    effect
#         │           │        │        │           │           │
#         └───────────┴────────┼────────┴───────────┴───────────┘
#                              ▼
#                    ┌───────────────────┐
#                    │ supervisor_compile │  writes the brief
#                    └─────────┬─────────┘
#                              ▼
#                             END
#
# WHY FAN OUT TO ALL SIX, ALWAYS:
#   The supervisor does not choose specialists. Every run activates all
#   six, because they look for different things in the same data and
#   running six in parallel costs roughly what running one costs in wall
#   clock time. Selective routing would only create a way to miss a
#   signal type.
#
# HOW PARALLEL AGENTS AVOID OVERWRITING EACH OTHER:
#   All six nodes write to state["signals"] at the same time. That is
#   safe because of the reducer on that field in graph/state.py:
#
#       signals: Annotated[list[SignalOutput], _merge_lists]
#
#   LangGraph calls _merge_lists to APPEND each branch's output instead
#   of letting the last writer win. Without the reducer, five of the six
#   agents' findings would silently vanish.
#
# WHY supervisor_compile ONLY RUNS ONCE, AT THE END:
#   All six specialists have an edge into it, and LangGraph will not run
#   a node until every one of its incoming edges has been satisfied. So
#   it waits for the slowest agent, then runs a single time with the
#   complete set of signals.
#
# HOW TO USE:
#   from graph.graph_builder import mosaic_graph
#   result = await mosaic_graph.ainvoke(initial_state)
##############################################################################


from langgraph.graph import StateGraph, START, END

from graph.state import MosaicState

from agents.supervisor import supervisor_route, supervisor_compile
from agents.missing_results_agent import missing_results_node
from agents.broken_promises_agent import broken_promises_node
from agents.track_record_agent import track_record_node
from agents.pattern_finder_agent import pattern_finder_node
from agents.side_effect_agent import side_effect_node
from agents.timeline_agent import timeline_node

from config.logging_config import setup_logging

logger = setup_logging(__name__)


NODE_SUPERVISOR = "supervisor"
NODE_COMPILE    = "supervisor_compile"
# Node names are also the labels in LangSmith traces, so they are
# defined as constants rather than typed as strings in three places.

SPECIALIST_NODES = {
    "missing_results_agent":  missing_results_node,
    "broken_promises_agent":  broken_promises_node,
    "track_record_agent":     track_record_node,
    "pattern_finder_agent":   pattern_finder_node,
    "side_effect_agent":      side_effect_node,
    "timeline_agent":         timeline_node,
}
# The six specialists, mapped name → node function.
#
# Adding a seventh agent means adding one entry here. The fan-out and
# fan-in edges below are built from this dict, so nothing else in this
# file needs to change — and nothing anywhere else does either.
#
# Each key matches the AGENT_NAME constant inside that agent's module,
# which is also what it writes into agents_activated and into the
# procedures table. One name, used everywhere, so the API, the brief
# and the learned rules all line up.


def build_graph() -> StateGraph:
    """
    Builds the MOSAIC graph and returns it compiled and ready to invoke.

    Separated from the module-level `mosaic_graph` below so tests can
    build a fresh graph without relying on import order or on whatever
    state a shared instance has accumulated.

    Returns:
        The compiled graph. Call .ainvoke(initial_state) on it.
    """

    builder = StateGraph(MosaicState)
    # StateGraph needs the state schema so it knows which fields exist
    # and which reducer to apply to each one when branches merge.

    # ── NODES ─────────────────────────────────────────────────

    builder.add_node(NODE_SUPERVISOR, supervisor_route)
    builder.add_node(NODE_COMPILE, supervisor_compile)

    for name, node in SPECIALIST_NODES.items():
        builder.add_node(name, node)

    # ── EDGES ─────────────────────────────────────────────────

    builder.add_edge(START, NODE_SUPERVISOR)
    # Every run enters at the supervisor.

    for name in SPECIALIST_NODES:
        builder.add_edge(NODE_SUPERVISOR, name)
        # Six edges out of one node = fan-out. LangGraph sees six
        # independent branches and runs them concurrently.

        builder.add_edge(name, NODE_COMPILE)
        # Six edges into one node = fan-in. Compile waits for all six.

    builder.add_edge(NODE_COMPILE, END)
    # The brief is the last thing that happens.

    compiled = builder.compile()

    logger.info(
        f"MOSAIC graph compiled | "
        f"specialists={len(SPECIALIST_NODES)} | "
        f"nodes={list(SPECIALIST_NODES.keys())}"
    )

    return compiled


mosaic_graph = build_graph()
# Compiled ONCE when this module is first imported.
#
# Compiling is pure graph wiring — it opens no connections and makes no
# API calls — so doing it at import time is cheap and means the first
# request to /api/v1/analyze does not pay for it.
#
# api/dependencies.get_graph() returns this same object to every
# request, which is why it is not wrapped in lru_cache there.
