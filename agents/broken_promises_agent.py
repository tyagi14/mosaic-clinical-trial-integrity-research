##############################################################################
# agents/broken_promises_agent.py
#
# PURPOSE:
#   Detects outcome switching — when a clinical trial changes what it
#   promised to measure AFTER the study has already begun.
#
# WHY OUTCOME SWITCHING MATTERS:
#   Before a trial starts, sponsors must register exactly what they
#   plan to measure as the primary outcome.
#   Example: "Reduction in HbA1c at 26 weeks"
#
#   If the drug fails to show results on that measure, some sponsors
#   quietly change the primary outcome to something the drug DID work for.
#   This is called "outcome switching" — it is scientific fraud.
#   The published paper then reports success on the NEW outcome
#   while hiding that the original outcome was a failure.
#
#   This agent finds these cases by comparing what was originally
#   registered against what was actually measured and reported.
##############################################################################

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from graph.state import MosaicState, SignalOutput
from memory.procedural_store import ProceduralStore
from memory.episodic_store import EpisodicStore
from tools.search_tools import (
    search_studies_by_meaning,
    search_past_episodes,
    save_episode,
    get_sponsor_profile,
    update_sponsor_profile,
)
from tools.clinical_tools import fetch_study_details, get_study_amendments
from agents.signal_parser import parse_signals
# Shared across all six specialists — pulls the JSON signal blocks out
# of the model's final answer. See agents/signal_parser.py for why this
# is not a regex.

from agents.tool_executor import execute_tool
# Shared across all six specialists — runs the tool the model asked for
# and always returns text, never raises.

from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)

AGENT_NAME  = "broken_promises_agent"
SIGNAL_TYPE = "broken_promise"

MAX_ITERATIONS = 10
# How many reasoning turns this agent may take before the loop stops.
#
# One turn = one LLM call plus whatever tools it asked for. The ceiling
# exists because a model that keeps requesting tools would otherwise
# loop until the HTTP request timed out, spending tokens the whole way.
#
# Ten turns is comfortably more than a normal investigation needs —
# observed runs finish in three to five.

AGENT_TOOLS = [
    search_studies_by_meaning,
    search_past_episodes,
    save_episode,
    get_sponsor_profile,
    update_sponsor_profile,
    fetch_study_details,
    get_study_amendments,
    # get_study_amendments is unique to this agent —
    # it fetches the full protocol amendment history which contains
    # evidence of when and how outcomes were changed.
]

_procedural = ProceduralStore()
_episodic   = EpisodicStore()

_llm = ChatOpenAI(
    model=settings.openai_chat_model,
    temperature=0.1,
    api_key=settings.openai_api_key,
).bind_tools(AGENT_TOOLS)


async def broken_promises_node(state: MosaicState) -> dict:
    """
    Broken Promises Agent node — detects outcome switching in clinical trials.
    Runs in parallel with the other 5 specialist agents.
    """

    logger.info(f"{AGENT_NAME} | Starting analysis")

    try:
        procedures = await _procedural.get_procedures(AGENT_NAME)
        procedures_text = "\n".join(f"- {r}" for r in procedures)

        system_prompt = f"""You are the Broken Promises Agent for MOSAIC.

YOUR MISSION:
Detect outcome switching — cases where a clinical trial changed its
primary outcome measure AFTER the study began enrolling participants.

This is one of the most serious research integrity violations because
it allows sponsors to hide negative results by measuring something
different from what they originally promised.

YOUR REASONING RULES:
{procedures_text}

YOUR WORKFLOW:
1. Search past episodes for similar outcome switching cases
2. Search for studies with protocol amendments
3. For suspicious studies, fetch full amendment history with get_study_amendments
4. Compare the registered primary outcome against what was actually measured
5. Check the timing — did the change happen AFTER enrollment began?
6. Generate signals for confirmed cases with appropriate confidence

CONFIDENCE SCORING:
- 0.9+ : Primary outcome changed after >50% enrollment, no scientific justification
- 0.8  : Primary outcome changed mid-study, weak justification
- 0.7  : Secondary outcome changed to primary after enrollment
- 0.6  : Measurement method changed (not the outcome itself)
- Below 0.6: Ambiguous — send to human review

OUTPUT FORMAT:
{{
  "nct_id": "NCT_ID",
  "signal_type": "broken_promise",
  "summary": "What changed, when it changed, why it matters",
  "evidence": ["original outcome was X", "changed to Y after Z enrollment"],
  "confidence": 0.85
}}

If no violations found, say "NO_SIGNALS_FOUND".
"""

        task    = state.get("task", "Find trials with outcome switching")
        nct_ids = state.get("nct_ids", [])

        human_message = f"""
ANALYSIS TASK: {task}
SPECIFIC STUDIES: {nct_ids if nct_ids else "Search broadly"}

Investigate now. Focus on protocol amendments and primary outcome changes.
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message),
        ]

        signals_found  = []

        for iteration in range(MAX_ITERATIONS):
            response = await _llm.ainvoke(messages)
            messages.append(AIMessage(content=response.content or ""))

            if not response.tool_calls:
                logger.info(f"{AGENT_NAME} | Analysis complete | iteration={iteration+1}")
                signals_found = parse_signals(
                    response.content,
                    AGENT_NAME,
                    SIGNAL_TYPE,
                )
                break

            for tool_call in response.tool_calls:
                tool_result = await execute_tool(tool_call, AGENT_TOOLS)
                messages.append(
                    HumanMessage(content=f"Tool result for {tool_call['name']}:\n{tool_result}")
                )

        await _episodic.save_episode(
            agent_name=AGENT_NAME,
            content=f"Task: {task}. Found {len(signals_found)} broken promise signals.",
            outcome="signal_generated" if signals_found else "no_signal",
        )

        logger.info(f"{AGENT_NAME} | Complete | signals_found={len(signals_found)}")

        return {
            "signals":          state.get("signals", []) + signals_found,
            "agents_activated": state.get("agents_activated", []) + [AGENT_NAME],
        }

    except Exception as e:
        logger.error(f"{AGENT_NAME} | Error | {e}")
        return {
            "error_log":        state.get("error_log", []) + [f"{AGENT_NAME}: {str(e)}"],
            "agents_activated": state.get("agents_activated", []) + [AGENT_NAME],
        }
