##############################################################################
# agents/tool_executor.py
#
# PURPOSE:
#   Runs the tool GPT-4o asked for and returns its output as text.
#
# WHY THIS IS ITS OWN MODULE:
#   Identical to the signal parser story — all six specialists had a
#   byte-for-byte copy of this function. One shared version means a fix
#   or an added safeguard lands for every agent at once.
#
# WHAT A TOOL CALL LOOKS LIKE:
#   When GPT-4o decides to use a tool, its response carries:
#
#       {"name": "check_results_posted", "args": {"nct_id": "NCT00011531"}}
#
#   This module looks the name up in the agent's own toolset, invokes it
#   with those arguments, and hands the result back as a string for the
#   next reasoning turn.
#
# WHY EVERY FAILURE COMES BACK AS A STRING AND NEVER RAISES:
#   The agent loop is a conversation. If a tool raises, the whole agent
#   node dies and every finding it had already gathered is lost. Handing
#   the error back as text keeps the conversation alive — the model reads
#   "Error: study not found", and tries a different trial or moves on.
#   An unreachable API costs one wasted turn instead of the entire run.
##############################################################################


from typing import Any

from evaluation.blind_mode import redact_if_blind
# Returns the tool's output untouched unless the run is a blind
# evaluation, in which case the pre-computed violation verdict and the
# embedded "RESULTS POSTED" label are stripped out. Applied here because
# this is the one function every specialist routes its tool calls
# through — see evaluation/blind_mode.py for why that matters.

from config.logging_config import setup_logging

logger = setup_logging(__name__)


MAX_RESULT_CHARS = 20_000
# Hard cap on how much text one tool result may add to the conversation.
#
# Tools return JSON built from live API responses, and a few of them can
# return a great deal of it. Ten reasoning turns of unbounded tool output
# would blow past the model's context window mid-run, which fails the
# whole node. Truncating one oversized result is always the better trade,
# and the message below tells the model plainly that it happened.


async def execute_tool(
    tool_call: dict[str, Any],
    available_tools: list,
) -> str:
    """
    Executes one tool call requested by an agent's LLM.

    Args:
        tool_call:       The tool call from the model response, with
                         "name" and "args" keys.
        available_tools: The AGENT_TOOLS list for the calling agent.
                         Scoped per agent on purpose — the timeline
                         agent cannot reach into the PubMed tools, so a
                         hallucinated tool name fails cleanly instead of
                         doing something unexpected.

    Returns:
        The tool's output as a string, or a readable error message. This
        function does not raise.
    """
    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {}) or {}

    tool_func = next(
        (t for t in available_tools if t.name == tool_name),
        None,
    )

    if tool_func is None:
        available = ", ".join(t.name for t in available_tools)
        logger.warning(f"Unknown tool requested | tool={tool_name!r}")
        return (
            f"Error: Tool '{tool_name}' is not available to you. "
            f"Your tools are: {available}"
        )
        # Listing the real tool names matters: it lets the model correct
        # itself on the next turn instead of retrying the same wrong
        # name until the iteration limit runs out.

    try:
        result = redact_if_blind(str(tool_func.invoke(tool_args)))
        # Redacted before the length check below, so the cap applies to
        # what the model actually receives.
    except Exception as e:
        logger.error(
            f"Tool execution failed | tool={tool_name} | error={e}"
        )
        return redact_if_blind(f"Error executing tool '{tool_name}': {e}")

    if len(result) > MAX_RESULT_CHARS:
        logger.warning(
            f"Tool result truncated | tool={tool_name} | "
            f"chars={len(result)} | limit={MAX_RESULT_CHARS}"
        )
        return (
            result[:MAX_RESULT_CHARS]
            + f"\n\n[... truncated at {MAX_RESULT_CHARS} characters. "
              f"Narrow your query if you need the rest.]"
        )

    return result
