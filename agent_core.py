"""Agentic loop: wires Claude tool-use around the forecasting tools in agent_tools.py.

This is a manual agentic loop (not the SDK's beta tool runner) so the
Streamlit UI can show exactly which tools were called, with what inputs
and outputs, at each step.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import anthropic

from agent_tools import TOOL_FUNCTIONS, TOOLS

MODEL = os.environ.get("AGENT_MODEL", "claude-opus-5")
MAX_TOKENS = 4096
MAX_TOOL_ITERATIONS = 6

SYSTEM_PROMPT = """You are an inventory planning assistant for Walmart store managers.

You help answer questions like "Will Store 20 need more inventory for Department 5
next month?" by grounding your answer in real data, not guesses. You have three tools:

- query_sales_history: recent historical weekly sales for a store/department
- forecast_demand: a trained model's forecast of upcoming weekly sales
- detect_anomalies: flags statistically unusual recent weeks (spikes/drops)

For any question about demand, restocking, or inventory for a specific store and
department, call query_sales_history and forecast_demand before answering, and call
detect_anomalies when you need to judge whether the recent trend is reliable (e.g. a
recent drop or spike could be a one-off, not a real trend shift). Do not fabricate
sales figures or trends — every number in your answer should come from a tool result.

When you have enough information, give a direct, clear answer for a store manager:
- State plainly whether a restock is likely needed (or not), and how confident you are.
- Ground the reasoning in the specific numbers: the recent trend, the forecast, and
  whether anomalies suggest the trend is noisy.
- Keep it concise — a manager should be able to read this in a few seconds and know
  what to do next.

If a store/department combination doesn't exist or has no data, say so plainly instead
of guessing."""


@dataclass
class ToolCallRecord:
    name: str
    input: dict
    output: dict


@dataclass
class AgentTurnResult:
    reply: str
    tool_calls: list = field(default_factory=list)
    messages: list = field(default_factory=list)


def _get_client(api_key: str = None) -> anthropic.Anthropic:
    key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not key:
        raise RuntimeError(
            "No Anthropic API key provided. Pass one in (e.g. from the Streamlit sidebar), "
            "or set the ANTHROPIC_API_KEY environment variable for local development."
        )
    return anthropic.Anthropic(api_key=key)


def run_agent_turn(
    history: list,
    user_message: str,
    api_key: str = None,
    max_iterations: int = MAX_TOOL_ITERATIONS,
) -> AgentTurnResult:
    """Run one user turn through the tool-use agentic loop.

    `history` is the prior conversation in Anthropic message format
    (list of {"role", "content"} dicts). `api_key`, if given, is used instead
    of the ANTHROPIC_API_KEY environment variable — this is how the Streamlit
    UI passes through a key a visitor typed in themselves. Returns the final
    text reply plus a trace of every tool call made along the way, and the
    updated message history to carry into the next turn.
    """
    client = _get_client(api_key)
    messages = [*history, {"role": "user", "content": user_message}]
    tool_calls: list = []

    for _ in range(max_iterations):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "refusal":
            return AgentTurnResult(
                reply="I'm not able to help with that request.",
                tool_calls=tool_calls,
                messages=messages,
            )

        if response.stop_reason != "tool_use":
            reply = "".join(block.text for block in response.content if block.type == "text")
            return AgentTurnResult(reply=reply, tool_calls=tool_calls, messages=messages)

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            func = TOOL_FUNCTIONS.get(block.name)
            if func is None:
                output = {"error": f"Unknown tool: {block.name}"}
            else:
                try:
                    output = func(**block.input)
                except Exception as exc:  # surface the failure to Claude instead of crashing
                    output = {"error": str(exc)}
            tool_calls.append(ToolCallRecord(name=block.name, input=block.input, output=output))
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(output),
                    "is_error": "error" in output,
                }
            )

        messages.append({"role": "user", "content": tool_results})

    return AgentTurnResult(
        reply="I wasn't able to reach a final answer within the allotted tool-call budget.",
        tool_calls=tool_calls,
        messages=messages,
    )
