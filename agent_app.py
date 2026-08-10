"""Streamlit chat UI for the Walmart inventory-planning agent.

Wraps the Claude tool-use agent in agent_core.py in a chat interface that
also shows which tools were called (with inputs/outputs) at each step.
"""
import os

import streamlit as st

from agent_core import run_agent_turn

st.set_page_config(page_title="Store Inventory Assistant", page_icon="🤖", layout="centered")

# Streamlit Community Cloud users set this in .streamlit/secrets.toml; locally,
# export ANTHROPIC_API_KEY before running. st.secrets raises if no secrets.toml
# exists at all, so this has to be guarded rather than checked with `in`.
if "ANTHROPIC_API_KEY" not in os.environ:
    try:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    except (FileNotFoundError, KeyError, st.errors.StreamlitSecretNotFoundError):
        pass

st.title("🤖 Store Inventory Assistant")
st.caption(
    "Ask about demand and restocking for any store/department, e.g. "
    "\"Will Store 20 need more inventory for Department 5 next month?\""
)

if not os.environ.get("ANTHROPIC_API_KEY"):
    st.warning(
        "No `ANTHROPIC_API_KEY` found. Set it as an environment variable, or add it to "
        "`.streamlit/secrets.toml`, then reload this page.",
        icon="⚠️",
    )
    st.stop()

if "agent_history" not in st.session_state:
    st.session_state.agent_history = []  # Anthropic-format messages, carried across turns
if "display_turns" not in st.session_state:
    st.session_state.display_turns = []  # what we render in the chat


def render_tool_calls(tool_calls):
    with st.expander(f"🔧 {len(tool_calls)} tool call(s)", expanded=False):
        for call in tool_calls:
            st.markdown(f"**`{call.name}`**")
            col1, col2 = st.columns(2)
            with col1:
                st.caption("Input")
                st.json(call.input)
            with col2:
                st.caption("Output")
                st.json(call.output)


for turn in st.session_state.display_turns:
    with st.chat_message(turn["role"]):
        if turn.get("tool_calls"):
            render_tool_calls(turn["tool_calls"])
        st.markdown(turn["content"])

if prompt := st.chat_input("Ask about restocking..."):
    st.session_state.display_turns.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking sales data, forecast, and anomalies..."):
            try:
                result = run_agent_turn(st.session_state.agent_history, prompt)
            except Exception as exc:
                st.error(f"Agent error: {exc}")
                st.stop()

        if result.tool_calls:
            render_tool_calls(result.tool_calls)
        st.markdown(result.reply)

    st.session_state.agent_history = result.messages
    st.session_state.display_turns.append(
        {"role": "assistant", "content": result.reply, "tool_calls": result.tool_calls}
    )

with st.sidebar:
    st.header("About")
    st.write(
        "This assistant uses Claude (with tool use) to answer restocking questions. "
        "It calls into the same forecasting model and pipeline as the main "
        "[Sales Forecast app](app.py) — it queries historical sales, runs the trained "
        "demand forecast, checks for recent anomalies, and reasons about whether a "
        "restock is needed."
    )
    if st.button("Clear conversation"):
        st.session_state.agent_history = []
        st.session_state.display_turns = []
        st.rerun()
