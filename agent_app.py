"""Streamlit chat UI for the Walmart inventory-planning agent.

Wraps the Claude tool-use agent in agent_core.py in a chat interface that
also shows which tools were called (with inputs/outputs) at each step.

This deploy uses a "bring your own key" pattern: each visitor pastes their
own Anthropic API key into the sidebar, so the person hosting this app pays
nothing and no key sits in the deployment's secrets. Locally, export
ANTHROPIC_API_KEY (or add it to .streamlit/secrets.toml) and it'll pre-fill
the sidebar field for convenience.
"""
import os

import streamlit as st

from agent_core import run_agent_turn

st.set_page_config(page_title="Store Inventory Assistant", page_icon="🤖", layout="centered")


def _local_default_key() -> str:
    """Pre-fill the sidebar field from env/secrets for local dev convenience only."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except (FileNotFoundError, KeyError, st.errors.StreamlitSecretNotFoundError):
        return ""


st.title("🤖 Store Inventory Assistant")
st.caption(
    "Ask about demand and restocking for any store/department, e.g. "
    "\"Will Store 20 need more inventory for Department 5 next month?\""
)

with st.sidebar:
    st.header("Your Anthropic API key")
    api_key = st.text_input(
        "API key",
        value=_local_default_key(),
        type="password",
        placeholder="sk-ant-...",
        help="Get one at console.anthropic.com. Used only for your requests in this "
        "browser session — never stored or logged.",
        label_visibility="collapsed",
    )
    st.caption(
        "This app doesn't ship with a shared key, so each visitor uses their own. "
        "Get a key at [console.anthropic.com](https://console.anthropic.com)."
    )

if not api_key:
    st.info(
        "👈 Paste your Anthropic API key in the sidebar to start chatting. "
        "Don't have one? Create one at "
        "[console.anthropic.com](https://console.anthropic.com).",
        icon="🔑",
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
                result = run_agent_turn(st.session_state.agent_history, prompt, api_key=api_key)
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
        "It queries historical sales, runs a trained demand forecast, checks for "
        "recent anomalies, and reasons about whether a restock is needed — all "
        "grounded in real data via tool calls, shown above each answer."
    )
    if st.button("Clear conversation"):
        st.session_state.agent_history = []
        st.session_state.display_turns = []
        st.rerun()
