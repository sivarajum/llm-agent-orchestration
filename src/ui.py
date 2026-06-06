"""Streamlit UI for the Agent Orchestration system."""

import logging
import os

import httpx
import streamlit as st

from src.settings import API_PORT

logger = logging.getLogger(__name__)

API_URL: str = os.getenv("API_URL", f"http://localhost:{API_PORT}")

st.set_page_config(page_title="Agent Orchestration", page_icon="", layout="wide")

st.title("Multi-Agent Orchestration System")
st.caption("Researcher -> Writer -> Reviewer pipeline powered by LangGraph")

# --- Sidebar ---
with st.sidebar:
    st.header("How It Works")
    st.markdown(
        "1. **Researcher** searches the web for information\n"
        "2. **Writer** produces a structured article\n"
        "3. **Reviewer** scores the draft (1-10)\n"
        "4. If score < 7, Writer revises (up to 3 iterations)\n"
    )
    st.divider()
    st.markdown("**Mode:** " + (
        "LLM-enhanced" if os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        else "Fallback (no API keys)"
    ))

# --- Main input ---
topic = st.text_input(
    "Enter a topic to research and write about:",
    placeholder="e.g., Quantum Computing breakthroughs in 2025",
)

if st.button("Run Pipeline", type="primary", disabled=not topic):
    # Progress stages
    stages = ["Researching", "Writing", "Reviewing", "Complete"]
    progress_bar = st.progress(0, text="Starting pipeline...")
    status_container = st.empty()

    with status_container.container():
        st.info("Pipeline running... This may take 15-30 seconds.")

    # Call the API
    try:
        progress_bar.progress(10, text="Sending request to agent pipeline...")
        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{API_URL}/run", json={"topic": topic})
            response.raise_for_status()
            data = response.json()
    except httpx.ConnectError:
        st.error(
            f"Cannot connect to API at {API_URL}. "
            "Make sure the API container is running."
        )
        st.stop()
    except Exception as e:
        st.error(f"API request failed: {e}")
        st.stop()

    progress_bar.progress(100, text="Pipeline complete!")
    status_container.empty()

    # --- Results display ---
    st.divider()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Status", data["status"].upper())
    with col2:
        st.metric("Iterations", data["iteration"])
    with col3:
        # Extract score from review feedback
        score_text = "N/A"
        if "Score:" in data.get("review_feedback", ""):
            score_text = data["review_feedback"].split("\n")[0].replace("Score: ", "")
        st.metric("Review Score", score_text)
    with col4:
        st.metric("Time (s)", data["elapsed_seconds"])

    st.divider()

    # Stage-by-stage output
    st.subheader("Stage 1: Research")
    with st.expander("View research results", expanded=True):
        st.markdown(data.get("research", "*No research output*"))

    st.subheader("Stage 2: Draft Article")
    with st.expander("View written draft", expanded=True):
        st.markdown(data.get("draft", "*No draft output*"))

    st.subheader("Stage 3: Review Feedback")
    with st.expander("View review feedback", expanded=True):
        st.markdown(data.get("review_feedback", "*No review output*"))

    # Agent flow visualization
    st.divider()
    st.subheader("Agent Flow")
    flow_cols = st.columns(4)
    stage_labels = ["Researcher", "Writer", "Reviewer", "Done"]
    stage_icons = ["[magnifying glass]", "[pen]", "[check]", "[flag]"]
    for i, col in enumerate(flow_cols):
        with col:
            st.markdown(
                f"**{stage_icons[i]} {stage_labels[i]}**\n\n"
                f"{'ACTIVE' if i < 3 and data['iteration'] > 0 else 'PASSED'}"
            )
