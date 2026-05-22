"""Streamlit dashboard for the AI code review agent."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from agent.models import LOW_CONFIDENCE_THRESHOLD
from agent.output import comments_to_markdown, run_to_json
from agent.pipeline import CodeReviewAgent


st.set_page_config(page_title="AI Code Review Agent", layout="wide")

st.title("AI Code Review Agent")

with st.sidebar:
    repo_url = st.text_input("GitHub repository", placeholder="psf/requests")
    use_llm = st.toggle("Use OpenAI LLM", value=True)
    max_chunks = st.slider("Max chunks to review", min_value=1, max_value=100, value=50)
    run_review = st.button("Run review", type="primary", use_container_width=True)

if run_review:
    if not repo_url.strip():
        st.error("Enter a GitHub repository URL or owner/repo shorthand.")
        st.stop()

    progress = st.empty()
    status_lines: list[str] = []

    def update(message: str) -> None:
        # Securely print all incoming streams to the background terminal stdout console
        print(f"[Agent Pipeline] {message}")
        
        # Guard clause: only update web UI text boxes for synchronous pipeline operations
        if "cloning" not in message.lower():
            status_lines.append(message)
            progress.code("\n".join(status_lines[-10:]))

    with st.spinner("Reviewing repository..."):
        agent = CodeReviewAgent(max_chunks=max_chunks, use_llm=use_llm)
        st.session_state["review_run"] = agent.review_repository(repo_url, progress_callback=update)

run = st.session_state.get("review_run")
if not run:
    st.info("Enter a public GitHub repository and run the review.")
    st.stop()

if run.errors:
    st.error(run.errors[0])
    st.stop()

for warning in run.warnings:
    st.warning(warning)

comments = run.comments
high_conf = run.high_confidence_comments
low_conf = run.low_confidence_comments

# =========================================================================
# 🔥 PIZZAZZ #1: REPO HEALTH SCORE GRAPHICAL BANNER & CALCULATION
# =========================================================================
severity_deductions = {
    "critical": 20,
    "high": 10,
    "medium": 5,
    "low": 2,
    "info": 0
}

health_score = 100
for c in comments:
    severity_type = str(c.severity).lower()
    health_score -= severity_deductions.get(severity_type, 0)

# Constraint score within bounds of 0 and 100
health_score = max(0, min(100, health_score))

if health_score >= 85:
    score_emoji = "🟢"
    score_status = "Excellent / Highly Maintainable"
    st.success(f"### {score_emoji} Repository Health Score: {health_score}/100 — **{score_status}**")
elif health_score >= 60:
    score_emoji = "🟡"
    score_status = "Needs Active Attention / Structural Concerns"
    st.warning(f"### {score_emoji} Repository Health Score: {health_score}/100 — **{score_status}**")
else:
    score_emoji = "🔴"
    score_status = "Critical Security or Runtime Stability Risks"
    st.error(f"### {score_emoji} Repository Health Score: {health_score}/100 — **{score_status}**")

# Visual feedback bar element
st.progress(health_score / 100)
st.markdown("---")

# =========================================================================
# 📊 RECONFIGURED COMPREHENSIVE METRICS GRID
# =========================================================================
metric_cols = st.columns(5)
metric_cols[0].metric("Chunks Reviewed", run.chunks_reviewed)
metric_cols[1].metric("Total Comments", len(comments))
metric_cols[2].metric("High Confidence", len(high_conf))
metric_cols[3].metric("Verify This", len(low_conf))

# Count syntax exceptions or parsing failures dynamically
syntax_breaks_count = sum(
    1 for c in comments 
    if "syntax" in str(c.title).lower() or (c.category == "bug" and c.severity == "high")
)
metric_cols[4].metric("Syntax Breaks", syntax_breaks_count)

st.markdown("---")

if not comments:
    st.success("No review comments were generated.")
    st.stop()

df = pd.DataFrame([c.to_dict() for c in comments])

filters = st.columns(3)
severity = filters[0].multiselect("Severity", sorted(df["severity"].unique()), default=sorted(df["severity"].unique()))
category = filters[1].multiselect("Category", sorted(df["category"].unique()), default=sorted(df["category"].unique()))
confidence_range = filters[2].slider("Confidence", 0, 100, (0, 100))

filtered = df[
    df["severity"].isin(severity)
    & df["category"].isin(category)
    & df["confidence"].between(confidence_range[0], confidence_range[1])
]

tab_review, tab_verify, tab_download = st.tabs(["Review", "Verify this", "Download"])

with tab_review:
    for row in filtered.sort_values(["severity", "file_path", "line"]).itertuples():
        label = f"{row.file_path}:{row.line} - {row.title}"
        with st.expander(label, expanded=row.severity in {"critical", "high"}):
            st.caption(f"Severity: {row.severity} | Category: {row.category} | Confidence: {row.confidence}% | Source: {row.source}")
            st.write(row.comment)
            st.info(row.suggestion)

with tab_verify:
    verify_df = filtered[filtered["confidence"] < LOW_CONFIDENCE_THRESHOLD]
    if verify_df.empty:
        st.success("No low-confidence comments in the current filter.")
    else:
        st.dataframe(
            verify_df[["file_path", "line", "severity", "category", "title", "confidence", "verify_label"]],
            use_container_width=True,
            hide_index=True,
        )

with tab_download:
    markdown = comments_to_markdown(comments, f"AI Code Review - {run.repo_name}")
    st.download_button("Download Markdown", markdown, file_name="review.md", mime="text/markdown")
    st.download_button("Download JSON", run_to_json(run), file_name="review.json", mime="application/json")