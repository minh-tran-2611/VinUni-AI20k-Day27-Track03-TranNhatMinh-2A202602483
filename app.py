"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer,
                                  then the refined review for a final confirm
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from audit.replay import fetch_threads
from common.db import db_path, replay_events
from common.schemas import AuditEntry, ReviewState, risk_level_for
from exercises import exercise_4_audit as ex4


load_dotenv()

# The exercise-4 nodes log progress with rich ("→", "✓"). When the server's
# stdout is a non-UTF-8 pipe (e.g. cp1252 on Windows) that would raise
# UnicodeEncodeError mid-graph, so degrade unencodable characters instead.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

# Must be the first Streamlit command on the page.
st.set_page_config(page_title="HITL PR Review", layout="wide")


# ─── Graph: exercise 4 nodes + a final confirm after escalation ───────────
async def node_confirm_refined(state):
    """Show the refined (post-escalation) review for a final approve/reject/edit."""
    t0 = time.monotonic()
    a = state["analysis"]
    await ex4.audit_once(state, AuditEntry(
        agent_id=ex4.AGENT_ID,
        action="confirm_refined",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason=f"Refined review shown for final confirm: {len(a.comments)} comment(s)",
        execution_time_ms=ex4._elapsed_ms(t0),
    ))
    resp = interrupt({
        "kind": "approval_request",
        "stage": "refined",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
        "escalation_answers": state.get("escalation_answers") or {},
    })
    await ex4.audit(state, AuditEntry(
        agent_id=ex4.AGENT_ID,
        action="confirm_refined",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=ex4._reviewer_id(),
        decision=resp.get("choice") or "reject",
        reason=resp.get("feedback") or None,
        execution_time_ms=ex4._elapsed_ms(t0),
    ))
    return {"human_choice": resp.get("choice"), "human_feedback": resp.get("feedback")}


def build_graph(checkpointer):
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", ex4.node_fetch_pr), ("analyze", ex4.node_analyze), ("route", ex4.node_route),
        ("auto_approve", ex4.node_auto_approve), ("human_approval", ex4.node_human_approval),
        ("commit", ex4.node_commit), ("escalate", ex4.node_escalate),
        ("synthesize", ex4.node_synthesize), ("confirm_refined", node_confirm_refined),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "confirm_refined")
    g.add_edge("confirm_refined", "commit")
    g.add_edge("commit", END)
    return g.compile(checkpointer=checkpointer)


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None
if "error" not in st.session_state:
    st.session_state.error = None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}.

    Without `resume_value` this starts a new thread, or — for a thread that
    already has checkpoints — returns its pending interrupt / final state.
    An empty `pr_url` means "restore only": never start a new review.
    """
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        if resume_value is not None:
            return await app.ainvoke(Command(resume=resume_value), cfg)
        if not pr_url and not (await app.aget_state(cfg)).values:
            raise LookupError(f"No saved session for thread {thread_id}")
        return await ex4.start_or_resume(app, pr_url, thread_id)


def apply_result(result: dict) -> None:
    """Store a graph result (interrupt or final state) in the session."""
    st.session_state.pr_url = result.get("pr_url") or st.session_state.pr_url
    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
        st.session_state.final = None
    else:
        st.session_state.interrupt_payload = None
        st.session_state.final = result


def drive(pr_url: str, thread_id: str, resume_value=None, spinner: str = "Working...") -> None:
    """Run one graph step and keep any failure on screen instead of crashing the page."""
    st.session_state.error = None
    try:
        with st.spinner(spinner):
            apply_result(asyncio.run(run_graph(pr_url, thread_id, resume_value)))
    except Exception as e:  # missing API keys, GitHub 401/404, LLM errors, ...
        st.session_state.error = f"{type(e).__name__}: {e}"


def open_thread(thread_id: str | None, pr_url: str = "") -> None:
    """Make `thread_id` the current session (also kept in the URL for reloads)."""
    st.session_state.thread_id = thread_id
    st.session_state.pr_url = pr_url
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    st.session_state.error = None
    if thread_id:
        st.query_params["thread"] = thread_id
    else:
        st.query_params.clear()


# Reloading the page drops st.session_state; the ?thread=... query param
# brings the same session back from the SQLite checkpointer.
if st.session_state.thread_id is None and st.query_params.get("thread"):
    open_thread(st.query_params["thread"])
    drive("", st.session_state.thread_id, spinner="Restoring session...")
    if st.session_state.error:  # unknown thread: drop it from the URL, keep the message
        error = st.session_state.error
        open_thread(None)
        st.session_state.error = error


# ─── Page setup ────────────────────────────────────────────────────────────
st.title("HITL PR Review Agent")


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
RISK_ICON = {"low": "🟢", "med": "🟡", "high": "🔴"}

with st.sidebar:
    st.header("Recent sessions")
    if st.button("➕ New review", width="stretch"):
        open_thread(None)
        st.rerun()
    try:
        threads = asyncio.run(fetch_threads(limit=15))
    except Exception as e:
        threads = []
        st.caption(f"Could not read audit_events: {e}")
    if not threads:
        st.caption("No sessions yet — run a review.")
    for t in threads:
        pr_short = t["pr_url"].split("github.com/")[-1].replace("/pull/", " #")
        label = (f"{RISK_ICON.get(t['worst_risk'], '⚪')} {pr_short}\n\n"
                 f"{t['last_event'][:19].replace('T', ' ')} · {t['events']} events")
        current = t["thread_id"] == st.session_state.thread_id
        if st.button(label, key=f"thread_{t['thread_id']}", width="stretch",
                     type="primary" if current else "secondary", help=t["thread_id"]):
            open_thread(t["thread_id"], t["pr_url"])
            drive(t["pr_url"], t["thread_id"], spinner="Loading session...")
            st.rerun()


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review", type="primary")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None.

    Also used for the final confirm of a refined review after escalation.
    """
    conf = payload["confidence"]
    refined = payload.get("stage") == "refined"
    if refined:
        st.subheader(f"Refined review — final confirm (confidence {conf:.0%})")
    else:
        st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    if payload.get("escalation_answers"):
        with st.expander("Your answers to the escalation questions"):
            for q, a in payload["escalation_answers"].items():
                st.markdown(f"**{q}**  \n{a}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    key = f"{st.session_state.thread_id}_{payload.get('stage', 'initial')}"
    feedback = st.text_input("Feedback (optional — required for Edit)", key=f"feedback_{key}")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary", key=f"approve_{key}", width="stretch"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject", key=f"reject_{key}", width="stretch"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit", key=f"edit_{key}", width="stretch"):
        if not feedback.strip():
            st.warning("Write what should change in the feedback box, then click Edit.")
            return None
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("**Risk factors**\n" + "\n".join(f"- {r}" for r in payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form(f"escalation_{st.session_state.thread_id}"):
        st.markdown("**The agent needs answers to these questions before it can finish the review:**")
        answers: dict[str, str] = {}
        for i, q in enumerate(payload["questions"]):
            answers[q] = st.text_area(q, key=f"answer_{st.session_state.thread_id}_{i}", height=80)
        if st.form_submit_button("Submit answers", type="primary"):
            if any(not a.strip() for a in answers.values()):
                st.warning("Please answer every question (write 'unknown' if you don't know).")
                return None
            return {q: a.strip() for q, a in answers.items()}
    return None


def render_final(final: dict) -> None:
    action = final.get("final_action", "?")
    a = final.get("analysis")
    pr = final.get("pr_url") or st.session_state.pr_url
    if action.endswith("failed"):
        st.error(f"✗ {action} — the review could not be posted to GitHub "
                 "(check GITHUB_TOKEN scope and the server log).")
    elif action.startswith("auto"):
        st.success(f"✓ Auto-approved — confidence {a.confidence:.0%}, no human needed. "
                   "The agent already posted its review.")
        st.markdown(a.summary)
        st.link_button("View comment on GitHub", pr)
    elif action.startswith("committed"):
        st.success(f"✓ {action} — review posted (confidence {a.confidence:.0%})")
        st.link_button("View comment on GitHub", pr)
    elif action == "rejected":
        if final.get("human_choice") == "edit":
            st.warning("Edit requested — no comment posted. Feedback recorded in the audit trail: "
                       f"_{final.get('human_feedback')}_")
        else:
            st.warning("Rejected — no comment posted")
    else:
        st.info(f"final_action = {action}")


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    open_thread(str(uuid.uuid4()), pr_url)
    drive(pr_url, st.session_state.thread_id, spinner="Fetching PR + asking the LLM...")
    st.rerun()  # refresh the sidebar so the new session shows up

if st.session_state.error:
    st.error(st.session_state.error)

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        spinner = "Re-analysing with your answers..." if kind == "escalation" else "Resuming..."
        drive(st.session_state.pr_url, st.session_state.thread_id, resume_value=answer, spinner=spinner)
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    render_final(st.session_state.final)

if st.session_state.thread_id:
    with st.expander("Audit trail for this session"):
        events = asyncio.run(replay_events(st.session_state.thread_id))
        st.dataframe(
            [{k: e[k] for k in ("timestamp", "action", "confidence", "risk_level",
                                "decision", "reviewer_id", "execution_time_ms", "reason")}
             for e in events],
            width="stretch", hide_index=True,
        )
    st.caption(f"thread_id = {st.session_state.thread_id}  ·  replay: "
               f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`")
