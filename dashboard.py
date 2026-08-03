"""
Monitoring dashboard: five views over the query log.

Chart selection follows what this project learned to worry about rather
than what is easy to plot: filter coverage because metadata filtering
was the largest driver of retrieval quality, refusal rate because a
system that declines too readily fails as surely as one that fabricates,
and cold-start-separated latency because averaging the two hides which
component is slow.
"""
import json
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st

from monitoring.store import fetch_queries, fetch_feedback

st.set_page_config(page_title="Filings Analyst — Monitoring",
                   layout="wide")

st.title("Monitoring")

queries = pd.DataFrame(fetch_queries())
feedback = pd.DataFrame(fetch_feedback())

if queries.empty:
    st.info("No queries logged yet. Ask something in the app first.")
    st.stop()

queries["ts"] = pd.to_datetime(queries["ts"])
queries["date"] = queries["ts"].dt.date

# --- Headline numbers -------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Queries", len(queries))
warm = queries[queries["cold_start"] == 0]
c2.metric("Median latency (warm)",
          f"{warm['generation_ms'].median():.0f} ms" if not warm.empty else "—")
c3.metric("Refusal rate", f"{queries['refused'].mean():.0%}")
if not feedback.empty:
    positive = (feedback["rating"] > 0).mean()
    c4.metric("Positive feedback", f"{positive:.0%}",
              help=f"{len(feedback)} ratings")
else:
    c4.metric("Positive feedback", "—")

st.divider()

# --- 1. Query volume over time ----------------------------------------
st.subheader("Query volume")
st.caption("Grouped by hour.")
by_hour = (
    queries.set_index("ts").resample("h").size().reset_index(name="queries")
)
st.plotly_chart(
    px.bar(by_hour, x="ts", y="queries", labels={"ts": ""}),
    use_container_width=True,
)

# --- 2. Latency, cold vs warm -----------------------------------------
st.subheader("Latency")
st.caption(
    "Cold-start queries load the embedding model and build the keyword "
    "index; they are shown separately so they don't distort the typical case."
)
lat = queries.copy()
lat["start"] = lat["cold_start"].map({1: "cold", 0: "warm"})
st.plotly_chart(
    px.box(lat, x="start", y="generation_ms", points="all",
           labels={"generation_ms": "generation (ms)"}),
    use_container_width=True,
)

col_a, col_b = st.columns(2)

# --- 3. Filter coverage ------------------------------------------------
with col_a:
    st.subheader("Filter coverage")
    st.caption(
        "How constrained each search was. Zero filters means searching "
        "the whole corpus — evaluation showed metadata filtering to be "
        "the largest single driver of retrieval quality."
    )
    cov = queries["n_filters"].value_counts().sort_index().reset_index()
    cov.columns = ["filters inferred", "queries"]
    st.plotly_chart(
        px.bar(cov, x="filters inferred", y="queries"),
        use_container_width=True,
    )

# --- 4. Path routing ---------------------------------------------------
with col_b:
    st.subheader("Routing")
    st.caption(
        "Which path handled each question. The router is a keyword "
        "heuristic and is known to misroute causal questions containing "
        "words like 'increase'."
    )
    st.plotly_chart(
        px.pie(queries, names="path", hole=0.5),
        use_container_width=True,
    )

# --- 5. Outcomes -------------------------------------------------------
st.subheader("Outcomes")
outcomes = pd.Series({
    "answered": int(((queries["refused"] == 0) &
                     (queries["error"].isna())).sum()),
    "refused": int(queries["refused"].sum()),
    "error": int(queries["error"].notna().sum()),
}).reset_index()
outcomes.columns = ["outcome", "count"]
st.plotly_chart(
    px.bar(outcomes, x="outcome", y="count", color="outcome"),
    use_container_width=True,
)

# --- Recent queries ----------------------------------------------------
st.divider()
st.subheader("Recent queries")
st.dataframe(
    queries[["ts", "question", "path", "filters", "generation_ms",
             "refused"]].tail(20).iloc[::-1],
    use_container_width=True, hide_index=True,
)
