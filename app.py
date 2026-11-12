"""
STORM: Social Stress Indicator. Streamlit mock-up of the planned Replit app.

Upload microblogs (text + time stamp) and an ISB file, pick the libraries,
press Run, and download the SSI CSV, the column dictionary, the QC summary
and the graphs. All calculations come from ssi_pipeline.py, unchanged.

Run:  streamlit run app.py
"""
from __future__ import annotations

import contextlib
import io
import tempfile
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

import ssi_pipeline as sp

HERE = Path(__file__).parent
DEMO_POSTS = HERE / "demo" / "demo_posts_2024.csv"
DEMO_ISB = HERE / "demo" / "demo_isb_2024.csv"

LIGHT_ENGINES = {"textblob": "TextBlob", "vader": "VADER"}
HEAVY_ENGINES = {"cryptobert": "CryptoBERT", "twitter_roberta": "Twitter-RoBERTa", "finbert": "FinBERT"}
FREQS = {"Auto-detect (recommended)": None, "Hourly": "h", "Daily": "D", "Weekly": "W", "Monthly": "MS"}

st.set_page_config(page_title="STORM SSI", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def save_upload(upload, folder: Path, name: str) -> Path:
    path = folder / name
    path.write_bytes(upload.getvalue())
    return path


@st.cache_data(show_spinner=False)
def header_of(path: str) -> list[str]:
    return pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()


@st.cache_data(show_spinner=False)
def preview_of(path: str, n: int = 5) -> pd.DataFrame:
    return pd.read_csv(path, nrows=n, encoding="utf-8-sig", dtype=str)


def guess(columns: list[str], candidates: list[str]) -> int:
    found = sp._pick(columns, None, candidates, "", required=False)
    return columns.index(found) if found else 0


def zip_outputs(out_dir: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out_dir.rglob("*")):
            if f.is_file() and "post_scores" not in f.parts:
                z.write(f, f.relative_to(out_dir))
    return buf.getvalue()


def issues_from(qc_text: str) -> list[str]:
    part = qc_text.split("## Issues flagged for review", 1)[-1]
    return [l[2:] for l in part.splitlines() if l.startswith("- ") and l[2:] != "none"]


if "work" not in st.session_state:
    st.session_state.work = Path(tempfile.mkdtemp(prefix="ssi_"))
work: Path = st.session_state.work

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("Social Stress Indicator")
st.caption("STORM · SSI = (−SA + (1 ± SUB) + ISB) / 3, after Combrink (2025). "
           "Prototype of the planned Replit app: the code stays on the server, users only upload data and download results.")

# --------------------------------------------------------------------------- #
# Sidebar: settings
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Settings")

    st.subheader("Libraries")
    engines = [k for k, label in LIGHT_ENGINES.items() if st.checkbox(label, value=True, key=f"eng_{k}")]
    for k, label in HEAVY_ENGINES.items():
        st.checkbox(label, value=False, disabled=True, key=f"eng_{k}",
                    help="BERT models need a GPU and several GB of memory. Planned for version 2.")
    st.caption("BERT models (greyed out) run on JupyterHub for now.")

    st.subheader("Timeline")
    freq_label = st.selectbox("Time unit", list(FREQS), index=0,
                              help="Auto-detect uses the most aggregated source: the microblogs or the ISB file.")
    use_window = st.checkbox("Limit to a date window", value=False)
    start = end = None
    if use_window:
        c1, c2 = st.columns(2)
        start = c1.date_input("From", value=date(2024, 1, 1))
        end = c2.date_input("To (included)", value=date(2024, 12, 31))

    st.subheader("Data quality")
    ssi_min_posts = st.number_input("Minimum posts per bucket for an SSI", 0, 1000, 20,
                                    help="Buckets with fewer posts keep their components but get no SSI.")
    st.caption("Data voids: collection gaps left empty, never interpolated.")
    voids = st.data_editor(pd.DataFrame({"from": pd.Series(dtype="datetime64[ns]"),
                                         "to": pd.Series(dtype="datetime64[ns]")}),
                           num_rows="dynamic", use_container_width=True, key="voids",
                           column_config={"from": st.column_config.DateColumn("From"),
                                          "to": st.column_config.DateColumn("To")})

# --------------------------------------------------------------------------- #
# Step 1 and 2: data
# --------------------------------------------------------------------------- #
st.subheader("1 · Data")
use_demo = st.toggle("Use the demo data (Bitcoin posts and Google Trends, 2024)", value=False)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Microblogs** (required)")
    if use_demo:
        posts_path = DEMO_POSTS
        st.success(f"Demo: {DEMO_POSTS.name}")
    else:
        up = st.file_uploader("CSV with a text column and a time stamp column", type="csv", key="posts")
        posts_path = save_upload(up, work, "posts.csv") if up else None
with c2:
    st.markdown("**ISB: search interest** (optional)")
    if use_demo:
        isb_path = DEMO_ISB
        st.success(f"Demo: {DEMO_ISB.name}")
    else:
        up = st.file_uploader("CSV with a date column and a value column (e.g. Google Trends)", type="csv", key="isb")
        isb_path = save_upload(up, work, "isb.csv") if up else None
    st.caption("Without ISB the components are still computed, but the SSI columns stay empty.")

text_col = time_col = None
if posts_path:
    cols = header_of(str(posts_path))
    c1, c2 = st.columns(2)
    text_col = c1.selectbox("Text column", cols, index=guess(cols, sp.TEXT_CANDIDATES))
    time_col = c2.selectbox("Time stamp column", cols, index=guess(cols, sp.TIME_CANDIDATES))
    with st.expander("Preview the microblogs"):
        st.dataframe(preview_of(str(posts_path))[[text_col, time_col]], use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# Step 3: run
# --------------------------------------------------------------------------- #
st.subheader("2 · Run")
ready = bool(posts_path) and bool(engines)
if not posts_path:
    st.info("Upload a microblog file, or switch on the demo data.")
elif not engines:
    st.warning("Pick at least one library in the sidebar.")

if st.button("Calculate SSI", type="primary", disabled=not ready):
    out_dir = work / "outputs"
    void_periods = [(str(pd.Timestamp(r["from"]).date()), str(pd.Timestamp(r["to"]).date()))
                    for _, r in voids.dropna().iterrows()]
    cfg = sp.Config(posts_csv=str(posts_path), isb_csv=str(isb_path) if isb_path else None,
                    out_dir=str(out_dir), engines=engines, force_freq=FREQS[freq_label],
                    start=str(start) if start else None, end=str(end) if end else None,
                    text_col=text_col, time_col=time_col, id_col=None,
                    ssi_min_posts=int(ssi_min_posts) or None, void_periods=void_periods)
    log = io.StringIO()
    with st.status("Scoring posts and building the SSI…", expanded=False) as status:
        try:
            with contextlib.redirect_stdout(log):
                table = sp.run(cfg)
            status.update(label="Done", state="complete")
            st.session_state.result = {"table": table, "out": out_dir, "log": log.getvalue()}
        except Exception as e:  # show the reason, not a traceback, to end users
            status.update(label="The run stopped", state="error")
            st.error(f"{type(e).__name__}: {e}")
            st.code(log.getvalue() or "(no log)")

# --------------------------------------------------------------------------- #
# Step 4: results
# --------------------------------------------------------------------------- #
res = st.session_state.get("result")
if res:
    table: pd.DataFrame = res["table"]
    out: Path = res["out"]
    csv_file = next(out.glob("ssi_*.csv"))
    qc_text = (out / "qc_summary.md").read_text()
    run_engines = [c[:-3] for c in table.columns if c.endswith("_SA")]
    unit = csv_file.stem.split("_", 1)[1]

    st.subheader("3 · Results")
    m = st.columns(4)
    m[0].metric("Timeline", unit)
    m[1].metric("Buckets", f"{len(table):,}")
    m[2].metric("Posts", f"{int(table['posts'].sum()):,}")
    m[3].metric("Libraries", len(run_engines))

    flags = issues_from(qc_text)
    if flags:
        with st.expander(f"⚠️ {len(flags)} issue(s) flagged for review", expanded=True):
            for f in flags:
                st.markdown(f"- {f}")

    tab_chart, tab_graphs, tab_data, tab_qc, tab_dl = st.tabs(
        ["SSI chart", "Graphs", "Data", "QC summary", "Downloads"])

    with tab_chart:
        c1, c2, c3 = st.columns([2, 2, 1])
        eng = c1.selectbox("Library", run_engines, format_func=lambda e: sp.ENGINE_LABELS.get(e, e))
        has_ssi = table[f"{eng}_SSI_pos"].notna().any()
        options = (["SSI_pos", "SSI_neg"] if has_ssi else []) + ["SA", "SUB", "ISB"]
        names = {"SSI_pos": "SSI positive adjusted", "SSI_neg": "SSI negative adjusted",
                 "SA": "Polarity (SA)", "SUB": "Subjectivity (SUB)", "ISB": "ISB"}
        pick = c2.multiselect("Series", options, default=options[:2], format_func=names.get)
        smooth = c3.number_input("Rolling mean", 1, 90, 7 if unit in ("daily", "hourly") else 1)
        if pick:
            data = pd.DataFrame({names[p]: table[f"{eng}_{p}"] for p in pick})
            data.index = data.index.tz_convert(None)
            st.line_chart(data.rolling(smooth, min_periods=1).mean() if smooth > 1 else data, height=380)
            if has_ssi:
                st.caption("Bands (Combrink 2025): low < 0.2, moderate 0.2–0.4, high > 0.4. Empty stretches are data voids or thin buckets.")
        if not has_ssi:
            st.info("No ISB file, so there is no SSI yet. The chart shows the components.")

    with tab_graphs:
        for png in sorted((out / "graphs").glob("*.png")):
            st.image(str(png), use_container_width=True)

    with tab_data:
        st.dataframe(table.reset_index(), use_container_width=True, height=420)
        st.markdown("**Column dictionary**")
        st.dataframe(pd.read_csv(out / "column_dictionary.csv"), use_container_width=True, hide_index=True)

    with tab_qc:
        st.markdown(qc_text)
        with st.expander("Run log"):
            st.code(res["log"])

    with tab_dl:
        st.download_button("⬇️ Everything (zip)", zip_outputs(out), file_name="ssi_outputs.zip",
                           mime="application/zip", type="primary")
        c1, c2, c3 = st.columns(3)
        c1.download_button(f"SSI table ({csv_file.name})", csv_file.read_bytes(), file_name=csv_file.name, mime="text/csv")
        c2.download_button("Column dictionary", (out / "column_dictionary.csv").read_bytes(),
                           file_name="column_dictionary.csv", mime="text/csv")
        c3.download_button("QC summary", qc_text.encode(), file_name="qc_summary.md", mime="text/markdown")
