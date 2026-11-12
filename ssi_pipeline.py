"""
STORM — Social Stress Indicator (SSI) pipeline
==============================================

Builds SSI time series from microblogs (X posts) and, when supplied, a Google
Trends information-seeking (ISB) series, following Combrink (2025), PLOS ONE
20(9): e0328768:

    SSI = (-SA + (1 - SUB) + ISB) / 3        -> "SSI_neg" (negative adjustment)
    SSI = (-SA + (1 + SUB) + ISB) / 3        -> "SSI_pos" (positive adjustment)

  SA  = mean sentiment polarity of the posts in a time bucket, -1 .. 1
  SUB = mean subjectivity of the posts in a time bucket, 0 .. 1
  ISB = search interest in the bucket relative to the series maximum, 0 .. 1

One set of columns is produced for every engine (TextBlob, VADER, BERT models).

Steps
  1. Load posts (and the ISB file, if any).
  2. Detect the timeline: the native resolution of each source (microblogs and
     ISB) is measured, and the coarsest ("most aggregated") one is used.
     Without ISB the microblogs decide, raised to default_freq if finer.
  3. Score every post with every engine (cached per engine, so reruns are fast).
  4. Average SA and SUB per time bucket; compute both SSIs.
  5. Write one CSV, a column dictionary, a QC summary and a set of graphs.

Usage (Jupyter):  import ssi_pipeline as sp; cfg = sp.Config(...); sp.run(cfg)
Usage (shell):    python ssi_pipeline.py --posts posts.csv --isb trends.csv --engines textblob vader
"""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    posts_csv: str
    out_dir: str = "ssi_outputs"
    isb_csv: str | None = None              # Google Trends file; None = no ISB
    isb_time_col: str | None = None         # auto-detected when None
    isb_value_col: str | None = None        # auto-detected when None
    engines: list[str] = field(default_factory=lambda: ["textblob", "vader"])
    force_freq: str | None = None           # 'h', 'D', 'W' or 'MS' to override detection
    default_freq: str = "D"                 # used when there is no ISB file
    start: str | None = None                # optional study window (UTC, inclusive)
    end: str | None = None
    text_col: str | None = None             # microblog text; auto-detected when None (text_clean, text, ...)
    time_col: str | None = None             # post time stamp; auto-detected when None (timestamp_utc, created_at, ...)
    id_col: str | None = None               # post id; auto-detected when None, row number if the file has none
    min_posts: int = 20                     # buckets with fewer posts are flagged 'thin'
    ssi_min_posts: int | None = 20          # SSI left empty for buckets with fewer posts; None = keep all
    # Data voids: collection gaps where the posts cannot be trusted. SA, SUB and SSI are left
    # empty inside them (no interpolation); ISB is kept, since Trends has no gap there.
    void_periods: list[tuple[str, str]] = field(default_factory=lambda: [("2023-06-14", "2023-10-12")])
    epsilon: float = 1e-3                   # smoothing constant in the SUB formula (Eq 2)
    subjectivity_model: str | None = None   # HF classifier for BERT engines; None = TextBlob SUB
    batch_size: int | None = None           # BERT batch; None = 256 on a GPU, 32 on a CPU
    max_length: int = 128                   # BERT tokens per post (longer posts are cut)
    checkpoint_minutes: float = 2           # BERT progress is saved this often; a rerun resumes
    device: int | str | None = None         # e.g. 0 or "cuda" for a GPU, "cpu"; None = auto
    smooth_window: int | None = None        # rolling mean for graphs; None = auto


# Hugging Face models behind the BERT engines. SA = P(positive) - P(negative).
HF_MODELS = {
    "cryptobert": "ElKulako/cryptobert",                              # Bearish / Neutral / Bullish
    "twitter_roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "finbert": "ProsusAI/finbert",
}
NEGATIVE_LABELS = {"negative", "bearish", "neg", "label_0"}
POSITIVE_LABELS = {"positive", "bullish", "pos", "label_2"}

FREQ_ORDER = ["min", "h", "D", "W", "MS"]          # finest -> coarsest
FREQ_NAMES = {"min": "minute", "h": "hourly", "D": "daily", "W": "weekly", "MS": "monthly"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
TEXT_CANDIDATES = ["text_clean", "text", "full_text", "tweet", "content", "body", "message", "post", "microblog"]
TIME_CANDIDATES = ["timestamp_utc", "timestamp", "created_at", "datetime", "date_time", "time", "date"]
ID_CANDIDATES = ["post_id", "tweet_id", "id", "status_id", "message_id"]


def _pick(columns: list[str], wanted: str | None, candidates: list[str], what: str, required: bool = True):
    """Use the column named in the config, else the first candidate present (case-insensitive)."""
    if wanted:
        if wanted not in columns:
            raise ValueError(f"{what} column '{wanted}' is not in the file. Columns: {columns}")
        return wanted
    lower = {c.lower().strip(): c for c in columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    if required:
        raise ValueError(f"Could not find a {what} column. Name it in the Config. Columns: {columns}")
    return None


def load_posts(cfg: Config) -> pd.DataFrame:
    header = pd.read_csv(cfg.posts_csv, nrows=0, encoding="utf-8-sig").columns.tolist()
    cfg.text_col = _pick(header, cfg.text_col, TEXT_CANDIDATES, "text")
    cfg.time_col = _pick(header, cfg.time_col, TIME_CANDIDATES, "time")
    cfg.id_col = _pick(header, cfg.id_col, ID_CANDIDATES, "id", required=False)
    cols = [c for c in (cfg.id_col, cfg.time_col, cfg.text_col) if c]
    print(f"Microblog columns: text = {cfg.text_col}, time = {cfg.time_col}, id = {cfg.id_col or '(row number)'}")
    df = pd.read_csv(cfg.posts_csv, dtype=str, keep_default_na=False,
                     encoding="utf-8-sig", usecols=cols)
    if not cfg.id_col:
        cfg.id_col = "row_id"
        df[cfg.id_col] = df.index.astype(str)
    df["ts"] = pd.to_datetime(df[cfg.time_col], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    if cfg.start:
        df = df[df["ts"] >= pd.Timestamp(cfg.start, tz="UTC")]
    if cfg.end:
        df = df[df["ts"] < pd.Timestamp(cfg.end, tz="UTC") + pd.Timedelta(days=1)]
    return df.reset_index(drop=True)


def load_isb(cfg: Config) -> pd.DataFrame | None:
    if not cfg.isb_csv:
        return None
    raw = pd.read_csv(cfg.isb_csv)
    tcol = cfg.isb_time_col or next(c for c in raw.columns
                                    if c.lower() in ("timestamp", "date", "time", "datetime"))
    if cfg.isb_value_col:
        vcol = cfg.isb_value_col
    else:  # prefer the stitched column from the Trends notebook, else the first numeric one
        preferred = [c for c in ("isb_daily", "trend_hourly_adjusted", "trend_daily_adjusted", "trend_daily_anchor")
                     if c in raw.columns]
        numeric = [c for c in raw.columns if c != tcol and pd.api.types.is_numeric_dtype(raw[c])]
        vcol = (preferred or numeric)[0]
    isb = pd.DataFrame({"ts": pd.to_datetime(raw[tcol], utc=True), "isb_raw": pd.to_numeric(raw[vcol])})
    print(f"ISB: {len(isb):,} rows from {Path(cfg.isb_csv).name} (time = {tcol}, value = {vcol})")
    return isb.dropna()


# --------------------------------------------------------------------------- #
# Timeline detection
# --------------------------------------------------------------------------- #
def native_freq(ts: pd.Series) -> str:
    """Resolution of a regular series, snapped to minute/hour/day/week/month."""
    diffs = pd.Series(np.sort(ts.unique())).diff().dropna()
    if diffs.empty:
        return "D"
    step = diffs.median()
    if step < pd.Timedelta(minutes=30):
        return "min"
    if step < pd.Timedelta(hours=12):
        return "h"
    if step < pd.Timedelta(days=4):
        return "D"
    if step < pd.Timedelta(days=20):
        return "W"
    return "MS"


def detect_timeline(cfg: Config, posts: pd.DataFrame, isb: pd.DataFrame | None) -> tuple[str, str]:
    """The coarsest ("most aggregated") resolution among the sources.

    Each source's native resolution is measured from its time stamps: posts
    scraped with a time of day are minute-level (event data); posts already
    stamped per day or week are daily or weekly. The coarser of microblogs and
    ISB is the timeline. Minute level is raised to hourly, the finest grid the
    SSI is built on. Without ISB, a minute or hourly post series is raised to
    default_freq, since one hour of posts is too few to average reliably.
    """
    post_freq = native_freq(posts["ts"])
    isb_freq = native_freq(isb["ts"]) if isb is not None else None
    found = f"microblogs {FREQ_NAMES[post_freq]}" + (f", ISB {FREQ_NAMES[isb_freq]}" if isb_freq else ", no ISB file")
    if cfg.force_freq:
        freq, why = cfg.force_freq, f"set by force_freq; {found}"
    elif isb_freq:
        freq = max(post_freq, isb_freq, key=FREQ_ORDER.index)
        source = "ISB" if FREQ_ORDER.index(isb_freq) > FREQ_ORDER.index(post_freq) else (
            "microblogs" if FREQ_ORDER.index(post_freq) > FREQ_ORDER.index(isb_freq) else "both sources")
        why = f"{found}; most aggregated: {source}"
    else:
        freq = max(post_freq, cfg.default_freq, key=FREQ_ORDER.index)
        why = f"{found}; default_freq {FREQ_NAMES[cfg.default_freq]} is the floor"
    freq = "h" if freq == "min" else freq
    print(f"Timeline: {FREQ_NAMES[freq]} ({why})")
    return freq, why


def bucket(ts: pd.Series, freq: str) -> pd.Series:
    if freq in ("min", "h", "D"):
        return ts.dt.floor(freq)
    if freq == "W":   # weeks start on Monday
        return (ts.dt.floor("D") - pd.to_timedelta(ts.dt.weekday, unit="D"))
    return ts.dt.tz_convert(None).dt.to_period("M").dt.start_time.dt.tz_localize("UTC")


def full_range(start: pd.Timestamp, end: pd.Timestamp, freq: str) -> pd.DatetimeIndex:
    if freq == "W":
        return pd.date_range(start, end, freq="W-MON")
    return pd.date_range(start, end, freq=freq)


# --------------------------------------------------------------------------- #
# Text preparation
# --------------------------------------------------------------------------- #
_MENTION = re.compile(r"@\w+")
_TAG = re.compile(r"[#$](\w+)")


def prepare_text(s: pd.Series) -> pd.Series:
    """Hashtags and cashtags keep their word (#bullish -> bullish); mentions become @user."""
    return (s.str.replace(_MENTION, "@user", regex=True)
             .str.replace(_TAG, r"\1", regex=True)
             .str.replace(r"\s+", " ", regex=True).str.strip())


# --------------------------------------------------------------------------- #
# Engines: each returns a frame with SA and SUB per post (NaN = not scored)
# --------------------------------------------------------------------------- #
def textblob_scores(texts: pd.Series, eps: float) -> pd.DataFrame:
    """TextBlob polarity, and subjectivity by Eq 2 of the paper, rescaled to 0..1.

    Eq 2: SUB(t) = mean over the post's subjective words l of
          log((S(l) + eps) / (1 - S(l) + eps)),  S = TextBlob word subjectivity.
    The log-odds runs from -K to K with K = log((1 + eps) / eps); it is mapped
    linearly to 0..1, so a word rated 0 gives 0, 0.5 gives 0.5 and 1 gives 1.
    Posts with no word in the lexicon have nothing to score and stay NaN for
    both SA and SUB, so they do not count as neutral zeros.
    """
    from textblob import TextBlob

    k = math.log((1 + eps) / eps)
    sa, sub = [], []
    for t in texts:
        blob = TextBlob(t)
        assessed = blob.sentiment_assessments.assessments
        if not assessed:
            sa.append(np.nan); sub.append(np.nan)
            continue
        sa.append(blob.sentiment.polarity)
        logodds = [math.log((s + eps) / (1 - s + eps)) for _, _, s, _ in assessed]
        sub.append((np.mean(logodds) + k) / (2 * k))
    return pd.DataFrame({"SA": sa, "SUB": sub}, index=texts.index)


def vader_scores(texts: pd.Series) -> pd.DataFrame:
    """VADER compound score (-1..1). VADER has no subjectivity; SUB is paired in later."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyser = SentimentIntensityAnalyzer()
    return pd.DataFrame({"SA": [analyser.polarity_scores(t)["compound"] for t in texts],
                         "SUB": np.nan}, index=texts.index)


def _hf_classify(texts: pd.Series, model_name: str, cfg: Config, to_score, checkpoint: Path) -> pd.Series:
    """Run a Hugging Face classifier over all texts, fast and resumable.

    - each distinct text is scored once (reposted copies share the score)
    - texts are sorted by length, so each batch pads to similar lengths
    - batches go straight to the model under inference mode (half precision on a GPU)
    - finished scores are saved to `checkpoint` every few minutes; a rerun continues from there
    """
    import os
    import time
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = cfg.device if cfg.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    device = f"cuda:{device}" if isinstance(device, int) else device
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval().to(device)
    if device.startswith("cuda"):
        model = model.half()
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]

    uniq = pd.Series(pd.unique(texts.values))
    done = pd.read_pickle(checkpoint) if checkpoint.exists() else pd.Series(dtype=float)
    scores = np.array(uniq.map(done), dtype=float)
    todo = np.flatnonzero(np.isnan(scores))
    todo = todo[np.argsort(uniq.str.len().to_numpy()[todo], kind="stable")]
    batch = cfg.batch_size if cfg.batch_size else (256 if device.startswith("cuda") else 32)
    print(f"  {model_name} on {device}: {len(texts):,} posts, {len(uniq):,} distinct texts, "
          f"{len(uniq) - len(todo):,} already done, labels {labels}")

    def save():
        pd.Series(scores, index=uniq.values).dropna().to_pickle(checkpoint)

    t0 = last_save = last_print = time.time()
    for n, i in enumerate(range(0, len(todo), batch), start=1):
        idx = todo[i:i + batch]
        enc = tok(uniq.iloc[idx].tolist(), padding=True, truncation=True,
                  max_length=cfg.max_length, return_tensors="pt").to(device)
        with torch.inference_mode():
            probs = model(**enc).logits.float().softmax(-1).cpu().numpy()
        scores[idx] = to_score(probs, labels)
        now = time.time()
        if now - last_save > 60 * cfg.checkpoint_minutes:
            save(); last_save = now
        if now - last_print > 30 or i + batch >= len(todo):
            k = min(i + batch, len(todo)); rate = k / (now - t0)
            print(f"  {k:,}/{len(todo):,} texts | {rate:,.0f}/s | about {(len(todo) - k) / rate / 60:,.0f} min left",
                  flush=True)
            last_print = now
    save()
    by_text = pd.Series(scores, index=uniq.values)
    return pd.Series(by_text.reindex(texts.values).values, index=texts.index)


def _polarity(probs: np.ndarray, labels: list[str]) -> np.ndarray:
    pos = [i for i, l in enumerate(labels) if l in POSITIVE_LABELS]
    neg = [i for i, l in enumerate(labels) if l in NEGATIVE_LABELS]
    if not pos or not neg:
        raise ValueError(f"Cannot tell positive from negative in labels {labels}")
    return probs[:, pos].sum(1) - probs[:, neg].sum(1)


def _subjective(probs: np.ndarray, labels: list[str]) -> np.ndarray:
    subj = [i for i, l in enumerate(labels) if "subj" in l]
    if not subj:
        raise ValueError(f"No subjective label in {labels}")
    return probs[:, subj].sum(1)


def hf_sentiment(texts: pd.Series, model_name: str, cfg: Config, checkpoint: Path) -> pd.DataFrame:
    """SA = P(positive or bullish) - P(negative or bearish), from a 3-class model."""
    return pd.DataFrame({"SA": _hf_classify(texts, model_name, cfg, _polarity, checkpoint), "SUB": np.nan})


def hf_subjectivity(texts: pd.Series, cfg: Config, checkpoint: Path) -> pd.Series:
    """P(subjective) from a binary subjectivity classifier (label containing 'subj')."""
    return _hf_classify(texts, cfg.subjectivity_model, cfg, _subjective, checkpoint)


def score_posts(posts: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """Score every post with every engine, caching each engine's results on disk."""
    cache = Path(cfg.out_dir) / "post_scores"
    cache.mkdir(parents=True, exist_ok=True)
    texts = prepare_text(posts[cfg.text_col])

    def cached(name, fn):
        path = cache / f"{name}.csv.gz"
        if path.exists():
            s = pd.read_csv(path, dtype={cfg.id_col: str}).set_index(cfg.id_col)
            if s.index.equals(pd.Index(posts[cfg.id_col])):
                print(f"{name}: loaded cached scores")
                return s.reset_index(drop=True)
        print(f"{name}: scoring {len(texts):,} posts")
        s = fn().reset_index(drop=True)
        s.assign(**{cfg.id_col: posts[cfg.id_col].values}).to_csv(path, index=False)
        return s

    # TextBlob SUB is the default subjectivity for engines that have none of their own
    needs_tb = "textblob" in cfg.engines or any(e != "textblob" for e in cfg.engines)
    tb = cached("textblob", lambda: textblob_scores(texts, cfg.epsilon)) if needs_tb else None
    hf_sub = None
    if cfg.subjectivity_model and any(e in HF_MODELS for e in cfg.engines):
        hf_sub = cached("subjectivity_model",
                        lambda: hf_subjectivity(texts, cfg, cache / "subjectivity_model.partial.pkl").to_frame("SUB"))["SUB"]

    scores = {}
    for eng in cfg.engines:
        if eng == "textblob":
            s = tb.copy()
        elif eng == "vader":
            s = cached("vader", lambda: vader_scores(texts)).assign(SUB=tb["SUB"])
        elif eng in HF_MODELS:
            s = cached(eng, lambda: hf_sentiment(texts, HF_MODELS[eng], cfg, cache / f"{eng}.partial.pkl"))
            s["SUB"] = hf_sub if hf_sub is not None else tb["SUB"]
        else:
            raise ValueError(f"Unknown engine '{eng}'. Choose from textblob, vader, {', '.join(HF_MODELS)}")
        scores[eng] = s
    return scores


# --------------------------------------------------------------------------- #
# Aggregation and SSI
# --------------------------------------------------------------------------- #
def build_table(posts: pd.DataFrame, scores: dict[str, pd.DataFrame],
                isb: pd.DataFrame | None, freq: str, cfg: Config) -> pd.DataFrame:
    b = bucket(posts["ts"], freq).rename("time")
    start = b.min() if isb is None else min(b.min(), bucket(isb["ts"], freq).min())
    end = b.max() if isb is None else max(b.max(), bucket(isb["ts"], freq).max())
    if cfg.start:
        start = bucket(pd.Series([pd.Timestamp(cfg.start, tz="UTC")]), freq).iloc[0]
    if cfg.end:
        last = pd.Timestamp(cfg.end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        end = bucket(pd.Series([last]), freq).iloc[0]
    idx = full_range(start, end, freq)

    # ISB: mean per bucket, then relative to the highest value in the series (0..1)
    if isb is not None:
        isb_b = isb.groupby(bucket(isb["ts"], freq))["isb_raw"].mean().reindex(idx)
        isb_b = isb_b / isb_b.max()
    else:
        isb_b = pd.Series(np.nan, index=idx)

    n_posts = posts.groupby(b).size().reindex(idx).fillna(0).astype(int)
    void = pd.Series(False, index=idx)
    for v_start, v_end in cfg.void_periods or []:
        lo = pd.Timestamp(v_start, tz="UTC")
        hi = pd.Timestamp(v_end, tz="UTC") + pd.Timedelta(days=1)
        void |= (idx >= lo) & (idx < hi)
    too_few = n_posts < (cfg.ssi_min_posts or 0)

    table = pd.DataFrame(index=idx)
    table.index.name = "time"
    for eng, s in scores.items():
        g = s.groupby(b)
        sa = g["SA"].mean().reindex(idx).mask(void)
        sub = g["SUB"].mean().reindex(idx).mask(void)
        table[f"{eng}_SSI_pos"] = ((-sa + (1 + sub) + isb_b) / 3).mask(too_few)
        table[f"{eng}_SSI_neg"] = ((-sa + (1 - sub) + isb_b) / 3).mask(too_few)
        table[f"{eng}_SUB"] = sub
        table[f"{eng}_SA"] = sa
        table[f"{eng}_1_minus_SUB"] = 1 - sub
        table[f"{eng}_1_plus_SUB"] = 1 + sub
        table[f"{eng}_ISB"] = isb_b              # repeated per library, as the spec lists "ISB only" per library
        table[f"{eng}_posts_scored"] = g["SA"].count().reindex(idx).fillna(0).astype(int)
    table["ISB"] = isb_b
    table["posts"] = n_posts
    table["void"] = void.values
    table["thin"] = (table["posts"] > 0) & (table["posts"] < cfg.min_posts)
    table["no_posts"] = table["posts"] == 0
    return table


# --------------------------------------------------------------------------- #
# Graphs
# --------------------------------------------------------------------------- #
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ENGINE_LABELS = {"textblob": "TextBlob", "vader": "VADER", "cryptobert": "CryptoBERT",
                 "twitter_roberta": "Twitter-RoBERTa", "finbert": "FinBERT"}


def _style(ax, ylabel):
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)


def _smooth(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=max(1, window // 3), center=True).mean()


def _auto_window(freq: str, n: int) -> int:
    base = {"h": 24 * 7, "D": 30, "W": 8, "MS": 3}.get(freq, 1)
    return max(1, min(base, n // 20))


def make_graphs(table: pd.DataFrame, freq: str, cfg: Config) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 11,
                         "axes.titleweight": "bold", "axes.titlecolor": INK})
    out = Path(cfg.out_dir) / "graphs"
    out.mkdir(parents=True, exist_ok=True)
    w = cfg.smooth_window or _auto_window(freq, len(table))
    unit = FREQ_NAMES[freq]
    smooth_note = f"{w}-period rolling mean" if w > 1 else "no smoothing"
    has_isb = table["ISB"].notna().any()
    files = []

    def raw_and_smooth(ax, s, color, label):
        ax.plot(s.index, s, color=color, linewidth=0.6, alpha=0.25)
        ax.plot(s.index, _smooth(s, w), color=color, linewidth=2, label=label)

    def shade_voids(ax):
        for v_start, v_end in cfg.void_periods or []:
            ax.axvspan(pd.Timestamp(v_start, tz="UTC"), pd.Timestamp(v_end, tz="UTC") + pd.Timedelta(days=1),
                       color=GRID, alpha=0.7, linewidth=0, zorder=0)

    def finish(fig, name):
        path = out / name
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        files.append(path)

    engines = [c[:-len("_SA")] for c in table.columns if c.endswith("_SA")]
    for eng in engines:
        label = ENGINE_LABELS.get(eng, eng)
        panels = (["ssi"] if has_isb else []) + ["sa", "sub"] + (["isb"] if has_isb else [])
        fig, axes = plt.subplots(len(panels), 1, figsize=(12, 2.4 * len(panels)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, p in zip(axes, panels):
            if p == "ssi":
                raw_and_smooth(ax, table[f"{eng}_SSI_pos"], SERIES[0], "SSI positive adjusted (1 + SUB)")
                raw_and_smooth(ax, table[f"{eng}_SSI_neg"], SERIES[1], "SSI negative adjusted (1 − SUB)")
                for y in (0.2, 0.4):
                    ax.axhline(y, color=INK_2, linewidth=0.8, linestyle=(0, (3, 3)))
                for y, name in ((0.4, "high"), (0.2, "moderate")):
                    ax.text(1.005, y, name, transform=ax.get_yaxis_transform(), color=INK_2,
                            fontsize=8, va="center", ha="left", clip_on=False)
                ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
                _style(ax, "SSI")
                ax.set_title(f"{label}: Social Stress Indicator ({unit})", loc="left")
            elif p == "sa":
                raw_and_smooth(ax, table[f"{eng}_SA"], SERIES[0], "SA")
                ax.axhline(0, color=INK_2, linewidth=0.8)
                _style(ax, "Polarity (SA)")
                if not has_isb:
                    ax.set_title(f"{label}: sentiment and subjectivity ({unit})", loc="left")
            elif p == "sub":
                raw_and_smooth(ax, table[f"{eng}_SUB"], SERIES[2], "SUB")
                _style(ax, "Subjectivity (SUB)")
            else:
                raw_and_smooth(ax, table["ISB"], SERIES[6], "ISB")
                _style(ax, "ISB (0–1)")
        for ax in axes:
            shade_voids(ax)
        axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
        axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator()))
        fig.text(0.01, -0.01, f"Thin lines: raw {unit} values. Bold lines: {smooth_note}. "
                 f"Grey band: data void (collection gap, left empty). "
                 + (f"SSI is empty on {unit} buckets with fewer than {cfg.ssi_min_posts} posts."
                    if cfg.ssi_min_posts and has_isb else "Gaps are periods with no posts."),
                 color=INK_2, fontsize=8)
        fig.tight_layout()
        finish(fig, f"{eng}_components.png")

    # Engines side by side, one panel per measure
    if len(engines) > 1:
        measures = ([("SSI_neg", "SSI negative adjusted"), ("SSI_pos", "SSI positive adjusted")] if has_isb else []) \
                   + [("SA", "Polarity (SA)"), ("SUB", "Subjectivity (SUB)")]
        fig, axes = plt.subplots(len(measures), 1, figsize=(12, 2.4 * len(measures)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, (m, title) in zip(axes, measures):
            # engines that share a series (e.g. VADER using TextBlob SUB) are drawn once
            groups: list[list[str]] = []
            for eng in engines:
                same = next((g for g in groups if table[f"{g[0]}_{m}"].equals(table[f"{eng}_{m}"])), None)
                if same:
                    same.append(eng)
                else:
                    groups.append([eng])
            for g in groups:
                ax.plot(table.index, _smooth(table[f"{g[0]}_{m}"], w), color=SERIES[engines.index(g[0])],
                        linewidth=2, label=" = ".join(ENGINE_LABELS.get(e, e) for e in g))
            shade_voids(ax)
            if len(groups) < len(engines):
                ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=len(groups))
            if m == "SA":
                ax.axhline(0, color=INK_2, linewidth=0.8)
            _style(ax, title)
        axes[0].legend(loc="upper left", fontsize=8, frameon=False, ncol=len(engines))
        axes[0].set_title(f"Engines compared ({unit}, {smooth_note})", loc="left")
        axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator()))
        fig.tight_layout()
        finish(fig, "engines_compared.png")

    # Posts per bucket, so readers can see where the index rests on few posts
    fig, ax = plt.subplots(figsize=(12, 2.4))
    ax.bar(table.index, table["posts"], width={"h": 1 / 24, "D": 1, "W": 6, "MS": 25}.get(freq, 1),
           color=SERIES[0], linewidth=0)
    thin = table[table["thin"]]
    ax.scatter(thin.index, thin["posts"], s=10, color=SERIES[1], zorder=3, label=f"thin (< {cfg.min_posts} posts)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    _style(ax, "Posts")
    ax.set_title(f"Posts per {unit.rstrip('ly') if unit != 'daily' else 'day'} bucket", loc="left")
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()
    finish(fig, "posts_per_bucket.png")
    return files


# --------------------------------------------------------------------------- #
# Column dictionary and QC summary (for reviewers)
# --------------------------------------------------------------------------- #
def column_dictionary(table: pd.DataFrame, unit: str) -> pd.DataFrame:
    per_engine = {
        "SSI_pos": "SSI positive adjusted: (-SA + (1 + SUB) + ISB) / 3. Empty in data voids and thin buckets.",
        "SSI_neg": "SSI negative adjusted: (-SA + (1 - SUB) + ISB) / 3. Empty in data voids and thin buckets.",
        "SUB": f"Subjectivity only: mean post subjectivity in the {unit} bucket, 0..1.",
        "SA": f"Polarity only: mean post sentiment polarity in the {unit} bucket, -1..1.",
        "1_minus_SUB": "1 - SUB, the term used in SSI_neg.",
        "1_plus_SUB": "1 + SUB, the term used in SSI_pos.",
        "ISB": "ISB only: search interest relative to the series maximum, 0..1 (same for every library).",
        "posts_scored": "Posts in the bucket this library could score (TextBlob skips posts with no lexicon word).",
    }
    shared = {
        "time": f"Start of the {unit} bucket (UTC). One row per bucket, no gaps.",
        "ISB": "Search interest relative to the series maximum, 0..1 (Google Trends).",
        "posts": "All posts in the bucket.",
        "void": "True inside a data void (collection gap): SA, SUB and SSI left empty, never interpolated.",
        "thin": "True when the bucket has some posts but fewer than min_posts.",
        "no_posts": "True when the bucket has no posts.",
    }
    rows = []
    for c in ["time"] + list(table.columns):
        for key in sorted(per_engine, key=len, reverse=True):   # longest first: 1_minus_SUB before SUB
            if c not in shared and c.endswith("_" + key):
                eng = c[: -len(key) - 1]
                rows.append((c, ENGINE_LABELS.get(eng, eng), per_engine[key]))
                break
        else:
            rows.append((c, "all", shared.get(c, "")))
    return pd.DataFrame(rows, columns=["column", "library", "meaning"])


def qc_summary(table: pd.DataFrame, posts: pd.DataFrame, scores: dict, freq: str, why: str, cfg: Config) -> str:
    """Plain checks a reviewer can read without running code. Flags issues; fixes nothing."""
    unit = FREQ_NAMES[freq]
    engines = list(scores)
    has_isb = table["ISB"].notna().any()
    lines, flags = [], []
    f = lambda x: "" if pd.isna(x) else f"{x:.3f}"

    lines += ["# SSI pipeline: QC summary", "",
              f"- Microblogs: {Path(cfg.posts_csv).name}, {len(posts):,} posts, "
              f"{posts['ts'].min():%Y-%m-%d} to {posts['ts'].max():%Y-%m-%d} (text = `{cfg.text_col}`, time = `{cfg.time_col}`)",
              f"- ISB: {Path(cfg.isb_csv).name if cfg.isb_csv else 'none'}",
              f"- Timeline: **{unit}** ({why})",
              f"- Engines: {', '.join(ENGINE_LABELS.get(e, e) for e in engines)}",
              f"- Data voids: {', '.join(f'{a} to {b}' for a, b in cfg.void_periods) or 'none'}; "
              f"SSI needs at least {cfg.ssi_min_posts or 0} posts per bucket", ""]

    idx = table.index
    steps = pd.Series(idx).diff().dropna()
    lines += ["## Timeline", "",
              "| check | result |", "|---|---|",
              f"| buckets | {len(idx):,} ({idx.min():%Y-%m-%d} to {idx.max():%Y-%m-%d}) |",
              f"| duplicate buckets | {idx.duplicated().sum()} |",
              f"| missing buckets | {0 if len(idx) == len(full_range(idx.min(), idx.max(), freq)) else 'yes'} |",
              f"| buckets with no posts | {int(table['no_posts'].sum()):,} |",
              f"| thin buckets (< {cfg.min_posts} posts) | {int(table['thin'].sum()):,} |",
              f"| buckets in data voids | {int(table['void'].sum()):,} |",
              f"| posts counted in buckets | {int(table['posts'].sum()):,} of {len(posts):,} |", ""]
    if int(table["posts"].sum()) != len(posts):
        flags.append("Not every post landed in a bucket (check start/end).")

    lines += ["## Per library", "",
              "| library | posts scored | SA min..max | SUB min..max | SSI_pos min..max | SSI_neg min..max | "
              "formula max error | SSI_pos low/mod/high | SSI_neg low/mod/high |",
              "|---|---|---|---|---|---|---|---|---|"]
    for e in engines:
        sa, sub, isb = table[f"{e}_SA"], table[f"{e}_SUB"], table[f"{e}_ISB"]
        ok = table[f"{e}_SSI_pos"].notna()
        err = max(((-sa + 1 + sub + isb) / 3 - table[f"{e}_SSI_pos"])[ok].abs().max(),
                  ((-sa + 1 - sub + isb) / 3 - table[f"{e}_SSI_neg"])[ok].abs().max()) if ok.any() else np.nan

        def bands(s):
            s = s.dropna()
            if s.empty:
                return ""
            return f"{(s < .2).mean():.0%} / {((s >= .2) & (s <= .4)).mean():.0%} / {(s > .4).mean():.0%}"
        share = scores[e]["SA"].notna().mean()
        lines.append(f"| {ENGINE_LABELS.get(e, e)} | {share:.1%} | {f(sa.min())}..{f(sa.max())} | "
                     f"{f(sub.min())}..{f(sub.max())} | {f(table[f'{e}_SSI_pos'].min())}..{f(table[f'{e}_SSI_pos'].max())} | "
                     f"{f(table[f'{e}_SSI_neg'].min())}..{f(table[f'{e}_SSI_neg'].max())} | "
                     f"{'' if pd.isna(err) else f'{err:.1e}'} | {bands(table[f'{e}_SSI_pos'])} | {bands(table[f'{e}_SSI_neg'])} |")
        if not pd.isna(err) and err > 1e-5:
            flags.append(f"{e}: SSI does not match its components (max error {err:.1e}).")
        if share < 0.9:
            flags.append(f"{ENGINE_LABELS.get(e, e)} scores only {share:.0%} of posts; the rest have no score and are left out of the averages.")
        if ok.any():
            pos_high = (table[f"{e}_SSI_pos"].dropna() > .4).mean()
            neg_low = (table[f"{e}_SSI_neg"].dropna() < .2).mean()
            if pos_high > .8 or neg_low > .8:
                flags.append(f"{ENGINE_LABELS.get(e, e)}: the 0.2/0.4 bands barely separate days "
                             f"(SSI_pos high on {pos_high:.0%} of days, SSI_neg low on {neg_low:.0%}).")
    lines.append("")
    lines.append("Bands: low < 0.2, moderate 0.2 to 0.4, high > 0.4 (Combrink 2025). "
                 "Formula error: SSI recomputed from the SA, SUB and ISB columns.")
    lines.append("")

    void = table["void"]
    blank_ok = table.loc[void, [c for c in table.columns if c.endswith(("_SSI_pos", "_SSI_neg", "_SA", "_SUB"))]].notna().sum().sum() == 0
    if not blank_ok:
        flags.append("Some SA, SUB or SSI values are filled inside a data void.")

    if len(engines) > 1:
        same_sub = [e for e in engines[1:] if table[f"{e}_SUB"].equals(table[f"{engines[0]}_SUB"])]
        if same_sub:
            flags.append(f"Subjectivity is the same series for {', '.join(ENGINE_LABELS.get(e, e) for e in [engines[0]] + same_sub)}: "
                         "only TextBlob gives a subjectivity score, so the others borrow it "
                         "(set subjectivity_model to use a separate classifier).")
        m = "SSI_neg" if has_isb else "SA"
        corr = table[[f"{e}_{m}" for e in engines]].corr()
        corr.index = corr.columns = [ENGINE_LABELS.get(e, e) for e in engines]
        lines += [f"## Agreement between libraries ({m}, correlation of {unit} values)", "",
                  "| | " + " | ".join(corr.columns) + " |", "|---" * (len(corr) + 1) + "|"]
        lines += [f"| {r} | " + " | ".join(f"{v:.2f}" for v in corr.loc[r]) + " |" for r in corr.index]
        lines.append("")

    if has_isb:
        isb = table["ISB"]
        lines += ["## ISB", "",
                  f"- range {f(isb.min())} to {f(isb.max())}, maximum on {isb.idxmax():%Y-%m-%d}, {int(isb.isna().sum())} empty buckets",
                  "- yearly mean: " + ", ".join(f"{y}: {v:.3f}" for y, v in isb.groupby(isb.index.year).mean().items()), ""]
        low_years = [y for y, v in isb.groupby(isb.index.year).mean().items() if v < 0.05]
        if low_years:
            flags.append(f"ISB averages below 0.05 in {', '.join(map(str, low_years))} (scaled to the series peak), "
                         "so it adds almost nothing to the SSI in those years.")
    else:
        flags.append("No ISB file: SSI columns are empty; only the components are filled.")

    lines += ["## Issues flagged for review", ""] + ([f"- {x}" for x in flags] or ["- none"])
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run(cfg: Config) -> pd.DataFrame:
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    posts = load_posts(cfg)
    print(f"Posts: {len(posts):,} ({posts['ts'].min():%Y-%m-%d} to {posts['ts'].max():%Y-%m-%d})")
    isb = load_isb(cfg)
    freq, why = detect_timeline(cfg, posts, isb)
    scores = score_posts(posts, cfg)
    for eng, s in scores.items():
        print(f"  {eng}: {s['SA'].notna().mean():.1%} of posts scored")
    table = build_table(posts, scores, isb, freq, cfg)

    csv_path = Path(cfg.out_dir) / f"ssi_{FREQ_NAMES[freq]}.csv"
    table.reset_index().to_csv(csv_path, index=False, float_format="%.6f")
    print(f"CSV: {csv_path} ({len(table):,} rows, {table['no_posts'].sum():,} with no posts, "
          f"{table['thin'].sum():,} thin, {table['void'].sum():,} in data voids)")
    if table["ISB"].isna().all():
        print("No ISB series: SSI columns stay empty; the components (SA, SUB, 1 ± SUB) are filled.")
    dict_path = Path(cfg.out_dir) / "column_dictionary.csv"
    column_dictionary(table, FREQ_NAMES[freq]).to_csv(dict_path, index=False)
    qc_path = Path(cfg.out_dir) / "qc_summary.md"
    qc_path.write_text(qc_summary(table, posts, scores, freq, why, cfg), encoding="utf-8")
    print(f"Column dictionary: {dict_path}")
    print(f"QC summary: {qc_path}")
    for p in make_graphs(table, freq, cfg):
        print(f"Graph: {p}")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--posts", required=True)
    ap.add_argument("--isb")
    ap.add_argument("--out", default="ssi_outputs")
    ap.add_argument("--engines", nargs="+", default=["textblob", "vader"])
    ap.add_argument("--freq", choices=["h", "D", "W", "MS"])
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--subjectivity-model")
    ap.add_argument("--text-col"); ap.add_argument("--time-col")
    a = ap.parse_args()
    run(Config(posts_csv=a.posts, isb_csv=a.isb, out_dir=a.out, engines=a.engines,
               force_freq=a.freq, start=a.start, end=a.end, subjectivity_model=a.subjectivity_model,
               text_col=a.text_col, time_col=a.time_col))
