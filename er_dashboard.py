"""
ER Dashboard + Sector Relative Strength Overlay  (revised)

What changed vs. the original
-----------------------------
1. SIGNED gap. The original used daily_pct.abs().max(), so a -12% crash and a
   +12% rip scored identically and both produced an *upside* target. Gap is now
   a true overnight gap (Open / prior Close - 1) and keeps its sign.
2. Anchored to the earnings date. The reaction is measured on the session in the
   earnings window, not "biggest move in the last 7 days". If no ER date lands
   in the price history the row falls back to the largest move and is flagged
   NO-ER so you can see it is not an earnings reaction.
3. One batched download. SPY / sector ETFs are fetched once instead of once per
   ticker (was 29x for SPY alone).
4. 3mo history for relative strength. period="1mo" gave ~21 rows and
   pct_change(20) silently produced NaN -> 0.0 on any holiday.
5. RVOL baseline excludes the day being measured (was diluted by itself).
6. Follow-through replaces vol_weighted_move, whose thresholds (>=8 on an
   average of three daily returns) essentially never fired.
7. Missing data is NaN and flagged, not silently 0.0 / 1.0. A failed fetch no
   longer looks like a neutral reading.
8. Reaction target and analyst target are separate columns. The original
   preferred the analyst target whenever it sat in the 0.95-1.55x band, so the
   "high upside" star was mostly an analyst-target screen in disguise.
9. Yahoo calls retry with exponential backoff instead of giving up on the
   first error, and price/OHLCV history falls back to Stooq (a separate,
   keyless data source) when Yahoo has none for a symbol. A row using
   fallback data is flagged "price-src:stooq", not silently indistinguishable
   from a normal Yahoo-served row.

This ranks price/volume behaviour around earnings. It is not investment advice
and says nothing about whether a business is worth owning.

Native usage:
    pip install -r requirements.txt
    python3 er_dashboard.py
"""

import os
import time
import warnings
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)


def notify_ntfy(title: str, message: str) -> None:
    """Push a summary to the user's phone via ntfy.sh. No-op if NTFY_TOPIC is unset."""
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        print("[notify] NTFY_TOPIC not set; skipping push notification.")
        return
    try:
        requests.post(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": "default"},
            timeout=10,
        )
    except requests.RequestException as exc:
        print(f"[notify] Failed to send ntfy push: {exc}")


def with_retries(fn, *args, retries=3, base_delay=1.0, label="", **kwargs):
    """Call fn(*args, **kwargs) with exponential backoff. Returns None if every attempt fails.

    Most Yahoo fetch failures we've hit in practice are transient rate-limiting
    (a "cookie/crumb" fetch error, a bare 403) rather than the symbol having no
    data, so a short retry clears most of them without needing a second source.
    """
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            tag = label or getattr(fn, "__name__", "call")
            if attempt == retries:
                print(f"[retry] {tag} failed after {retries} attempts: {exc}")
                return None
            delay = base_delay * (2 ** (attempt - 1))
            print(f"[retry] {tag} attempt {attempt} failed ({exc}); retrying in {delay:.0f}s")
            time.sleep(delay)
    return None


# ====================== SETTINGS ======================
AFTER_HOURS_FOCUS = True   # only shifts RVOL thresholds; see note in main()
HISTORY_PERIOD = "3mo"     # needs >= ~21 sessions for the 20d RS calc
ER_WINDOW_BEFORE = 1       # sessions before the ER date to consider (BMO prints)
ER_WINDOW_AFTER = 3        # sessions after (AMC prints react the next day)
RVOL_LOOKBACK = 20         # baseline length, ending the day *before* the gap day
FOLLOW_SESSIONS = 3        # follow-through window after the gap day

CORE_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "AMD", "AVGO", "ASML", "QCOM", "TXN", "INTC", "ARM", "MRVL",
    "ORCL", "IBM",
    "SNOW", "DDOG", "NET", "CRWD", "MDB",
    "MU", "SNDK", "WDC", "STX", "AMAT","PLTR","BE"
]

BENCHMARK = "SPY"

SECTOR_ETF = {
    "Technology": "XLK",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}
DEFAULT_ETF = "XLK"


# ====================== DATA SHAPING ======================
def _naive_index(idx):
    idx = pd.to_datetime(idx)
    try:
        if idx.tz is not None:
            idx = idx.tz_convert(None)
    except (AttributeError, TypeError):
        pass
    return idx.normalize()


def ticker_frame(raw, ticker):
    """Slice one ticker's OHLCV out of a batched yf.download frame."""
    fields = ("Open", "High", "Low", "Close", "Volume")
    cols = {}
    for f in fields:
        if isinstance(raw.columns, pd.MultiIndex):
            if (f, ticker) not in raw.columns:
                return None
            cols[f] = raw[(f, ticker)]
        else:
            if f not in raw.columns:
                return None
            cols[f] = raw[f]
    df = pd.DataFrame(cols).dropna()
    if df.empty:
        return None
    df.index = _naive_index(df.index)
    return df


def close_panel(raw, tickers):
    """Close-price DataFrame for the whole universe, columns = tickers."""
    out = {}
    for t in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if ("Close", t) in raw.columns:
                out[t] = raw[("Close", t)]
        elif "Close" in raw.columns:
            out[t] = raw["Close"]
    if not out:
        return pd.DataFrame()
    px = pd.DataFrame(out)
    px.index = _naive_index(px.index)
    return px


# ====================== STOOQ FALLBACK (keyless, independent of Yahoo) ======================
STOOQ_HISTORY_DAYS = 130   # calendar days; comfortably covers HISTORY_PERIOD="3mo" of sessions


def fetch_stooq_history(ticker, days=STOOQ_HISTORY_DAYS):
    """Daily OHLCV from Stooq. No API key; separate infra from Yahoo, so it
    isn't affected by Yahoo rate-limiting/blocking. Returns None on any
    failure so callers can flag the row rather than fabricate a value."""
    sym = f"{ticker.lower()}.us"
    try:
        resp = requests.get(f"https://stooq.com/q/d/l/?s={sym}&i=d", timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[stooq] {ticker}: request failed ({exc})")
        return None

    text = resp.text.strip()
    if not text or "Date" not in text.splitlines()[0]:
        return None   # Stooq returns a plain "No data" body for unknown symbols

    try:
        df = pd.read_csv(StringIO(text), parse_dates=["Date"])
    except Exception as exc:
        print(f"[stooq] {ticker}: parse failed ({exc})")
        return None

    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    if df.empty or len(keep) < 5:
        return None

    df = df.set_index("Date").sort_index()[keep].astype(float)
    df.index = _naive_index(df.index)
    return df.tail(days) if len(df) > days else df


def fill_missing_close_columns(px, symbols, min_sessions=21):
    """Backfill sparse/missing Close columns (sector ETFs, benchmark) from
    Stooq. These are shared across every ticker's relative-strength calc, so
    a single missing ETF silently blanks that stat for every stock in the
    sector -- worth patching even though most rows never hit this path."""
    for sym in symbols:
        have = px[sym].notna().sum() if sym in px.columns else 0
        if have >= min_sessions:
            continue
        hist = fetch_stooq_history(sym)
        if hist is None or "Close" not in hist.columns:
            continue
        px = px.combine_first(pd.DataFrame({sym: hist["Close"]}))
        print(f"[stooq] backfilled {sym} close history ({len(hist)} sessions)")
    return px


def get_current_price(ticker):
    """Live-ish spot price for one symbol: Yahoo fast_info with retries,
    falling back to Stooq's latest close. Returns NaN if both fail."""
    def _fast_price():
        p = yf.Ticker(ticker).fast_info["lastPrice"]
        if p is None or not np.isfinite(p) or p <= 0:
            raise ValueError("no usable lastPrice")
        return float(p)

    price = with_retries(_fast_price, label=f"{ticker} fast_info.lastPrice")
    if price is not None:
        return price

    hist = fetch_stooq_history(ticker, days=10)
    if hist is not None and not hist.empty:
        return float(hist["Close"].iloc[-1])
    return np.nan


# ====================== EARNINGS DATES ======================
def get_earnings_dates(ticker):
    """(last_reported, next_scheduled) as tz-naive Timestamps, or (None, None)."""
    ed = with_retries(lambda: yf.Ticker(ticker).earnings_dates, label=f"{ticker} earnings_dates")
    if ed is None or ed.empty:
        return None, None

    idx = pd.to_datetime(ed.index, utc=True, errors="coerce")
    idx = idx.dropna()
    if len(idx) == 0:
        return None, None
    idx = idx.tz_convert(None).normalize().sort_values()

    now = pd.Timestamp.now().normalize()
    past = idx[idx <= now]
    future = idx[idx > now]
    last_er = past[-1] if len(past) else None
    next_er = future[0] if len(future) else None
    return last_er, next_er


def get_eps_surprise(ticker):
    """Surprise % from the most recent row that has both estimate and actual."""
    ed = with_retries(lambda: yf.Ticker(ticker).earnings_dates, label=f"{ticker} earnings_dates")
    if ed is None or ed.empty:
        return np.nan

    ed = ed.copy()
    ed.index = pd.to_datetime(ed.index, utc=True, errors="coerce")
    ed = ed[ed.index.notna()].sort_index(ascending=False)   # newest first

    for _, row in ed.iterrows():
        est = row.get("EPS Estimate", np.nan)
        act = row.get("Reported EPS", np.nan)
        if pd.notna(est) and pd.notna(act) and est != 0:
            return round(((act - est) / abs(est)) * 100, 1)
    return np.nan


# ====================== BASIC INFO ======================
def get_basic_info(ticker):
    """Sector / short interest / analyst target. Missing values stay NaN."""
    out = {
        "name": ticker,
        "sector": None,
        "short_percent": np.nan,
        "current_price": np.nan,
        "target_mean": np.nan,
        "ok": False,
    }
    info = with_retries(lambda: yf.Ticker(ticker).info, label=f"{ticker} info")
    if not info:
        return out

    def num(key):
        v = info.get(key)
        try:
            return float(v) if v is not None and not pd.isna(v) else np.nan
        except (TypeError, ValueError):
            return np.nan

    out["name"] = str(info.get("shortName") or ticker)[:14]
    out["sector"] = info.get("sector")
    out["short_percent"] = num("shortPercentOfFloat")
    price = num("currentPrice")
    if pd.isna(price):
        price = num("regularMarketPrice")
    if pd.isna(price):
        price = num("previousClose")
    out["current_price"] = price
    out["target_mean"] = num("targetMeanPrice")
    out["ok"] = True
    return out


# ====================== REACTION ======================
def compute_reaction(df, last_er=None):
    """
    Measure the earnings reaction from a single ticker's OHLCV frame.

    Returns gap (signed, %), rvol on the gap day, follow-through over the next
    FOLLOW_SESSIONS closes, the gap date, and a source tag:
      ER      - the gap day fell inside the earnings window
      NO-ER   - no ER date in range; largest absolute gap used instead
    """
    blank = {
        "gap": np.nan, "rvol": np.nan, "follow": np.nan,
        "gap_date": None, "source": "NO-DATA", "last_close": np.nan,
    }
    if df is None or len(df) < RVOL_LOOKBACK + 2:
        if df is not None and len(df):
            blank["last_close"] = float(df["Close"].iloc[-1])
        return blank

    gaps = (df["Open"] / df["Close"].shift(1) - 1) * 100
    gaps = gaps.dropna()
    if gaps.empty:
        blank["last_close"] = float(df["Close"].iloc[-1])
        return blank

    source = "NO-ER"
    window = gaps
    if last_er is not None:
        lo = last_er - pd.Timedelta(days=ER_WINDOW_BEFORE + 2)   # +2 for weekends
        hi = last_er + pd.Timedelta(days=ER_WINDOW_AFTER + 2)
        w = gaps[(gaps.index >= lo) & (gaps.index <= hi)]
        if not w.empty:
            window = w
            source = "ER"

    gap_date = window.abs().idxmax()
    gap = float(window.loc[gap_date])

    pos = df.index.get_loc(gap_date)

    # RVOL: gap-day volume vs the RVOL_LOOKBACK sessions *before* it
    start = max(0, pos - RVOL_LOOKBACK)
    base = df["Volume"].iloc[start:pos]
    rvol = np.nan
    if len(base) >= 5:
        m = float(base.mean())
        if m > 0:
            rvol = float(df["Volume"].iloc[pos]) / m

    # Follow-through: gap-day close -> close FOLLOW_SESSIONS later
    follow = np.nan
    end = pos + FOLLOW_SESSIONS
    if end < len(df):
        follow = float(df["Close"].iloc[end] / df["Close"].iloc[pos] - 1) * 100
    elif pos < len(df) - 1:
        # partial window: use what exists, still informative
        follow = float(df["Close"].iloc[-1] / df["Close"].iloc[pos] - 1) * 100

    return {
        "gap": gap,
        "rvol": rvol,
        "follow": follow,
        "gap_date": gap_date,
        "source": source,
        "last_close": float(df["Close"].iloc[-1]),
    }


def rel_strength(px, ticker, etf, bench=BENCHMARK):
    """Stock vs sector ETF (5d, 20d) and sector vs benchmark (20d), in %."""
    blank = {"rs_5d": np.nan, "rs_20d": np.nan, "sector_vs_bench": np.nan}
    need = [c for c in (ticker, etf, bench) if c in px.columns]
    if len(need) < 3:
        return blank
    sub = px[[ticker, etf, bench]].dropna()
    if len(sub) < 21:
        return blank

    r5 = sub.iloc[-1] / sub.iloc[-6] - 1
    r20 = sub.iloc[-1] / sub.iloc[-21] - 1
    return {
        "rs_5d": float((r5[ticker] - r5[etf]) * 100),
        "rs_20d": float((r20[ticker] - r20[etf]) * 100),
        "sector_vs_bench": float((r20[etf] - r20[bench]) * 100),
    }


# ====================== SCORING ======================
def score_reaction(gap, rvol, follow, after_hours_focus=False):
    """Score the print reaction. NaN inputs contribute nothing."""
    score = 0.0
    if pd.notna(gap):
        if gap >= 15:   score += 3.0
        elif gap >= 8:  score += 2.0
        elif gap >= 4:  score += 1.0
        elif gap <= -10: score -= 2.5
        elif gap <= -5:  score -= 1.5

    if pd.notna(rvol):
        if after_hours_focus:
            if rvol >= 3.5:   score += 3.0
            elif rvol >= 2.5: score += 2.0
            elif rvol >= 1.8: score += 1.0
        else:
            if rvol >= 4.0:   score += 2.0
            elif rvol >= 2.5: score += 1.5
            elif rvol >= 1.8: score += 0.7

    # Recalibrated for a multi-session cumulative move, not a single day.
    if pd.notna(follow):
        if follow >= 6:    score += 1.5
        elif follow >= 3:  score += 0.7
        elif follow <= -6: score -= 1.5
        elif follow <= -3: score -= 0.7

    return round(score, 1)


def score_fundamental(eps_surprise):
    if pd.isna(eps_surprise):
        return 0.0
    if eps_surprise >= 15:   return 1.5
    if eps_surprise >= 8:    return 0.8
    if eps_surprise <= -12:  return -1.5
    return 0.0


def score_sector(rs_20d, sector_vs_bench):
    score = 0.0
    if pd.notna(sector_vs_bench):
        if sector_vs_bench > 3:    score += 1.5
        elif sector_vs_bench > 1:  score += 0.7
        elif sector_vs_bench < -3: score -= 1.5
        elif sector_vs_bench < -1: score -= 0.7

    if pd.notna(rs_20d):
        if rs_20d > 4:      score += 1.0
        elif rs_20d > 1.5:  score += 0.5
        elif rs_20d < -4:   score -= 1.0
        elif rs_20d < -1.5: score -= 0.5

    return round(score, 1)


def reaction_target(price, gap, rvol):
    """
    Continuation target from the reaction itself. Only defined for an upside
    gap - a downside reaction does not imply an upside objective, which is what
    the original abs() gap silently produced.
    """
    if pd.isna(price) or price <= 0 or pd.isna(gap) or gap <= 0:
        return np.nan
    ext = min(max(gap * 0.45, 2.0), 11.0)
    if pd.notna(rvol):
        if rvol >= 3.0:   ext *= 1.15
        elif rvol >= 2.2: ext *= 1.05
    return round(price * (1 + ext / 100), 2)


def convergence_label(r_score, s_score):
    if r_score >= 2.0 and s_score >= 1.0:
        return "Aligned+"
    if r_score >= 2.0 and s_score <= -1.0:
        return "Fighting"
    if r_score >= 1.5 and s_score >= 0.5:
        return "Supportive"
    return ""


# ====================== MAIN ======================
def build_dashboard(tickers=CORE_TICKERS, after_hours_focus=AFTER_HOURS_FOCUS,
                    pause=0.15):
    etfs = sorted(set(SECTOR_ETF.values()) | {DEFAULT_ETF})
    universe = sorted(set(tickers) | set(etfs) | {BENCHMARK})

    def _download_batch():
        result = yf.download(universe, period=HISTORY_PERIOD, progress=False,
                             auto_adjust=True, group_by="column", threads=True)
        if result is None or result.empty:
            raise RuntimeError("empty result")
        return result

    print(f"Batch download: {len(universe)} symbols, period={HISTORY_PERIOD} ...")
    raw = with_retries(_download_batch, retries=3, base_delay=2.0, label="batched yf.download")
    if raw is None:
        print("Yahoo batch download failed after retries; falling back to per-symbol Stooq below.")
        raw = pd.DataFrame()

    px = close_panel(raw, universe)
    px = fill_missing_close_columns(px, set(etfs) | {BENCHMARK})
    print(f"Got {len(px)} sessions, {px.shape[1]} symbols.\n")

    rows = []
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}", end=" -> ")
        flags = []

        df = ticker_frame(raw, ticker)
        if df is None:
            df = fetch_stooq_history(ticker)
            if df is None:
                print("NO PRICE DATA (Yahoo + Stooq both failed)")
                continue
            flags.append("price-src:stooq")

        info = get_basic_info(ticker)
        if not info["ok"]:
            flags.append("no-info")
        time.sleep(pause)   # .info is a separate request per ticker

        last_er, next_er = get_earnings_dates(ticker)
        eps_surp = get_eps_surprise(ticker)
        if pd.isna(eps_surp):
            flags.append("no-eps")

        reaction = compute_reaction(df, last_er)
        if reaction["source"] == "NO-ER":
            flags.append("NO-ER")
        if pd.isna(reaction["rvol"]):
            flags.append("no-rvol")
        if pd.isna(reaction["follow"]):
            flags.append("fresh")   # gap is the latest bar; no follow-through yet

        sector = info["sector"]
        etf = SECTOR_ETF.get(sector, DEFAULT_ETF)
        rs = rel_strength(px, ticker, etf)
        if pd.isna(rs["rs_20d"]):
            flags.append("no-rs")

        price = info["current_price"]
        if pd.isna(price) or price <= 0:
            price = reaction["last_close"]

        r_score = score_reaction(reaction["gap"], reaction["rvol"],
                                 reaction["follow"], after_hours_focus)
        f_score = score_fundamental(eps_surp)
        s_score = score_sector(rs["rs_20d"], rs["sector_vs_bench"])
        si_boost = 1.0 if (pd.notna(info["short_percent"])
                           and info["short_percent"] >= 0.08) else 0.0
        final = round(r_score + f_score + si_boost + s_score, 1)

        r_tgt = reaction_target(price, reaction["gap"], reaction["rvol"])
        upside = round((r_tgt / price - 1) * 100, 1) if (pd.notna(r_tgt)
                                                         and price > 0) else np.nan

        a_tgt = info["target_mean"]
        a_upside = round((a_tgt / price - 1) * 100, 1) if (pd.notna(a_tgt)
                                                           and price > 0) else np.nan

        rows.append({
            "Ticker": ticker,
            "Name": info["name"],
            "Last ER": last_er.strftime("%Y-%m-%d") if last_er is not None else "",
            "Next ER": next_er.strftime("%Y-%m-%d") if next_er is not None else "",
            "Src": reaction["source"],
            "Gap Date": (reaction["gap_date"].strftime("%m-%d")
                         if reaction["gap_date"] is not None else ""),
            "Price": round(price, 2) if pd.notna(price) else np.nan,
            "Gap%": round(reaction["gap"], 1) if pd.notna(reaction["gap"]) else np.nan,
            "RVOL": round(reaction["rvol"], 1) if pd.notna(reaction["rvol"]) else np.nan,
            "Fol3%": round(reaction["follow"], 1) if pd.notna(reaction["follow"]) else np.nan,
            "EPS Surp%": eps_surp,
            "React": r_score,
            "SectorRS": round(rs["rs_20d"], 1) if pd.notna(rs["rs_20d"]) else np.nan,
            "Sect vs SPY": (round(rs["sector_vs_bench"], 1)
                            if pd.notna(rs["sector_vs_bench"]) else np.nan),
            "Sect Score": s_score,
            "Final": final,
            "React Tgt": r_tgt,
            "Upside%": upside,
            "Analyst Tgt": round(a_tgt, 2) if pd.notna(a_tgt) else np.nan,
            "Anlst Up%": a_upside,
            "Conv": convergence_label(r_score, s_score),
            "Short%": (round(info["short_percent"] * 100, 1)
                       if pd.notna(info["short_percent"]) else np.nan),
            "Flags": ",".join(flags),
        })
        print(f"OK | {reaction['source']} | React {r_score} | Sect {s_score}")

    if not rows:
        print("\nNo rows collected.")
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("Final", ascending=False).reset_index(drop=True)


def report(df):
    if df.empty:
        return

    print("\n" + "=" * 150)
    print("ER DASHBOARD + SECTOR RELATIVE STRENGTH")
    print("=" * 150)
    print(df.to_string())

    er_only = df[df["Src"] == "ER"]
    print(f"\nRows anchored to an actual earnings date: {len(er_only)}/{len(df)}")

    print("\n=== TOP IDEAS (earnings-anchored only) ===")
    top = er_only.head(12)
    if top.empty:
        print("No earnings-anchored rows in range.")
    else:
        for _, r in top.iterrows():
            if r["Final"] >= 6 and pd.notna(r["RVOL"]) and r["RVOL"] >= 2:
                flag = "STRONG"
            elif r["Final"] >= 3.5:
                flag = "Watch"
            else:
                flag = ""
            tgt = f"${r['React Tgt']:7.2f} ({r['Upside%']:+5.1f}%)" if pd.notna(r["React Tgt"]) else "     n/a (down gap)"
            rvol = f"{r['RVOL']:.1f}" if pd.notna(r["RVOL"]) else " n/a"
            conv = f" [{r['Conv']}]" if r["Conv"] else ""
            note = f" ({r['Flags']})" if r["Flags"] else ""
            print(f"{r['Ticker']:6} | ${r['Price']:8.2f} -> {tgt} | Final {r['Final']:5.1f} "
                  f"| Gap {r['Gap%']:+6.1f}% | RVOL {rvol} | Sect {r['Sect Score']:+.1f}{conv} {flag}{note}")

    print("\n=== DOWNSIDE REACTIONS (gap < -5%) ===")
    down = df[df["Gap%"] < -5].sort_values("Gap%")
    if down.empty:
        print("None.")
    else:
        for _, r in down.iterrows():
            print(f"{r['Ticker']:6} | Gap {r['Gap%']:+6.1f}% | Final {r['Final']:5.1f} | {r['Src']}")

    print("\n=== LEGEND ===")
    print("Src ER      = reaction measured in the earnings window")
    print("Src NO-ER   = no ER date in the price history; largest gap used instead")
    print("Gap%        = Open / prior Close - 1, SIGNED")
    print("RVOL        = gap-day volume vs the 20 sessions before it")
    print("Fol3%       = close-to-close move over the 3 sessions after the gap")
    print("Aligned+    = strong reaction inside a leading sector")
    print("Supportive  = decent reaction with a sector tailwind")
    print("Fighting    = strong reaction, lagging sector (higher fade risk)")
    print("Flags fresh = gap is the newest bar, no follow-through yet")
    print("\nNote: AFTER_HOURS_FOCUS only shifts RVOL thresholds. Nothing here")
    print("reads extended-hours prints; that needs history(prepost=True).")
    print("\nThis screens price/volume behaviour, not investment merit.")


def notify_summary(df):
    if df.empty:
        notify_ntfy("ER Dashboard: no data", "Download returned nothing this run.")
        return

    er_only = df[df["Src"] == "ER"]
    strong = er_only[(er_only["Final"] >= 6) & (er_only["RVOL"] >= 2)]
    watch = er_only[(er_only["Final"] >= 3.5) & (er_only["Final"] < 6)]

    top = pd.concat([strong, watch]).head(5)
    if top.empty:
        notify_ntfy("ER Dashboard: nothing notable",
                     f"{len(er_only)} earnings-anchored rows, none scored Watch/STRONG.")
        return

    lines = []
    for _, r in top.iterrows():
        tag = "STRONG" if r["Ticker"] in strong["Ticker"].values else "Watch"
        lines.append(f"{tag} {r['Ticker']} Final={r['Final']} Gap={r['Gap%']:+.1f}% RVOL={r['RVOL']:.1f}")
    notify_ntfy(f"ER Dashboard: {len(top)} idea(s)", "\n".join(lines))


if __name__ == "__main__":
    dashboard = build_dashboard()
    report(dashboard)
    if not dashboard.empty:
        dashboard.to_csv("er_dashboard.csv", index=False)
        print("\nSaved er_dashboard.csv")
    notify_summary(dashboard)
