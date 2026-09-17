"""
Institutional Options Entry & GEX Execution Engine
--------------------------------------------------
Black-Scholes gamma exposure scanner with signal classification,
cross-sectional ranking, and spread strike construction.

Native usage:
    pip install yfinance numpy pandas scipy
    python3 gex_scanner.py

NOTE: Signal thresholds, tier base scores, and the GEX bonus weight are
hand-set, not fitted. A paper loop lives in paper_trading/: oversold is
held up to 5 sessions, other directional setups are same-day.
python3 -m paper_trading.backtest_gex replays the price-only half of
those rules against daily OHLC. Neither is a walk-forward-validated
edge -- treat the ranking as a triage view.

RSI/EMA use the last complete daily bar (yesterday until 4pm ET plus a
15-minute close grace, and never a bar after the as-of date) so a 9:31
scan does not bake a few minutes of today's prints into the 14-day RSI.
Indicator history is split-adjusted; spot, walls, and GEX still use the
live unadjusted quote so strikes stay comparable.

If the earnings calendar is unreachable, the setup is still classified
(with a score haircut and a CONFIRM EARNINGS prefix) rather than
discarded. Paper trading will not open those names.

GEX sign convention (customer, not dealer): call GEX is positive and put
GEX is negative, so POSITIVE_GEX means customers are long gamma (dealers
short). SpotGamma's dealer GEX is the opposite sign. Ranking orients
bearish setups with `oriented_gex = -gex_bps`, so the sort still prefers
more-negative gamma for VOLATILITY_EXPANSION_BEAR. Do not compare the
raw `gex_1pct_m` column to a dealer-signed vendor feed without flipping.

To score live scans (including WALL_PIN / RESISTANCE, which are not
paper-traded) after they accumulate: python3 -m paper_trading.trade_gex score

Yahoo calls (history, fast_info, calendar, options, option_chain) retry with
exponential backoff instead of giving up on the first error, since most
failures seen in practice are transient rate-limiting. Price/OHLCV history
additionally falls back to Stooq (a separate, keyless data source) when
Yahoo has none for a symbol after retries -- logged as "[stooq] SYMBOL:
history backfilled from Stooq", so a fallback-served scan is visible in the
run log rather than silently indistinguishable from a normal one.
"""

import os
import time
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.stats import norm
from typing import Dict, Tuple, List, Union


def notify_ntfy(title: str, message: str, tags: str = "") -> None:
    """Push a summary to the user's phone via ntfy.sh. No-op if NTFY_TOPIC is unset."""
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        print("[notify] NTFY_TOPIC not set; skipping push notification.")
        return
    headers = {"Title": title, "Priority": "default"}
    if tags:
        headers["Tags"] = tags
    try:
        resp = requests.post(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        print(f"[notify] posted {title!r} ({resp.status_code})")
    except requests.RequestException as exc:
        print(f"[notify] Failed to send ntfy push: {exc}")


def with_retries(fn, *args, retries=3, base_delay=1.0, label="", **kwargs):
    """Call fn(*args, **kwargs) with exponential backoff. Returns None if every attempt fails.

    Most Yahoo fetch failures we've hit in practice are transient rate-limiting
    (a "cookie/crumb" fetch error, a bare 403) rather than the symbol having no
    data, so a short retry clears most of them before falling through to a
    weaker fallback or giving up.
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


# Daily OHLCV from Stooq: no API key, separate infra from Yahoo, so it isn't
# affected by Yahoo rate-limiting/blocking. Used as a last-resort fallback
# when Yahoo has no price history for a symbol after retries.
STOOQ_HISTORY_DAYS = 1150   # calendar days; comfortably covers the "3y" history this scanner uses


def fetch_stooq_history(ticker, days=STOOQ_HISTORY_DAYS):
    """Daily OHLCV from Stooq, shaped like yfinance's Ticker.history() output
    (Open/High/Low/Close/Volume, indexed by date). Returns None on any
    failure so callers can flag/log the fallback rather than fabricate data."""
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
    return df.tail(days) if len(df) > days else df


# ==========================================
# 1. VECTORIZED BLACK-SCHOLES GAMMA ENGINE
# ==========================================
# Floor 0DTE to a quarter day so gamma is finite, then cap per-contract
# gamma at the ATM value of that floor. A 0.5-day floor understated 0DTE
# GEX by ~2x; going all the way to T->0 explodes the book.
MIN_DTE_DAYS = 0.25


def calculate_vectorized_bs_gamma(
    S: float,
    K_array: np.ndarray,
    T_array: np.ndarray,
    r: float,
    sigma_array: np.ndarray,
    q: float = 0.0,
) -> np.ndarray:
    """Vectorized Black-Scholes Gamma (customer, not dealer-signed).

    0DTE is floored to MIN_DTE_DAYS and per-contract gamma is capped at the
    ATM gamma of that floor so a single expiry cannot explode the book.
    """
    t_floor = MIN_DTE_DAYS / 365.0
    T_work = np.maximum(np.asarray(T_array, dtype=float), t_floor)
    mask = (T_work > 0.0001) & (sigma_array > 0.001) & (S > 0) & (K_array > 0)
    gamma = np.zeros_like(K_array, dtype=float)

    if not np.any(mask):
        return gamma

    K_m = K_array[mask]
    T_m = T_work[mask]
    sig_m = sigma_array[mask]
    sqrt_T = np.sqrt(T_m)
    d1 = (np.log(S / K_m) + (r - q + 0.5 * sig_m**2) * T_m) / (sig_m * sqrt_T)
    pdf_d1 = norm.pdf(d1)
    disc_q = np.exp(-q * T_m)
    raw = disc_q * pdf_d1 / (S * sig_m * sqrt_T)
    # Cap at ATM gamma for T=MIN_DTE_DAYS, IV at least 15%.
    atm_cap = disc_q / (S * np.maximum(sig_m, 0.15) * np.sqrt(2.0 * np.pi * t_floor))
    gamma[mask] = np.minimum(raw, atm_cap)
    return np.nan_to_num(gamma, nan=0.0)


# ==========================================
# 2. CACHING & DATA MANAGER
# ==========================================
class TickerCacheManager:
    """Centralized cache managing rate limits, spot prices, market caps, and option chains."""

    def __init__(self, cache_ttl_seconds: int = 300, spot_ttl_seconds: int = 30):
        self.ttl = cache_ttl_seconds
        self.spot_ttl = spot_ttl_seconds
        self.tickers: Dict[str, yf.Ticker] = {}
        self.chain_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}
        self.history_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}
        self.earnings_cache: Dict[str, Tuple[float, Union[bool, str]]] = {}
        self.spot_cache: Dict[str, Tuple[float, float]] = {}
        self.mcap_cache: Dict[str, Tuple[float, float]] = {}
        self.div_cache: Dict[str, Tuple[float, float]] = {}
        self.history_source: Dict[str, str] = {}
        self.mcap_miss_ttl = 60.0  # cache NaN misses briefly so a 403 isn't retried every ticker

    def get_ticker(self, symbol: str) -> yf.Ticker:
        if symbol not in self.tickers:
            self.tickers[symbol] = yf.Ticker(symbol)
        return self.tickers[symbol]

    def get_history(self, symbol: str, period: str = "3y", auto_adjust: bool = False) -> pd.DataFrame:
        """Pulls historical price data (unadjusted by default for level comparability with spot)."""
        now = time.time()
        key = f"{symbol}_{period}_{auto_adjust}"
        if key in self.history_cache:
            ts, df = self.history_cache[key]
            if now - ts < self.ttl:
                return df

        time.sleep(0.02)
        t = self.get_ticker(symbol)
        df = with_retries(lambda: t.history(period=period, auto_adjust=auto_adjust),
                          label=f"{symbol} history")
        src = "yahoo"
        if df is None or df.empty:
            fallback = fetch_stooq_history(symbol)
            if fallback is not None and not fallback.empty:
                print(f"[stooq] {symbol}: history backfilled from Stooq ({len(fallback)} rows)")
                df = fallback
                src = "stooq"
            else:
                df = pd.DataFrame()
                src = "none"
        self.history_source[symbol] = src
        self.history_cache[key] = (now, df)
        return df

    def get_spot_price(self, symbol: str) -> float:
        """Pulls unadjusted spot price with a short TTL cache."""
        now = time.time()
        if symbol in self.spot_cache:
            ts, price = self.spot_cache[symbol]
            if now - ts < self.spot_ttl:
                return price

        time.sleep(0.02)
        t = self.get_ticker(symbol)
        price = np.nan

        def _fast_price():
            p = t.fast_info['last_price']
            if p is None or not np.isfinite(p) or p <= 0:
                raise ValueError("no usable last_price")
            return float(p)

        fast_price = with_retries(_fast_price, label=f"{symbol} fast_info.last_price")
        if fast_price is not None:
            price = fast_price

        if np.isnan(price):
            hist = self.get_history(symbol, period="3y", auto_adjust=False)
            if not hist.empty:
                price = float(hist['Close'].iloc[-1])

        if not np.isnan(price):
            self.spot_cache[symbol] = (now, price)

        return price

    @staticmethod
    def _coerce_positive(value) -> float:
        """Returns a positive float or NaN. Tolerates None, strings, numpy scalars."""
        if value is None:
            return np.nan
        try:
            v = float(value)
        except (TypeError, ValueError):
            return np.nan
        return v if np.isfinite(v) and v > 0 else np.nan

    def _read_field(self, obj, keys) -> float:
        """Try mapping access then attribute access for each candidate key."""
        if obj is None:
            return np.nan
        for key in keys:
            for accessor in (
                lambda: obj[key],
                lambda: getattr(obj, key),
            ):
                try:
                    v = self._coerce_positive(accessor())
                except Exception:
                    continue
                if not np.isnan(v):
                    return v
        return np.nan

    def get_market_cap(self, symbol: str) -> float:
        """
        Pulls market cap with TTL caching. Returns NaN when genuinely unavailable.

        fast_info's interface has shifted across yfinance versions (Mapping vs
        plain object, snake_case vs camelCase), so try several paths before
        falling back to shares * spot, then to the slow .info payload.
        """
        now = time.time()
        if symbol in self.mcap_cache:
            ts, mcap = self.mcap_cache[symbol]
            ttl = self.ttl if (mcap is not None and np.isfinite(mcap)) else self.mcap_miss_ttl
            if now - ts < ttl:
                return mcap

        time.sleep(0.02)
        t = self.get_ticker(symbol)

        fi = with_retries(lambda: t.fast_info, label=f"{symbol} fast_info")

        mcap = self._read_field(fi, ("market_cap", "marketCap"))

        # Fallback 1: shares outstanding * spot price
        if np.isnan(mcap):
            shares = self._read_field(fi, ("shares", "shares_outstanding", "sharesOutstanding"))
            if not np.isnan(shares):
                spot = self.get_spot_price(symbol)
                if not np.isnan(spot) and spot > 0:
                    mcap = shares * spot

        # Fallback 2: the slow .info payload
        if np.isnan(mcap):
            info = with_retries(lambda: t.get_info(), label=f"{symbol} get_info")
            if isinstance(info, dict):
                mcap = self._coerce_positive(info.get('marketCap'))

        # Cache hits *and* misses. A failed get_info() used to skip the cache
        # and re-hit Yahoo's slowest endpoint on every subsequent ticker.
        self.mcap_cache[symbol] = (now, mcap)
        return mcap

    def get_dividend_yield(self, symbol: str) -> float:
        """Continuous dividend yield q for BS gamma. 0.0 if unknown (not NaN —
        missing q must not blank the whole GEX book). Yahoo sometimes stores
        this as a percent (1.2) instead of a fraction (0.012). Cached so a
        failed fast_info is not retried on every GEX/flip call."""
        now = time.time()
        if symbol in self.div_cache:
            ts, q = self.div_cache[symbol]
            if now - ts < self.ttl:
                return q

        t = self.get_ticker(symbol)
        fi = with_retries(lambda: t.fast_info, label=f"{symbol} fast_info")
        q = self._read_field(fi, (
            "dividend_yield", "dividendYield", "trailing_annual_dividend_yield",
            "trailingAnnualDividendYield",
        ))
        if np.isnan(q):
            q = 0.0
        elif q > 0.25:  # 25%+ as a fraction is almost certainly a percent quote
            q = q / 100.0
        q = float(min(q, 0.20))
        self.div_cache[symbol] = (now, q)
        return q

    def check_earnings_status(self, symbol: str, days_threshold: int = 7) -> Union[bool, str]:
        """
        Tri-state earnings check.
        True    -> earnings inside the window
        False   -> confirmed clear
        UNKNOWN -> calendar check failed; caller must not treat as clear
        """
        now = time.time()
        if symbol in self.earnings_cache:
            ts, status = self.earnings_cache[symbol]
            if now - ts < self.ttl * 4:
                return status

        time.sleep(0.02)
        t = self.get_ticker(symbol)
        status = "UNKNOWN"
        cal = with_retries(lambda: t.calendar, label=f"{symbol} calendar")
        # t.calendar is deprecated in yfinance >= 0.2.40; fall through to
        # get_earnings_dates when it is missing or unusable.
        edates = self._earnings_dates_from_calendar(cal)
        if not edates:  # None (failed) or [] (empty/deprecated stub)
            ed = with_retries(lambda: t.get_earnings_dates(limit=8),
                              label=f"{symbol} get_earnings_dates")
            fallback = self._earnings_dates_from_index(ed)
            if fallback is not None:
                edates = fallback
        if edates is None:
            status = "UNKNOWN"
        elif len(edates) == 0:
            status = False
        else:
            today = _as_naive_et_day(pd.Timestamp.now(tz="America/New_York"))
            is_near = any(0 <= (ed_dt - today).days <= days_threshold for ed_dt in edates)
            status = is_near

        self.earnings_cache[symbol] = (now, status)
        return status

    @staticmethod
    def _earnings_dates_from_calendar(cal):
        """None = fetch failed; [] = parsed but empty; else list of timestamps."""
        if cal is None:
            return None
        try:
            raw = []
            if isinstance(cal, pd.DataFrame):
                if 'Earnings Date' in cal.index:
                    raw = cal.loc['Earnings Date'].values
                elif 'Earnings Date' in cal.columns:
                    raw = cal['Earnings Date'].values
            elif isinstance(cal, dict) and 'Earnings Date' in cal:
                raw = cal['Earnings Date']
            if raw is None:
                return []
            out = []
            for ed in list(raw):
                day = _as_naive_et_day(ed)
                if day is not None:
                    out.append(day)
            return out
        except Exception:
            return None

    @staticmethod
    def _earnings_dates_from_index(ed):
        if ed is None:
            return None
        try:
            if isinstance(ed, pd.DataFrame):
                if ed.empty:
                    return []
                idx = pd.to_datetime(ed.index, utc=True, errors='coerce')
            else:
                return None
            out = []
            for ts in idx.dropna():
                day = _as_naive_et_day(ts)
                if day is not None:
                    out.append(day)
            return out
        except Exception:
            return None

    def get_options_chain_within_dte(self, symbol: str, max_dte: int = 45) -> pd.DataFrame:
        """Pulls option chains with per-request sleep throttling."""
        now = time.time()
        if symbol in self.chain_cache:
            ts, df = self.chain_cache[symbol]
            if now - ts < self.ttl:
                return df

        t = self.get_ticker(symbol)
        expirations = with_retries(lambda: t.options, label=f"{symbol} options")
        if not expirations:
            return pd.DataFrame()

        today = pd.Timestamp.now().floor('D')
        valid_expirations = []
        for exp in expirations:
            try:
                dte = (pd.to_datetime(exp) - today).days
                if 0 <= dte <= max_dte:
                    valid_expirations.append((exp, dte))
            except Exception:
                continue

        chains = []
        for exp, dte in valid_expirations:
            time.sleep(0.04)
            opt = with_retries(lambda: t.option_chain(exp), retries=2, base_delay=1.0,
                               label=f"{symbol} option_chain {exp}")
            if opt is None:
                continue
            for opt_type, df_opt in [('call', opt.calls), ('put', opt.puts)]:
                if df_opt is None or df_opt.empty:
                    continue
                df_c = df_opt.copy()
                df_c['type'] = opt_type
                df_c['expiration'] = exp
                df_c['dte'] = dte
                chains.append(df_c)

        df_all = pd.concat(chains, ignore_index=True) if chains else pd.DataFrame()
        self.chain_cache[symbol] = (now, df_all)
        return df_all


def _as_naive_et_day(ts):
    """Calendar date in America/New_York. Aware stamps convert to ET first so
    a 20:00 UTC print does not roll to the next UTC day."""
    ts = pd.to_datetime(ts, errors="coerce")
    if pd.isna(ts):
        return None
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("America/New_York")
        return pd.Timestamp(year=ts.year, month=ts.month, day=ts.day)
    return ts.normalize()


cache = TickerCacheManager(cache_ttl_seconds=300, spot_ttl_seconds=30)


# ==========================================
# 3. TECHNICAL INDICATORS
# ==========================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if len(df) < 200:
        return df

    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()

    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
        df['RSI'] = np.where(avg_loss == 0, 100.0, 100.0 - (100.0 / (1.0 + rs)))

    if "High" in df.columns and "Low" in df.columns:
        prev_close = df["Close"].shift(1)
        tr = pd.concat(
            [
                df["High"] - df["Low"],
                (df["High"] - prev_close).abs(),
                (df["Low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    return df


# ==========================================
# 4. GEX & WALL ENGINE
# ==========================================
# Maximum fractional distance from spot at which a strike can qualify as a wall.
# Tunable: too tight and the wall is just the nearest strike; too loose and it
# becomes a far-OTM round number with no bearing on near-term price action.
WALL_MAX_DIST_DEFAULT = 0.12
MARKET_CLOSE_ET = (16, 0)
# Yahoo's daily bar is still a partial print at 16:00. Wait this long
# after the bell before treating today as the last complete session.
CLOSE_GRACE_MINUTES = 15


def calculate_gex_and_walls(symbol: str, r: float = 0.045) -> Tuple[float, float, str, float, float, float, float]:
    """
    Returns: (gex_1pct_m, gex_rel_bps, regime, call_wall, put_wall, gamma_flip, curr_price)

    gex_rel_bps is NaN when the market-cap lookup fails, so the ranking layer can
    substitute a neutral percentile instead of scoring the name as weak.
    """
    try:
        curr_price = cache.get_spot_price(symbol)
        if np.isnan(curr_price) or curr_price <= 0:
            return (np.nan, np.nan, "NO_DATA", np.nan, np.nan, np.nan, np.nan)

        chain = cache.get_options_chain_within_dte(symbol, max_dte=45)
        if chain.empty:
            return (np.nan, np.nan, "NO_DATA", np.nan, np.nan, np.nan, curr_price)

        chain['openInterest'] = pd.to_numeric(chain['openInterest'], errors='coerce').fillna(0)
        chain['impliedVolatility'] = pd.to_numeric(chain['impliedVolatility'], errors='coerce').fillna(0)
        chain['strike'] = pd.to_numeric(chain['strike'], errors='coerce').fillna(0)

        min_strike = curr_price * 0.70
        max_strike = curr_price * 1.30
        valid_chain = chain[
            (chain['strike'] >= min_strike) &
            (chain['strike'] <= max_strike) &
            (chain['openInterest'] >= 5) &
            (chain['impliedVolatility'] > 0.001)
        ].copy()

        if valid_chain.empty:
            return (np.nan, np.nan, "NO_DATA", np.nan, np.nan, np.nan, curr_price)

        T_arr = np.maximum(valid_chain['dte'].to_numpy(dtype=float), MIN_DTE_DAYS) / 365.0
        K_arr = valid_chain['strike'].values
        sig_arr = valid_chain['impliedVolatility'].values
        q = cache.get_dividend_yield(symbol)

        gammas = calculate_vectorized_bs_gamma(curr_price, K_arr, T_arr, r, sig_arr, q=q)
        valid_chain['bs_gamma'] = gammas

        # Dollar gamma per 1% move, in $ millions
        valid_chain['raw_gex'] = (
            valid_chain['bs_gamma'] * valid_chain['openInterest'] * 100.0
            * (curr_price ** 2) * 0.01 * 1e-6
        )
        # Customer-signed GEX: calls positive, puts negative. Dealers are
        # typically short this book, so dealer GEX = -signed_gex. Ranking
        # orients bearish setups separately; do not flip this column.
        valid_chain['signed_gex'] = np.where(
            valid_chain['type'] == 'call', valid_chain['raw_gex'], -valid_chain['raw_gex']
        )

        # Walls come from OPEN INTEREST, constrained by side and distance.
        #
        # Two failure modes to avoid:
        #   - Gamma-weighted argmax peaks at the money, so the "wall" collapses to
        #     whichever strike is nearest spot -- a moneyness artifact.
        #   - Raw OI argmax over a wide window finds the most popular strike, which
        #     is a far-OTM round number (lottery calls), too distant to trade against.
        #
        # So: call wall is the heaviest call OI at or above spot, put wall the
        # heaviest put OI at or below spot, both within WALL_MAX_DIST of spot.
        # Un-netted by side so a strike heavy in both doesn't cancel itself out.
        #
        # CAVEAT: for names whose true OI peak lies beyond the window, the chosen
        # wall lands on the band edge -- that is the constraint binding, not a real
        # cluster. Widening WALL_MAX_DIST re-admits the lottery strikes. There is no
        # correct value; treat a wall within one strike of the edge with suspicion.
        WALL_MAX_DIST = WALL_MAX_DIST_DEFAULT

        call_wall, put_wall = select_oi_walls(valid_chain, curr_price, WALL_MAX_DIST)

        # Gamma flip: customer-signed net gamma changes sign (dealer is opposite).
        # See compute_gamma_flip for why this is not a cumulative sum over strikes.
        net_gex_by_strike = valid_chain.groupby('strike')['signed_gex'].sum().sort_index()

        gamma_flip = compute_gamma_flip(
            curr_price,
            K_arr,
            T_arr,
            sig_arr,
            valid_chain['openInterest'].values,
            (valid_chain['type'] == 'call').values,
            r,
            q=q,
        )

        total_net_gex_m = float(net_gex_by_strike.sum())
        regime = "POSITIVE_GEX" if total_net_gex_m >= 0 else "NEGATIVE_GEX"

        # FIX: NaN (not 0.0) when market cap is unavailable
        mcap = cache.get_market_cap(symbol)
        if not np.isnan(mcap) and mcap > 0:
            gex_rel_bps = float(((total_net_gex_m * 1e6) / mcap) * 10000.0)
        else:
            gex_rel_bps = np.nan

        return (total_net_gex_m, gex_rel_bps, regime, call_wall, put_wall, gamma_flip, curr_price)

    except Exception:
        return (np.nan, np.nan, "NO_DATA", np.nan, np.nan, np.nan, np.nan)


def select_oi_walls(
    chain: pd.DataFrame,
    spot: float,
    max_dist: float = WALL_MAX_DIST_DEFAULT,
) -> Tuple[float, float]:
    """Call wall = max call OI in [spot, spot*(1+max_dist)]; put wall = max
    put OI in [spot*(1-max_dist), spot]. Un-netted by side. Pure function so
    tests can pin the OI-constraint without hitting Yahoo."""
    if chain is None or chain.empty or not np.isfinite(spot) or spot <= 0:
        return np.nan, np.nan
    calls = chain[chain["type"] == "call"]
    puts = chain[chain["type"] == "put"]
    call_side = calls[
        (calls["strike"] >= spot) & (calls["strike"] <= spot * (1.0 + max_dist))
    ]
    put_side = puts[
        (puts["strike"] <= spot) & (puts["strike"] >= spot * (1.0 - max_dist))
    ]
    call_oi = call_side.groupby("strike")["openInterest"].sum()
    put_oi = put_side.groupby("strike")["openInterest"].sum()
    call_wall = float(call_oi.idxmax()) if not call_oi.empty else np.nan
    put_wall = float(put_oi.idxmax()) if not put_oi.empty else np.nan
    return call_wall, put_wall


def compute_gamma_flip(
    curr_price: float,
    K_arr: np.ndarray,
    T_arr: np.ndarray,
    sig_arr: np.ndarray,
    oi_arr: np.ndarray,
    is_call: np.ndarray,
    r: float,
    q: float = 0.0,
    lo: float = 0.80,
    hi: float = 1.20,
    n_grid: int = 121,
) -> float:
    """
    Spot price at which customer-signed net gamma changes sign.

    Dealers are typically short this book, so dealer GEX flips at the same
    level with the opposite sign. Computed by sweeping hypothetical spots and
    recomputing every contract's gamma at each, holding OI and IV fixed.

    This replaces the common cumulative-sum-across-strikes approximation, which
    accumulates from the bottom of the strike filter rather than from zero. That
    makes its crossing point a function of where the filter truncates -- move the
    window from 0.7x to 0.8x and the reported flip moves with it. The sweep has no
    such dependence: it asks "at what spot does net gamma change sign", which is
    the quantity the flip is supposed to name.

    Caveat: holding IV fixed while spot moves is a simplification. Real IV shifts
    along the skew as price moves, so treat the level as approximate.
    """
    if len(K_arr) == 0 or not np.isfinite(curr_price) or curr_price <= 0:
        return np.nan

    grid = np.linspace(curr_price * lo, curr_price * hi, n_grid)
    totals = np.empty(n_grid, dtype=float)
    for i, s in enumerate(grid):
        gamma = calculate_vectorized_bs_gamma(float(s), K_arr, T_arr, r, sig_arr, q=q)
        gex = gamma * oi_arr * 100.0 * (s ** 2) * 0.01 * 1e-6
        signed = np.where(is_call, gex, -gex)
        totals[i] = signed.sum()

    crossings = np.where(np.diff(np.signbit(totals)))[0]
    if len(crossings) == 0:
        return np.nan

    candidates = []
    for i in crossings:
        s1, s2 = grid[i], grid[i + 1]
        g1, g2 = totals[i], totals[i + 1]
        candidates.append(float(s1 - g1 * (s2 - s1) / (g2 - g1)) if g2 != g1 else float(s1))

    return min(candidates, key=lambda x: abs(x - curr_price))


# ==========================================
# 5. SPREAD STRIKE SELECTOR
# ==========================================
def get_grid_step(price: float) -> float:
    if price < 25:
        return 1.0
    elif price < 50:
        return 2.5
    elif price < 300:
        return 5.0
    else:
        return 10.0


def spread_width_steps(
    price: float,
    atr: float = np.nan,
    min_steps: int = 2,
    max_steps: int = 8,
    default_steps: int = 2,
) -> int:
    """Spread width in strike-grid steps: max(2, int(ATR / grid_step))."""
    step = get_grid_step(price)
    if not np.isfinite(atr) or atr <= 0 or step <= 0:
        return int(default_steps)
    steps = max(int(min_steps), int(float(atr) / step))
    return int(min(steps, max_steps))


def wall_at_band_edge(
    wall: float,
    spot: float,
    side: str,
    max_dist: float = WALL_MAX_DIST_DEFAULT,
) -> bool:
    """True when the chosen wall sits within 1% of the 12% band edge —
    the distance constraint is binding, not a real OI cluster."""
    if wall is None or not np.isfinite(wall) or not np.isfinite(spot) or spot <= 0:
        return False
    if side == "call":
        return float(wall) >= spot * (1.0 + max_dist - 0.01)
    return float(wall) <= spot * (1.0 - max_dist + 0.01)


def build_spread_strikes(
    curr_price: float,
    wall: float,
    spread_type: str = "BULL_PUT",
    min_width_steps: int = 2
) -> Tuple[float, float]:
    """
    BULL_PUT  -> (sell_short_put,  buy_long_put)
    BEAR_CALL -> (sell_short_call, buy_long_call)
    BEAR_PUT  -> (buy_long_put,    sell_short_put)
    """
    step = get_grid_step(curr_price)

    if spread_type == "BULL_PUT":
        min_ref = curr_price if np.isnan(wall) else min(wall, curr_price)
        sell_strike = np.floor(min_ref / step) * step
        buy_strike = sell_strike - (min_width_steps * step)
        return float(sell_strike), float(buy_strike)

    elif spread_type == "BEAR_CALL":
        max_ref = curr_price if np.isnan(wall) else max(wall, curr_price)
        sell_strike = np.ceil(max_ref / step) * step
        buy_strike = sell_strike + (min_width_steps * step)
        return float(sell_strike), float(buy_strike)

    elif spread_type == "BEAR_PUT":
        # Long leg anchored at spot so the debit retains convexity
        buy_strike = np.floor(curr_price / step) * step
        sell_strike = buy_strike - (min_width_steps * step)
        return float(buy_strike), float(sell_strike)

    return float(curr_price), float(curr_price)


def clamp_below(level: float, spot: float, near: float = 0.02, far: float = 0.08,
                atr: float = np.nan) -> float:
    """
    Pin a level strictly below spot, between `near` and `far` fractional distance.

    One-sided clamps were the source of 17%-wide stops: min(wall, spot*0.98)
    guarantees the level sits below spot but puts no floor under how far below.
    Falls back to the `near` bound when the reference level is missing.
    When ATR is known, the band is at least half a day's range and at most
    1.25 ATR, capped so a high-ATR name is not stopped out by a 2% tick.
    """
    near, far = _atr_scaled_band(spot, atr, near, far)
    lo = spot * (1.0 - far)
    hi = spot * (1.0 - near)
    if level is None or np.isnan(level):
        return float(hi)
    return float(min(max(level, lo), hi))


def clamp_above(level: float, spot: float, near: float = 0.02, far: float = 0.08,
                atr: float = np.nan) -> float:
    """Pin a level strictly above spot, between `near` and `far` fractional distance."""
    near, far = _atr_scaled_band(spot, atr, near, far)
    lo = spot * (1.0 + near)
    hi = spot * (1.0 + far)
    if level is None or np.isnan(level):
        return float(lo)
    return float(min(max(level, lo), hi))


def _atr_scaled_band(spot: float, atr: float, near: float, far: float) -> Tuple[float, float]:
    if not np.isfinite(atr) or atr <= 0 or not np.isfinite(spot) or spot <= 0:
        return float(near), float(far)
    frac = float(atr) / float(spot)
    near2 = float(min(max(near, 0.5 * frac), 0.05))
    far2 = float(min(max(far, 1.25 * frac), 0.15))
    if far2 < near2:
        far2 = near2
    return near2, far2


def last_complete_daily_row(hist: pd.DataFrame, now=None) -> pd.Series:
    """Last finished daily bar as of `now`.

    Live: during RTH (and 15 minutes after 16:00 ET) Yahoo's latest row is
    often today's incomplete session. Use the prior bar until close+grace.

    Backtest: pass historical `now` even if `hist` contains later days —
    bars after the as-of date are ignored so EMA/RSI cannot see the future.
    """
    if hist is None or hist.empty:
        raise ValueError("history is empty")
    now_et = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="America/New_York")
    if now_et.tzinfo is None:
        now_et = now_et.tz_localize("America/New_York")
    else:
        now_et = now_et.tz_convert("America/New_York")
    today = pd.Timestamp(year=now_et.year, month=now_et.month, day=now_et.day)

    complete_h = MARKET_CLOSE_ET[0]
    complete_m = MARKET_CLOSE_ET[1] + CLOSE_GRACE_MINUTES
    if complete_m >= 60:
        complete_h += complete_m // 60
        complete_m = complete_m % 60
    before_complete = (now_et.hour, now_et.minute) < (complete_h, complete_m)

    idx = hist.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx)
    if idx.tz is not None:
        idx_et = idx.tz_convert("America/New_York")
        idx_days = pd.DatetimeIndex([pd.Timestamp(ts.date()) for ts in idx_et])
    else:
        idx_days = idx.normalize()

    mask = (idx_days < today) if before_complete else (idx_days <= today)
    usable = hist.iloc[np.flatnonzero(np.asarray(mask))]
    if usable.empty:
        raise ValueError("history is empty")
    return usable.iloc[-1]


# ==========================================
# 6. SIGNAL & RANKING ENGINE
# ==========================================
def _empty_row(symbol: str, signal: str, strategy: str, score: float,
               has_earnings_data: bool, **overrides) -> Dict:
    row = {
        "symbol": symbol,
        "signal": signal,
        "base_score": score,
        "rank_score": score,
        "has_earnings_data": has_earnings_data,
        "price": np.nan,
        "rsi": np.nan,
        "gex_1pct_m": np.nan,
        "gex_bps": np.nan,
        "call_wall": np.nan,
        "put_wall": np.nan,
        "gamma_flip": np.nan,
        "stop_loss": np.nan,
        "target_price": np.nan,
        "hold_horizon": None,
        "call_wall_edge": False,
        "put_wall_edge": False,
        "recommended_strategy": strategy,
    }
    row.update(overrides)
    return row


# Sessions the paper loop (and the backtest) will hold a directional proxy.
# OVERSOLD is a swing: same-day paper P&L was a coin flip, while 1d/5d
# hold-to-close was positive on the watchlist. Everything else stays same-day.
HOLD_HORIZON_DAYS = {
    "OVERSOLD_BULL_PULLBACK": 5,
    "VOLATILITY_EXPANSION_BEAR": 1,
    "RESISTANCE_PINNED_SHORT_VOL": 1,
    "WALL_PIN": 1,
    "DAMPENED_BULL_TREND": 1,
    "HIGH_VOLATILITY_DANGER_ZONE": 1,
    "NO_GEX_REGIME": 0,
}


def classify_setup(
    curr_price: float,
    rsi: float,
    ema200: float,
    regime: str,
    call_wall: float,
    put_wall: float,
    gamma_flip: float,
    atr: float = np.nan,
) -> Dict:
    """Map price/RSI/EMA + GEX state onto a signal, strategy, stop, and target.

    Earnings and data-availability gating stay in calculate_equity_signal.
    When walls/flip are NaN, stops/targets fall back to the clamp `near`
    bounds (same as a live scan that couldn't locate a wall). When `regime`
    is neither POSITIVE_GEX nor NEGATIVE_GEX, wall/oversold branches still
    fire (they don't need a regime), and anything leftover is NO_GEX_REGIME
    so a historical backtest doesn't pretend it saw customer gamma.
    """
    at_call_wall_band = (
        not np.isnan(call_wall) and
        (curr_price >= call_wall * 0.985) and
        (curr_price <= call_wall * 1.020)
    )
    width_steps = spread_width_steps(curr_price, atr)

    if rsi < 35 and curr_price >= ema200:
        signal = "OVERSOLD_BULL_PULLBACK"
        base_score = 80.0 + min(35.0 - rsi, 10.0)
        sell_st, buy_st = build_spread_strikes(curr_price, put_wall, "BULL_PUT", width_steps)
        strat = f"Bull Put Spread ${sell_st:.1f}/${buy_st:.1f}"
        stop_loss = clamp_below(put_wall * 0.98 if not np.isnan(put_wall) else np.nan,
                                curr_price, near=0.02, far=0.08, atr=atr)
        target_price = clamp_above(np.nan, curr_price, near=0.04, far=0.04, atr=atr)

    elif at_call_wall_band and rsi > 68:
        signal = "RESISTANCE_PINNED_SHORT_VOL"
        base_score = 60.0 + min(rsi - 68.0, 10.0)
        sell_st, buy_st = build_spread_strikes(curr_price, call_wall, "BEAR_CALL", width_steps)
        strat = f"Bear Call Spread ${sell_st:.1f}/${buy_st:.1f}"
        stop_loss = clamp_above(call_wall * 1.02 if not np.isnan(call_wall) else np.nan,
                                curr_price, near=0.03, far=0.06, atr=atr)
        target_price = clamp_below(gamma_flip, curr_price, near=0.01, far=0.03, atr=atr)

    elif at_call_wall_band:
        signal = "WALL_PIN"
        base_score = 40.0
        strat = "Iron Condor / Short Volatility"
        # Upside breach only; a condor's downside leg needs its own level if traded
        stop_loss = clamp_above(call_wall * 1.02 if not np.isnan(call_wall) else np.nan,
                                curr_price, near=0.03, far=0.06, atr=atr)
        target_price = curr_price * 1.005

    elif regime == "POSITIVE_GEX":
        signal = "DAMPENED_BULL_TREND"
        base_score = 30.0
        strat = "Covered Calls / Cash-Secured Puts"
        stop_loss = clamp_below(gamma_flip, curr_price, near=0.03, far=0.08, atr=atr)
        target_price = clamp_above(call_wall, curr_price, near=0.02, far=0.10, atr=atr)

    elif regime == "NEGATIVE_GEX":
        if rsi < 40 and curr_price < ema200:
            signal = "VOLATILITY_EXPANSION_BEAR"
            base_score = 75.0 + min(40.0 - rsi, 10.0)
            buy_st, sell_st = build_spread_strikes(curr_price, put_wall, "BEAR_PUT", width_steps)
            strat = f"Bear Put Debit Spread ${buy_st:.1f}/${sell_st:.1f}"
            ref_stop = gamma_flip if not np.isnan(gamma_flip) else ema200
            stop_loss = clamp_above(ref_stop, curr_price, near=0.03, far=0.08, atr=atr)
            target_price = clamp_below(put_wall * 0.95 if not np.isnan(put_wall) else np.nan,
                                       curr_price, near=0.05, far=0.15, atr=atr)
        else:
            signal = "HIGH_VOLATILITY_DANGER_ZONE"
            base_score = 15.0
            strat = "Long Gamma / Long Strangles"
            # Non-directional trade; these levels describe the move size, not a side
            stop_loss = curr_price * 0.95
            target_price = curr_price * 1.10

    else:
        signal = "NO_GEX_REGIME"
        base_score = 0.0
        strat = "GEX regime unavailable"
        stop_loss = np.nan
        target_price = np.nan

    return {
        "signal": signal,
        "base_score": float(base_score),
        "recommended_strategy": strat,
        "stop_loss": float(stop_loss) if not np.isnan(stop_loss) else np.nan,
        "target_price": float(target_price) if not np.isnan(target_price) else np.nan,
        "hold_horizon": HOLD_HORIZON_DAYS.get(signal),
    }


def calculate_equity_signal(symbol: str) -> Dict:
    # Earnings check first: cheapest guard, avoids chain fetches on blackout names
    earnings_status = cache.check_earnings_status(symbol, days_threshold=7)

    if earnings_status is True:
        curr_price = cache.get_spot_price(symbol)
        return _empty_row(
            symbol, "EARNINGS_BLACKOUT_VOL_CRUSH",
            "AVOID / Long Straddle (IV Collapse Risk)", -100.0, True,
            price=round(curr_price, 2) if not np.isnan(curr_price) else np.nan,
        )

    hist = cache.get_history(symbol, period="3y", auto_adjust=True)
    if hist.empty or len(hist) < 200:
        # Split-adjusted history is preferred for RSI/EMA; fall back to
        # unadjusted only if Yahoo/Stooq has nothing on the adjusted path.
        hist = cache.get_history(symbol, period="3y", auto_adjust=False)
    if hist.empty or len(hist) < 200:
        return _empty_row(
            symbol, "NO_DATA", "Insufficient History (<200 bars)", -999.0,
            earnings_status != "UNKNOWN",
        )

    gex_1pct_m, gex_bps, regime, call_wall, put_wall, gamma_flip, curr_price = \
        calculate_gex_and_walls(symbol)

    if regime == "NO_DATA" or np.isnan(curr_price):
        return _empty_row(
            symbol, "NO_DATA", "Data Unavailable / Chain Failed", -999.0,
            earnings_status != "UNKNOWN",
        )

    hist = compute_indicators(hist)
    try:
        latest = last_complete_daily_row(hist)
    except ValueError:
        latest = hist.iloc[-1]

    rsi = latest.get('RSI', np.nan)
    ema200 = latest.get('EMA_200', np.nan)
    atr = latest.get('ATR', np.nan)

    if np.isnan(rsi) or np.isnan(ema200):
        return _empty_row(
            symbol, "NO_DATA", "Missing Technical Indicators", -999.0,
            earnings_status != "UNKNOWN",
            price=round(float(curr_price), 2),
            gex_1pct_m=gex_1pct_m, gex_bps=gex_bps,
            call_wall=call_wall, put_wall=put_wall, gamma_flip=gamma_flip,
            call_wall_edge=wall_at_band_edge(call_wall, curr_price, "call"),
            put_wall_edge=wall_at_band_edge(put_wall, curr_price, "put"),
        )

    classified = classify_setup(
        float(curr_price), float(rsi), float(ema200),
        regime, call_wall, put_wall, gamma_flip, atr=atr,
    )

    # Yahoo's earnings calendar flakes often. Previously that threw away a
    # finished GEX/technical read and emitted EARNINGS_DATA_UNAVAILABLE with
    # no setup. Still classify, flag it, haircut the rank, and let the paper
    # driver skip the open.
    earnings_ok = earnings_status is False
    base_score = float(classified["base_score"])
    if not earnings_ok:
        base_score = round(base_score * 0.5, 2)
        strat = "CONFIRM EARNINGS — " + classified["recommended_strategy"]
    else:
        strat = classified["recommended_strategy"]

    call_edge = wall_at_band_edge(call_wall, curr_price, "call")
    put_edge = wall_at_band_edge(put_wall, curr_price, "put")
    if call_edge or put_edge:
        edge_bits = []
        if call_edge:
            edge_bits.append("call")
        if put_edge:
            edge_bits.append("put")
        strat = strat + f" [WALL_EDGE {'/'.join(edge_bits)}]"

    return {
        "symbol": symbol,
        "signal": classified["signal"],
        "base_score": round(float(base_score), 2),
        "rank_score": round(float(base_score), 2),
        "has_earnings_data": bool(earnings_ok),
        "price": round(float(curr_price), 2),
        "rsi": round(float(rsi), 1),
        "gex_1pct_m": round(float(gex_1pct_m), 2),
        "gex_bps": round(float(gex_bps), 4) if not np.isnan(gex_bps) else np.nan,
        "call_wall": round(float(call_wall), 2) if not np.isnan(call_wall) else np.nan,
        "put_wall": round(float(put_wall), 2) if not np.isnan(put_wall) else np.nan,
        "gamma_flip": round(float(gamma_flip), 2) if not np.isnan(gamma_flip) else np.nan,
        "call_wall_edge": bool(call_edge),
        "put_wall_edge": bool(put_edge),
        "stop_loss": (round(float(classified["stop_loss"]), 2)
                      if not np.isnan(classified["stop_loss"]) else np.nan),
        "target_price": (round(float(classified["target_price"]), 2)
                         if not np.isnan(classified["target_price"]) else np.nan),
        "hold_horizon": classified.get("hold_horizon"),
        "recommended_strategy": strat,
    }


# ==========================================
# 7. PIPELINE RUNNER & CROSS-SECTIONAL RANKING
# ==========================================
# Signals representing an actual identified setup. Only these get ranked.
SETUP_SIGNALS = [
    "OVERSOLD_BULL_PULLBACK",
    "RESISTANCE_PINNED_SHORT_VOL",
    "WALL_PIN",
    "VOLATILITY_EXPANSION_BEAR",
]

# DAMPENED_BULL_TREND is the residual bucket: positive gamma, no wall proximity,
# no RSI extreme. It is "nothing identified here", not a setup. Ranking inside it
# sorts on gex_bps alone, which is net gamma over market cap -- a structural
# property of the ticker, not a timing signal. The same names would top that list
# every day. So it is reported unranked.
RESIDUAL_SIGNALS = ["DAMPENED_BULL_TREND"]

AVOID_SIGNALS = [
    "EARNINGS_BLACKOUT_VOL_CRUSH",
    "HIGH_VOLATILITY_DANGER_ZONE",
]


def generate_top_trades(symbols_list: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (ranked setups, unranked residual, avoid list)."""
    setups = []
    residual = []
    avoid_list = []

    # Order-preserving dedupe. A repeated ticker costs a duplicate row and, worse,
    # double-counts in the cross-sectional percentile, shifting every other name.
    seen = set()
    symbols = []
    for s in symbols_list:
        u = s.strip().upper()
        if u and u not in seen:
            seen.add(u)
            symbols.append(u)

    dropped = len(symbols_list) - len(symbols)
    if dropped:
        print(f"[INFO] Dropped {dropped} duplicate ticker(s) from the watchlist.")

    print(f"Scanning {len(symbols)} tickers with Normalized BS-GEX Engine...")

    for idx, sym in enumerate(symbols, 1):
        print(f"[{idx}/{len(symbols)}] Scanning {sym}...", end="\r")
        try:
            res = calculate_equity_signal(sym)
            if res["signal"] in AVOID_SIGNALS:
                avoid_list.append(res)
            elif res["signal"] in RESIDUAL_SIGNALS:
                residual.append(res)
            elif res["signal"] in SETUP_SIGNALS:
                setups.append(res)
            # NO_DATA and anything unrecognised is dropped
        except Exception as e:
            print(f"\n[ERROR] Exception processing {sym}: {e}")

    print("\nScan Complete!\n")

    df_setups = pd.DataFrame(setups)
    if not df_setups.empty:
        # Orient GEX by setup direction: bearish setups want negative gamma
        oriented_gex = np.where(
            df_setups['signal'] == "VOLATILITY_EXPANSION_BEAR",
            -df_setups['gex_bps'],
            df_setups['gex_bps']
        )

        # NaN gex_bps (failed market-cap lookup) falls through to a neutral 0.5.
        # Check the finite count first: np.std on an empty slice warns.
        finite = np.isfinite(oriented_gex)
        n_rows = len(df_setups)
        n_finite = int(finite.sum())

        if n_finite > 1 and np.std(oriented_gex[finite]) > 1e-6:
            df_setups['gex_pct'] = pd.Series(oriented_gex).rank(pct=True).fillna(0.5).values
        else:
            # With one setup row the bonus is a constant offset and changes no ordering.
            df_setups['gex_pct'] = 0.5

        # Only a genuine data problem warrants the warning. A single-row scan is
        # not a data problem, and previously tripped this by itself.
        if n_rows > 1 and n_finite <= n_rows / 2:
            print(f"[WARN] Market cap unavailable for {n_rows - n_finite}/{n_rows} ranked "
                  f"tickers; GEX ranking term is weak or inactive this scan.")

        df_setups['rank_score'] = (df_setups['base_score'] + df_setups['gex_pct'] * 9.0).round(2)
        df_setups = df_setups.sort_values(by="rank_score", ascending=False).reset_index(drop=True)

    df_residual = pd.DataFrame(residual)
    if not df_residual.empty:
        # Sorted by ticker, not by score -- there is no meaningful ordering here
        df_residual = df_residual.sort_values(by="symbol").reset_index(drop=True)

    df_avoid = pd.DataFrame(avoid_list)
    if not df_avoid.empty:
        df_avoid = df_avoid.sort_values(by="symbol").reset_index(drop=True)

    return df_setups, df_residual, df_avoid


# ==========================================
# 8. COLAB EXECUTION BLOCK
# ==========================================
# Duplicates are removed automatically, so a messy list is fine.
#
# CAUTION on leveraged ETFs (MSTU, SOXL, TQQQ, etc.): they have no meaningful
# market cap, so gex_bps comes back NaN and the GEX term goes neutral for them.
# Their RSI and 200 EMA are also distorted by daily-reset volatility decay, so
# the technical thresholds here do not mean what they mean on an equity.
WATCHLIST = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA",
    "AMD", "AMAT", "ASML", "DDOG", "HOOD", "MSTR", "ORCL", "MU",
    "INTC", "COIN", "PLTR", "IREN", "BE"
]

if __name__ == "__main__":
    df_setups, df_residual, df_avoid = generate_top_trades(WATCHLIST)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    if not df_setups.empty:
        print("=== IDENTIFIED SETUPS (ranked) ===")
        print(df_setups[[
            "symbol", "signal", "rank_score", "base_score", "price",
            "rsi", "gex_1pct_m", "gex_bps", "call_wall", "put_wall", "gamma_flip",
            "call_wall_edge", "put_wall_edge",
            "stop_loss", "target_price", "hold_horizon", "has_earnings_data",
            "recommended_strategy"
        ]])
    else:
        print("=== IDENTIFIED SETUPS (ranked) ===")
        print("None. No ticker hit a defined setup this scan.")

    if not df_residual.empty:
        print("\n=== NO SETUP IDENTIFIED (unranked, alphabetical) ===")
        print("Positive gamma, no wall proximity, no RSI extreme. Listed for reference only;")
        print("there is no ordering here because nothing distinguishes these names today.\n")
        print(df_residual[[
            "symbol", "price", "rsi", "gex_1pct_m", "call_wall", "put_wall", "gamma_flip"
        ]])

    if not df_avoid.empty:
        print("\n=== BLACKOUT / AVOID / DATA-UNAVAILABLE ===")
        print(df_avoid[[
            "symbol", "signal", "has_earnings_data", "price", "recommended_strategy"
        ]])

    if not df_setups.empty:
        top = df_setups.head(5)
        summary = "\n".join(
            f"{r.symbol} {r.signal} score={r.rank_score} hold={r.hold_horizon}d {r.recommended_strategy}"
            for r in top.itertuples()
        )
        title = f"GEX Scan: {len(df_setups)} setup(s) found"
    else:
        summary = "No ticker hit a defined setup this scan."
        title = "GEX Scan: no setups"

    notify_ntfy(title, summary)
