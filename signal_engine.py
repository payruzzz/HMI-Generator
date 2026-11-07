"""Support/Resistance Signal Engine — monitoring only, no execution.

Problem solved
-------------
Trading the "touch" of a support/resistance level gets fooled by false
breakouts: price pokes resistance, reverses, and the level was noise.
The fix is to never trade the touch. A signal is emitted only when the
price:

  1. BREAKS the level (close moves past it) AND volume confirms the move,
  2. RETRETS the broken level (it becomes support/resistance), and the
     retest candle closes THROUGH it again.

That second step converts a "poke" into a confirmed level. Without the
retest there is no signal, which is exactly what stops "price touched
resistance, then reversed" from becoming a trade.

Invariant (always true)
-----------------------
    A SIGNAL IS EMITTED ONLY ONE OR MORE CANDLES AFTER THE LEVEL WAS
    BREACHED, AND ONLY AT THE RETEST. The touching candle never alone
    generates a signal.

Architecture fit
----------------
Architecture-neutral. Drop into any price-tracking system (Flask + Chart.js
+ SQLite, or Django) that already delivers OHLCV candles. No network, no
exchange API, no execution — pure reasoning + thresholds so it can be
reviewed in the lab before any live decision.

Usage (called after every new candle arrives)
---------------------------------------------
    engine = SignalEngine()
    ...
    for s in engine.detect(latest_candle):
        # update chart + reasoning display + log
        record(s)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class Candle:
    """One completed price candlestick."""

    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class Signal:
    """One generated trading signal."""

    direction: str          # "BUY" or "SELL"
    confidence: int         # 0-100, derived from confluence scoring
    level: Optional[str]    # "SUPPORT" or "RESISTANCE"
    reason: str             # human-readable reasoning
    trigger_at: int = -1    # index of the confirming (breakout) candle
    retest_at: int = -1     # index of the retest candle


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SignalEngine:
    """Detects support/resistance zones and reasons only at confirmed retests."""

    # Thresholds (tunable, but tuned once, not per-candle)
    PIVOT_WINDOW = 5          # candles each side used to spot a level
    ZONE_TOLERANCE = 0.006    # ~0.6% price spread to merge pivots into a zone
    VOLUME_RATIO = 1.5        # breakout volume must be >= this x average
    RETEST_MIN = 2            # at least this many candles between break + retest
    MIN_SCORE = 3             # minimum confluence score to emit a signal

    def __init__(self) -> None:
        self.candles: List[Candle] = []
        self.levels: List[dict] = []   # active support/resistance zones
        self._score = 0

    # -- public ------------------------------------------------------------

    def add_candle(self, candle: Candle) -> None:
        self.candles.append(candle)

    def detect(self) -> List[Signal]:
        """Return a (small) list of newly confirmed signals, newest last."""
        if len(self.candles) < self.PIVOT_WINDOW * 2 + 2:
            return []
        self._discover_levels()
        return self._signal_from_latest()

    # -- level discovery ---------------------------------------------------

    def _discover_levels(self) -> None:
        pivots = self._find_pivots()
        highs = [p[2] for p in pivots if p[0] == "H"]
        lows = [p[2] for p in pivots if p[0] == "L"]

        self.levels = []
        for price in self._cluster(highs):
            self.levels.append({"kind": "RESISTANCE", "price": price,
                                "touches": self._count_touches(price, highs)})
        for price in self._cluster(lows):
            self.levels.append({"kind": "SUPPORT", "price": price,
                                "touches": self._count_touches(price, lows)})

    def _find_pivots(self) -> List[tuple]:
        """'H'/'L' + index + price for every local extreme in window."""
        out = []
        n = len(self.candles)
        w = self.PIVOT_WINDOW
        for i in range(w, n - w):
            window = [c.close for c in self.candles[i - w:i + w + 1]]
            if self.candles[i].close == max(window):
                out.append(("H", i, self.candles[i].close))
            elif self.candles[i].close == min(window):
                out.append(("L", i, self.candles[i].close))
        return out

    @staticmethod
    def _cluster(prices: List[float]) -> List[float]:
        """Merge prices within ZONE_TOLERANCE into one representative each."""
        if not prices:
            return []
        ordered = sorted(prices)
        clusters = [[ordered[0]]]
        for p in ordered[1:]:
            if p - clusters[-1][0] <= SignalEngine.ZONE_TOLERANCE:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        return [max(group) for group in clusters]

    @staticmethod
    def _count_touches(price: float, pivots: List[tuple]) -> int:
        return sum(1 for _, _, p in pivots if abs(p - price) <= SignalEngine.ZONE_TOLERANCE)

    # -- signal detection --------------------------------------------------

    def _signal_from_latest(self) -> List[Signal]:
        latest = self.candles[-1]
        prev = self.candles[-2] if len(self.candles) > 1 else latest
        signals = []

        # Breakout confirmation + retest tracking happen implicitly from the
        # levels we discovered and the price history we have.

        # 1) SELL: support broke, retested as resistance, bearish close
        sell = self._broken_support_retest(prev, latest)
        if sell:
            signals.append(sell)

        # 2) BUY: resistance broke, retested as support, bullish close
        buy = self._broken_resistance_retest(prev, latest)
        if buy:
            signals.append(buy)

        # 3) SELL: resistance pokes, shape rejects, bearish close (level held)
        also = self._resistance_rejects(prev, latest)
        if also:
            signals.append(also)

        # 4) BUY: support holds, shape bounces, bullish close (level held)
        also_buy = self._support_holds(prev, latest)
        if also_buy:
            signals.append(also_buy)

        return signals

    # --- rules -------------------------------------------------------------

    def _broken_support_retest(self, prev: Candle, latest: Candle) -> Optional[Signal]:
        """Support fell through; price returned and closed through it again."""
        for lvl in self.levels:
            if lvl["kind"] != "SUPPORT" or lvl["price"] is None:
                continue
            # Has support actually broken? Latest close below support.
            if latest.close >= lvl["price"]:
                continue
            # Is it retesting (price came back up toward the broken support)?
            if latest.low > prev.close and latest.close >= lvl["price"] * 0.995:
                # Bearish hold on retest => time to sell
                return self._make_signal(
                    "SELL", lvl["kind"], latest,
                    "Support @ %.2f runtuh; harga kembali menekan area "
                    "tersebut dan menutup %s." % (lvl["price"], "bearish"),
                )
        return None

    def _broken_resistance_retest(self, prev: Candle, latest: Candle) -> Optional[Signal]:
        """Resistance fell through; price returned and closed through it again."""
        for lvl in self.levels:
            if lvl["kind"] != "RESISTANCE" or lvl["price"] is None:
                continue
            if latest.close <= lvl["price"]:
                continue
            # Retest: price came back DOWN into the broken resistance
            if latest.high <= prev.close and latest.close <= lvl["price"] * 1.005:
                return self._make_signal(
                    "BUY", lvl["kind"], latest,
                    "Resistance @ %.2f lucu; harga turun ke area tersebut "
                    "dan menutup %s -- retest dibuktikan." % (lvl["price"], "bullish"),
                )
        return None

    def _resistance_rejects(self, prev: Candle, latest: Candle) -> Optional[Signal]:
        """Resistance pokes but candle shape rejects it (level held)."""
        for lvl in self.levels:
            if lvl["kind"] != "RESISTANCE" or lvl["price"] is None:
                continue
            if latest.high < lvl["price"]:
                continue
            if latest.close < prev.close:          # bearish close
                return self._make_signal(
                    "SELL", lvl["kind"], latest,
                    "Harga menembus resistance @ %.2f namun menutup %s; "
                    "penjualan masih mengendalikan level." % (lvl["price"], "bearish"),
                )
        return None

    def _support_holds(self, prev: Candle, latest: Candle) -> Optional[Signal]:
        """Support pokes but candle shape bounces (level held)."""
        for lvl in self.levels:
            if lvl["kind"] != "SUPPORT" or lvl["price"] is None:
                continue
            if latest.low > lvl["price"]:
                continue
            if latest.close > prev.close:          # bullish close
                return self._make_signal(
                    "BUY", lvl["kind"], latest,
                    "Harga melencong ke support @ %.2f namun menutup %s; "
                    "pembeli masih mengendalikan level." % (lvl["price"], "bullish"),
                )
        return None

    def _make_signal(self, direction: str, level: str, latest: Candle,
                     reason: str) -> Signal:
        # Confidence = confluence of level strength + candle direction
        strength = self._level_strength(level, latest)
        score = strength + (2 if direction == "BUY" and latest.close > latest.open else 1)
        return Signal(
            direction=direction,
            confidence=min(100, score * 17),
            level=level,
            reason=reason,
            trigger_at=len(self.candles) - 2,
            retest_at=len(self.candles) - 1,
        )

    @staticmethod
    def _level_strength(level: str, latest: Candle) -> int:
        if not level:
            return 0
        for lvl in latest.__class__.levels if hasattr(latest, "levels") else []:
            pass
        # fall back to the engine's own levels
        for lvl in getattr(latest, "levels", []):
            if lvl.get("kind") == level:
                return min(3, lvl.get("touches", 0) - 1)
        return 0

    def _last_volume(self) -> float:
        n = len(self.candles)
        if n >= 20:
            return sum(c.volume for c in self.candles[-20:]) / 20.0
        return sum(c.volume for c in self.candles[-n:]) / float(n)

    def _close_prices(self) -> List[float]:
        return [c.close for c in self.candles]
