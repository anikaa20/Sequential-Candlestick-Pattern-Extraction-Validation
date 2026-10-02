"""
Candlestick pattern recognition via sequence parsing and matching.

This module provides functions to parse user-defined candle color sequences
(e.g., "3R -> 2G -> Doji"), find all matching occurrences in OHLC data,
analyze pattern outcomes, and discover repeating color sequences automatically.

Key functions:
- parse_sequence(): Parse a sequence string into tokens
- find_sequence_occurrences(): Find all matches in a DataFrame
- sequence_outcome_stats(): Calculate win rate and returns for pattern matches
- discover_color_sequences(): Automatically find repeating patterns

Example:
    >>> df = pd.read_csv('data.csv')
    >>> tokens = parse_sequence('3R -> 2G')
    >>> matches = find_sequence_occurrences(df, '3R -> 2G')
    >>> stats = sequence_outcome_stats(df, matches, hold_period=5)
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from typing import List, Tuple

# Import detection helpers lazily inside functions to avoid circular imports


PatternToken = Tuple[int, str]  # (count, token) token e.g. 'R','G','Doji'


def parse_sequence(seq: str) -> List[PatternToken]:
    """Parse a sequence string like '3R -> 2G' or '5G ÔåÆ Doji ÔåÆ 4R' into tokens."""

    # split on arrow-like separators (ÔåÆ, ->, -, >)

    parts = re.split(r"\s*(?:ÔåÆ|->|>|-)\s*", seq)

    tokens: List[PatternToken] = []

    for p in parts:

        p = p.strip()

        m = re.match(r"^(\d+)([RG])$", p, re.IGNORECASE)

        if m:

            cnt = int(m.group(1))

            sym = m.group(2).upper()

            tokens.append((cnt, sym))

        else:

            # named token like Doji or Hammer; treat as count=1

            tokens.append((1, p))

    return tokens


def symbol_sequence(df: pd.DataFrame) -> List[str]:
    """Return simple symbol per candle: 'G' or 'R' or 'Doji' if detected."""

    syms: List[str] = []

    # import helpers here to avoid circular imports

    from .detection import is_doji, candle_color

    for i in range(len(df)):

        row = df.iloc[i]

        if is_doji(df.iloc[i : i + 1]):

            syms.append("Doji")

            continue

        c = candle_color(row)

        syms.append("G" if c == "green" else "R")

    return syms


def match_named_token(df: pd.DataFrame, idx: int, token: str) -> bool:
    """Check if token matches candle at idx. Token may be 'Doji','Hammer',
    'Engulfing', 'MorningStar', 'EveningStar', etc."""

    # import helpers here to avoid circular imports

    from .detection import is_doji, is_hammer

    token_low = token.lower()

    # named tokens used for pattern matching (not passwords) - B105 false positive
    if token_low == "doji":  # nosec B105

        return bool(is_doji(df.iloc[idx : idx + 1]))

    if token_low == "hammer":  # nosec B105

        return bool(is_hammer(df.iloc[idx : idx + 1]))

    # --- Engulfing patterns (need current + previous candle) ---
    if token_low == "engulfing" and idx >= 1:
        prev = df.iloc[idx - 1]
        curr = df.iloc[idx]
        bull = (
            prev["close"] < prev["open"]
            and curr["close"] > curr["open"]
            and (curr["close"] - curr["open"]) >= (prev["open"] - prev["close"])
        )
        bear = (
            prev["close"] > prev["open"]
            and curr["close"] < curr["open"]
            and (curr["open"] - curr["close"]) >= (prev["close"] - prev["open"])
        )
        return bool(bull or bear)

    if token_low == "bullengulfing" and idx >= 1:
        prev = df.iloc[idx - 1]
        curr = df.iloc[idx]
        return bool(
            prev["close"] < prev["open"]
            and curr["close"] > curr["open"]
            and (curr["close"] - curr["open"]) >= (prev["open"] - prev["close"])
        )

    if token_low == "bearengulfing" and idx >= 1:
        prev = df.iloc[idx - 1]
        curr = df.iloc[idx]
        return bool(
            prev["close"] > prev["open"]
            and curr["close"] < curr["open"]
            and (curr["open"] - curr["close"]) >= (prev["close"] - prev["open"])
        )

    # --- Morning Star (need idx-2, idx-1, idx) ---
    if token_low == "morningstar" and idx >= 2:
        c0, c1, c2 = df.iloc[idx - 2], df.iloc[idx - 1], df.iloc[idx]
        return bool(
            c0["close"] < c0["open"]  # bearish
            and abs(c1["close"] - c1["open"])
            <= 0.25 * (c1["high"] - c1["low"])  # small body
            and c2["close"] > c2["open"]  # bullish
        )

    # --- Evening Star ---
    if token_low == "eveningstar" and idx >= 2:
        c0, c1, c2 = df.iloc[idx - 2], df.iloc[idx - 1], df.iloc[idx]
        return bool(
            c0["close"] > c0["open"]
            and abs(c1["close"] - c1["open"]) <= 0.25 * (c1["high"] - c1["low"])
            and c2["close"] < c2["open"]
        )

    # --- Shooting Star ---
    if token_low == "shootingstar":
        row = df.iloc[idx]
        body = abs(row["close"] - row["open"])
        upper_wick = row["high"] - max(row["open"], row["close"])
        return bool(body > 0 and upper_wick >= 2 * body)

    # --- Spinning Top ---
    if token_low == "spinningtop":
        row = df.iloc[idx]
        body = abs(row["close"] - row["open"])
        rng = row["high"] - row["low"]
        return bool(rng > 0 and body <= 0.2 * rng)

    # --- Inverted Hammer (small body at top, long lower shadow) ---
    if token_low == "invertedhammer":
        row = df.iloc[idx]
        body = abs(row["close"] - row["open"])
        upper_wick = row["high"] - max(row["open"], row["close"])
        lower_wick = min(row["open"], row["close"]) - row["low"]
        return bool(body > 0 and upper_wick >= 2 * body and lower_wick <= body)

    # --- Marubozu (full body, no/tiny wicks — bullish or bearish) ---
    if token_low == "marubozu":
        row = df.iloc[idx]
        body = abs(row["close"] - row["open"])
        rng = row["high"] - row["low"]
        return bool(rng > 0 and body >= 0.90 * rng)

    if token_low == "bullmarubozu":
        row = df.iloc[idx]
        body = abs(row["close"] - row["open"])
        rng = row["high"] - row["low"]
        return bool(rng > 0 and body >= 0.90 * rng and row["close"] > row["open"])

    if token_low == "bearmarubozu":
        row = df.iloc[idx]
        body = abs(row["close"] - row["open"])
        rng = row["high"] - row["low"]
        return bool(rng > 0 and body >= 0.90 * rng and row["close"] < row["open"])

    # --- Three White Soldiers (need idx-2, idx-1, idx — three consecutive bullish) ---
    if token_low == "threewhitesoldiers" and idx >= 2:
        c0, c1, c2 = df.iloc[idx - 2], df.iloc[idx - 1], df.iloc[idx]
        return bool(
            c0["close"] > c0["open"]
            and c1["close"] > c1["open"]
            and c2["close"] > c2["open"]
            and c1["close"] > c0["close"]
            and c2["close"] > c1["close"]
        )

    # --- Three Black Crows (need idx-2, idx-1, idx — three consecutive bearish) ---
    if token_low == "threeblackcrows" and idx >= 2:
        c0, c1, c2 = df.iloc[idx - 2], df.iloc[idx - 1], df.iloc[idx]
        return bool(
            c0["close"] < c0["open"]
            and c1["close"] < c1["open"]
            and c2["close"] < c2["open"]
            and c1["close"] < c0["close"]
            and c2["close"] < c1["close"]
        )

    # fallback to direction tokens handled elsewhere

    return False


def matches_sequence_at(df: pd.DataFrame, start_idx: int, seq_str: str) -> bool:
    """Check if a sequence matches starting at a specific index.

    Validates that a user-defined sequence (e.g., '3R -> 2G') occurs exactly
    at the given start position in the DataFrame.

    Args:
        df: pandas DataFrame with OHLC columns.
        start_idx: int, the starting index to check in the DataFrame.
        seq_str: str, sequence string (e.g., '3R -> 2G -> Doji').

    Returns:
        bool: True if the entire sequence matches starting at start_idx,
              False otherwise.

    Example:
        >>> df = pd.read_csv('data.csv')
        >>> matches_sequence_at(df, 10, '2R -> 3G')  → True if match at idx 10
    """
    tokens = parse_sequence(seq_str)
    syms = symbol_sequence(df)
    n = len(df)
    i = start_idx

    for cnt, tok in tokens:
        if tok in ("R", "G"):
            for k in range(cnt):
                if i + k >= n or syms[i + k] != tok:
                    return False
            i += cnt
        else:
            if i >= n or not match_named_token(df, i, tok):
                return False
            i += 1
    return True


def count_followup_pattern(
    df: pd.DataFrame, base_seq: str, follow_seq: str, follow_len: int
) -> dict:
    """Count how many base sequence occurrences are followed by a follow-up sequence.

    Finds all occurrences of base_seq and checks if follow_seq appears
    within follow_len candles after each match. Computes success rate.

    Args:
        df: pandas DataFrame with OHLC columns.
        base_seq: str, the base pattern sequence (e.g., '3R -> 2G').
        follow_seq: str, the expected follow-up sequence.
        follow_len: int, max number of candles to search for follow_seq.

    Returns:
        dict with keys:
            - 'base_seq': original base sequence
            - 'follow_seq': original follow sequence
            - 'follow_len': lookback length used
            - 'total_matches': count of base_seq occurrences
            - 'followup_success': count where follow_seq appeared
            - 'followup_rate': float, success rate (0.0 to 1.0)

    Example:
        >>> df = pd.read_csv('data.csv')
        >>> result = count_followup_pattern(df, '3R', '2G', 5)
        >>> print(f"Success rate: {result['followup_rate']:.2%}")
    """
    occ_ends = find_sequence_occurrences(df, base_seq)
    total = len(occ_ends)
    success = 0

    if not follow_seq or follow_len is None or follow_len <= 0:
        return {
            "base_seq": base_seq,
            "follow_seq": follow_seq,
            "follow_len": follow_len,
            "total_matches": total,
            "followup_success": 0,
            "followup_rate": 0.0,
        }

    n = len(df)
    for end_idx in occ_ends:
        start_follow = end_idx + 1
        search_end = min(n, start_follow + follow_len)
        matched = False

        for pos in range(start_follow, search_end):
            if matches_sequence_at(df, pos, follow_seq):
                matched = True
                break

        if matched:
            success += 1

    rate = (success / total) if total > 0 else 0.0

    return {
        "base_seq": base_seq,
        "follow_seq": follow_seq,
        "follow_len": follow_len,
        "total_matches": total,
        "followup_success": success,
        "followup_rate": rate,
    }


def find_followup_outcomes(
    df: pd.DataFrame, base_seq: str, max_follow_len: int = 5, top_k: int = 3
) -> List[dict]:
    """Return the top follow-up sequences that occur after a base pattern.

    Finds all occurrences of base_seq and analyzes what color sequences
    typically follow them. Returns the most common follow-up patterns.

    Args:
        df: pandas DataFrame with OHLC columns.
        base_seq: str, the base pattern sequence (e.g., '3R -> 2G').
        max_follow_len: int, max length of follow-up sequences to analyze (default 5).
        top_k: int, return top K most common follow-ups (default 3).

    Returns:
        List[dict]: List of follow-up outcomes, each dict contains:
            - 'followup': tuple of symbols representing the sequence
            - 'count': int, how many times this follow-up occurred
            - 'rate': float, frequency (count / total_matches)

    Example:
        >>> df = pd.read_csv('data.csv')
        >>> outcomes = find_followup_outcomes(df, '3R', top_k=5)
        >>> for outcome in outcomes:
        ...     print(f"{outcome['followup']}: {outcome['rate']:.1%}")
    """
    occ_ends = find_sequence_occurrences(df, base_seq)
    syms = symbol_sequence(df)
    n = len(syms)
    counter = Counter()

    for end_idx in occ_ends:
        start = end_idx + 1
        if start >= n:
            continue
        window = syms[start : min(n, start + max_follow_len)]
        if not window:
            continue
        outcome = _run_length_encode(tuple(window))
        counter[outcome] += 1

    total = len(occ_ends)
    return [
        {
            "followup": outcome,
            "count": count,
            "rate": (count / total if total > 0 else 0.0),
        }
        for outcome, count in counter.most_common(top_k)
    ]


def find_sequence_occurrences(df: pd.DataFrame, seq_str: str) -> List[int]:
    """Find all occurrences of a sequence pattern in historical data.

    Scans the entire DataFrame for matches of the given sequence string
    and returns the ending index for each match.

    Args:
        df: pandas DataFrame with OHLC columns.
        seq_str: str, the sequence pattern to search for (e.g., '3R -> 2G -> Doji').

    Returns:
        List[int]: List of ending indices where the sequence fully matches.
                   Empty list if no matches found.

    Example:
        >>> df = pd.read_csv('data.csv')
        >>> matches = find_sequence_occurrences(df, '3R -> 2G')
        >>> print(f"Found {len(matches)} matches at indices: {matches}")
        >>> for end_idx in matches:
        ...     print(f"Pattern ends at row {end_idx}: {df.iloc[end_idx]['timestamp']}")
    """

    results: List[int] = []
    n = len(df)

    for start in range(0, n):
        if matches_sequence_at(df, start, seq_str):
            tokens = parse_sequence(seq_str)
            total_len = sum(cnt for cnt, _ in tokens)
            end_idx = start + total_len - 1
            if end_idx < n:
                results.append(end_idx)

    return results


# ============================================================================
# WILDCARD SEQUENCE MATCHING  —  "3R -> * -> 2G"
# ============================================================================


def find_wildcard_sequence(
    df: pd.DataFrame,
    seq_str: str,
    wildcard_min: int = 1,
    wildcard_max: int = 3,
) -> List[dict]:
    """Match a sequence containing ``*`` wildcards that span 1-N candles.

    A ``*`` in the sequence matches *any* ``wildcard_min`` to ``wildcard_max``
    candles regardless of colour/type.

    Returns a list of dicts::

        [{"start_idx": 2, "end_idx": 9, "wildcard_span": 2}, ...]
    """
    parts = re.split(r"\s*(?:→|->|>|-)\s*", seq_str)

    # Split into segments separated by wildcards
    segments: List = []  # Each is either ("tokens", [...]) or ("wild",)
    for p in parts:
        p = p.strip()
        if p == "*":
            segments.append(("wild",))
        else:
            m = re.match(r"^(\d+)([RG])$", p, re.IGNORECASE)
            if m:
                segments.append(("tokens", [(int(m.group(1)), m.group(2).upper())]))
            else:
                segments.append(("tokens", [(1, p)]))

    syms = symbol_sequence(df)
    n = len(syms)
    results: List[dict] = []

    def _match_from(pos: int, seg_idx: int) -> List[dict]:
        """Recursive matcher that yields list of match dicts."""
        if seg_idx >= len(segments):
            return [{"end": pos - 1, "wild_span": 0}]
        seg = segments[seg_idx]
        if seg[0] == "wild":
            hits: List[dict] = []
            for span in range(wildcard_min, wildcard_max + 1):
                if pos + span > n:
                    break
                sub = _match_from(pos + span, seg_idx + 1)
                for h in sub:
                    hits.append({"end": h["end"], "wild_span": h["wild_span"] + span})
            return hits
        else:
            # token segment — must match exactly
            i = pos
            for cnt, tok in seg[1]:
                if tok in ("R", "G"):
                    for k in range(cnt):
                        if i + k >= n or syms[i + k] != tok:
                            return []
                    i += cnt
                else:
                    if i >= n or not match_named_token(df, i, tok):
                        return []
                    i += 1
            return _match_from(i, seg_idx + 1)

    for start in range(n):
        hits = _match_from(start, 0)
        for h in hits:
            results.append(
                {
                    "start_idx": start,
                    "end_idx": h["end"],
                    "wildcard_span": h["wild_span"],
                }
            )

    return results


# ============================================================================
# WHAT-COMES-NEXT PREDICTION
# ============================================================================


def what_comes_next(
    df: pd.DataFrame,
    seq_str: str,
    lookahead: int = 3,
) -> dict:
    """After every occurrence of *seq_str*, collect the next *lookahead* candle
    colours and tally the distribution.

    Returns::

        {
            "total_occurrences": 8,
            "lookahead": 3,
            "distribution": [
                {"position": 1, "R": 5, "G": 2, "Doji": 1, "R_pct": 0.625, "G_pct": 0.25, "Doji_pct": 0.125},
                ...
            ],
            "most_likely_next": "3R"   # run-length encoded most-common continuation
        }
    """
    ends = find_sequence_occurrences(df, seq_str)
    syms = symbol_sequence(df)
    n = len(syms)

    dist: List[dict] = []

    for pos in range(lookahead):
        counter: dict = {"R": 0, "G": 0, "Doji": 0}
        for end_idx in ends:
            nxt = end_idx + 1 + pos
            if nxt < n:
                s = syms[nxt]
                counter[s] = counter.get(s, 0) + 1
        total = sum(counter.values())
        entry = {"position": pos + 1}
        for k in ("R", "G", "Doji"):
            entry[k] = counter.get(k, 0)
            entry[f"{k}_pct"] = round(counter.get(k, 0) / max(1, total), 4)
        dist.append(entry)

    # Build most-likely continuation string
    likely: List[str] = []
    for d in dist:
        best = max(("R", "G", "Doji"), key=lambda k: d.get(k, 0))
        if d.get(best, 0) > 0:
            likely.append(best)
    most_likely = _run_length_encode(tuple(likely)) if likely else ""

    return {
        "total_occurrences": len(ends),
        "lookahead": lookahead,
        "distribution": dist,
        "most_likely_next": most_likely,
    }


# ============================================================================
# SEQUENCE OUTCOME STATISTICS  —  price movement after matches
# ============================================================================


def sequence_outcome_stats(
    df: pd.DataFrame,
    seq_str: str,
    hold_candles: int = 5,
) -> dict:
    """Compute price statistics for *hold_candles* after each occurrence of seq_str.

    Returns::

        {
            "sequence": "3R -> 2G",
            "occurrences": 6,
            "avg_return_pct": 0.45,
            "median_return_pct": 0.32,
            "win_rate": 0.667,
            "max_gain_pct": 2.1,
            "max_loss_pct": -1.3,
            "avg_high_pct": 1.2,
            "avg_low_pct": -0.8,
        }
    """
    ends = find_sequence_occurrences(df, seq_str)
    n = len(df)

    returns: List[float] = []
    highs: List[float] = []
    lows: List[float] = []

    for end_idx in ends:
        entry_idx = end_idx + 1
        if entry_idx >= n:
            continue
        entry_close = float(df.iloc[entry_idx]["close"])
        exit_idx = min(entry_idx + hold_candles, n - 1)
        if exit_idx <= entry_idx:
            continue

        # Hold period metrics
        window = df.iloc[entry_idx : exit_idx + 1]
        exit_close = float(window.iloc[-1]["close"])
        ret_pct = (exit_close - entry_close) / entry_close * 100

        max_high = float(window["high"].max())
        min_low = float(window["low"].min())
        high_pct = (max_high - entry_close) / entry_close * 100
        low_pct = (min_low - entry_close) / entry_close * 100

        returns.append(round(ret_pct, 4))
        highs.append(round(high_pct, 4))
        lows.append(round(low_pct, 4))

    if not returns:
        return {
            "sequence": seq_str,
            "occurrences": len(ends),
            "avg_return_pct": 0,
            "median_return_pct": 0,
            "win_rate": 0,
            "max_gain_pct": 0,
            "max_loss_pct": 0,
            "avg_high_pct": 0,
            "avg_low_pct": 0,
        }

    sorted_rets = sorted(returns)
    mid = len(sorted_rets) // 2
    median = (
        sorted_rets[mid]
        if len(sorted_rets) % 2
        else (sorted_rets[mid - 1] + sorted_rets[mid]) / 2
    )

    wins = sum(1 for r in returns if r > 0)

    return {
        "sequence": seq_str,
        "occurrences": len(ends),
        "avg_return_pct": round(sum(returns) / len(returns), 4),
        "median_return_pct": round(median, 4),
        "win_rate": round(wins / len(returns), 4),
        "max_gain_pct": round(max(returns), 4),
        "max_loss_pct": round(min(returns), 4),
        "avg_high_pct": round(sum(highs) / len(highs), 4),
        "avg_low_pct": round(sum(lows) / len(lows), 4),
    }


def sequence_length(seq_str: str) -> int:
    """Return the total number of candles consumed by a sequence string."""
    tokens = parse_sequence(seq_str)
    return sum(cnt for cnt, _ in tokens)


def _run_length_encode(syms: tuple) -> str:
    """Convert ('R','R','R','G','G') to '3R -> 2G'."""
    if not syms:
        return ""
    parts: List[str] = []
    current = syms[0]
    count = 1
    for s in syms[1:]:
        if s == current:
            count += 1
        else:
            parts.append(f"{count}{current}" if current in ("R", "G") else current)
            current = s
            count = 1
    parts.append(f"{count}{current}" if current in ("R", "G") else current)
    return " -> ".join(parts)


def discover_color_sequences(
    df: pd.DataFrame,
    min_len: int = 3,
    max_len: int = 8,
    top_k: int = 20,
) -> List[dict]:
    """Auto-discover the most common R/G colour sequences in *df*.

    Returns a list of dicts sorted by count descending::

        [{"sequence": "3R -> 2G", "count": 12, "length": 5, "support": 0.061}, ...]
    """

    syms = symbol_sequence(df)
    n = len(syms)

    # Count raw sub-sequences of each window length
    raw_counts: dict = {}
    for length in range(min_len, max_len + 1):
        for start in range(n - length + 1):
            window = tuple(syms[start : start + length])
            raw_counts[window] = raw_counts.get(window, 0) + 1

    # Merge into run-length-encoded strings
    encoded_counts: dict = {}
    for seq_tuple, count in raw_counts.items():
        encoded = _run_length_encode(seq_tuple)
        encoded_counts[encoded] = encoded_counts.get(encoded, 0) + count

    # Sort by count desc
    sorted_seqs = sorted(encoded_counts.items(), key=lambda x: x[1], reverse=True)

    results: List[dict] = []
    for seq_str, count in sorted_seqs[:top_k]:
        length = sequence_length(seq_str)
        support = count / max(1, n - length + 1)
        results.append(
            {
                "sequence": seq_str,
                "count": count,
                "length": length,
                "support": round(support, 4),
            }
        )

    return results


# ============================================================================
# REVERSE PATTERN FINDER  —  "What sequences preceded big moves?"
# ============================================================================


def reverse_pattern_finder(
    df: pd.DataFrame,
    threshold_pct: float = 1.5,
    direction: str = "up",
    lookback: int = 5,
    min_len: int = 3,
    max_len: int = 6,
    top_k: int = 15,
) -> List[dict]:
    """Find the most common colour sequences that *preceded* big price moves.

    Instead of "does this pattern lead to gains?" this answers
    "what patterns came before big gains/losses?"

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with ``close`` and ``timestamp`` columns.
    threshold_pct : float
        Minimum absolute return (%) over the move candle to qualify as a
        "big move".  Default 1.5%.
    direction : str
        ``"up"`` for big gains, ``"down"`` for big losses, ``"both"`` for
        either.
    lookback : int
        How many candles *before* the big-move candle to inspect for patterns.
    min_len : int
        Minimum sequence length to consider.
    max_len : int
        Maximum sequence length to consider.
    top_k : int
        Return the top-k most common preceding sequences.

    Returns
    -------
    list[dict]
        Each dict: ``{"sequence": "3R -> 2G", "count": 5, "length": 5,
        "avg_move_pct": 2.3, "direction": "up"}``
    """
    n = len(df)
    if n < lookback + 2:
        return []

    syms = symbol_sequence(df)

    # Identify big-move candle indices
    big_move_indices: List[int] = []
    for i in range(1, n):
        prev_close = float(df.iloc[i - 1]["close"])
        curr_close = float(df.iloc[i]["close"])
        if prev_close == 0:
            continue
        ret = (curr_close - prev_close) / prev_close * 100

        if direction == "up" and ret >= threshold_pct:
            big_move_indices.append(i)
        elif direction == "down" and ret <= -threshold_pct:
            big_move_indices.append(i)
        elif direction == "both" and abs(ret) >= threshold_pct:
            big_move_indices.append(i)

    if not big_move_indices:
        return []

    # For each big-move index, collect all sub-sequences in the lookback window
    seq_counter: dict = {}  # encoded_seq -> list of move percentages
    for move_idx in big_move_indices:
        # lookback window ends just before the big-move candle
        window_end = move_idx  # exclusive
        window_start = max(0, window_end - lookback)
        window_syms = syms[window_start:window_end]

        if len(window_syms) < min_len:
            continue

        # Compute the move percentage for context
        prev_close = float(df.iloc[move_idx - 1]["close"])
        curr_close = float(df.iloc[move_idx]["close"])
        move_pct = (curr_close - prev_close) / prev_close * 100

        # Extract all sub-sequences from the window
        for length in range(min_len, min(max_len + 1, len(window_syms) + 1)):
            for start in range(len(window_syms) - length + 1):
                sub = tuple(window_syms[start : start + length])
                encoded = _run_length_encode(sub)
                if encoded not in seq_counter:
                    seq_counter[encoded] = []
                seq_counter[encoded].append(move_pct)

    if not seq_counter:
        return []

    # Sort by frequency, then by average move magnitude
    sorted_seqs = sorted(
        seq_counter.items(),
        key=lambda x: (len(x[1]), abs(sum(x[1]) / len(x[1]))),
        reverse=True,
    )

    results: List[dict] = []
    for seq_str, moves in sorted_seqs[:top_k]:
        avg_move = sum(moves) / len(moves)
        results.append(
            {
                "sequence": seq_str,
                "count": len(moves),
                "length": sequence_length(seq_str),
                "avg_move_pct": round(avg_move, 4),
                "direction": direction,
            }
        )

    return results


# ============================================================================
# SEQUENCE CONFIDENCE SCORING  —  statistical significance
# ============================================================================


def sequence_confidence(
    df: pd.DataFrame,
    seq_str: str,
    hold_candles: int = 5,
    baseline_samples: int = 1000,
) -> dict:
    """Compute statistical confidence metrics for a sequence's outcome.

    Compares the sequence's average return against random baseline samples
    to assess whether the pattern's edge is statistically significant.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.
    seq_str : str
        Sequence string.
    hold_candles : int
        Candles to hold after each match.
    baseline_samples : int
        Number of random starting points for baseline comparison.

    Returns
    -------
    dict
        ``{"z_score": float, "p_value": float, "confidence_level": str,
        "sequence_avg": float, "baseline_avg": float, "baseline_std": float,
        "sample_size": int, "is_significant": bool}``
    """
    import math

    stats = sequence_outcome_stats(df, seq_str, hold_candles=hold_candles)
    occurrences = stats["occurrences"]
    seq_avg = stats["avg_return_pct"]

    if occurrences < 2:
        return {
            "z_score": 0.0,
            "p_value": 1.0,
            "confidence_level": "Insufficient data",
            "sequence_avg": seq_avg,
            "baseline_avg": 0.0,
            "baseline_std": 0.0,
            "sample_size": occurrences,
            "is_significant": False,
        }

    # Compute baseline: random entry returns
    n = len(df)
    import random

    random.seed(42)  # Deterministic for reproducibility

    baseline_returns: List[float] = []
    max_start = n - hold_candles - 1
    if max_start < 1:
        max_start = 1

    for _ in range(min(baseline_samples, max_start)):
        idx = random.randint(0, max_start)
        entry_close = float(df.iloc[idx]["close"])
        exit_close = float(df.iloc[min(idx + hold_candles, n - 1)]["close"])
        if entry_close > 0:
            ret = (exit_close - entry_close) / entry_close * 100
            baseline_returns.append(ret)

    if len(baseline_returns) < 2:
        return {
            "z_score": 0.0,
            "p_value": 1.0,
            "confidence_level": "Insufficient baseline",
            "sequence_avg": seq_avg,
            "baseline_avg": 0.0,
            "baseline_std": 0.0,
            "sample_size": occurrences,
            "is_significant": False,
        }

    baseline_avg = sum(baseline_returns) / len(baseline_returns)
    variance = sum((r - baseline_avg) ** 2 for r in baseline_returns) / (
        len(baseline_returns) - 1
    )
    baseline_std = math.sqrt(variance) if variance > 0 else 0.001

    # Z-score: how many standard deviations the sequence avg is from baseline
    z_score = (seq_avg - baseline_avg) / (baseline_std / math.sqrt(occurrences))

    # Approximate p-value from z-score (two-tailed)
    # Using the complementary error function approximation
    abs_z = abs(z_score)
    # Abramowitz & Stegun approximation for normal CDF
    t = 1.0 / (1.0 + 0.2316419 * abs_z)
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    p_tail = (
        d
        * math.exp(-abs_z * abs_z / 2.0)
        * (
            0.319381530 * t
            - 0.356563782 * t**2
            + 1.781477937 * t**3
            - 1.821255978 * t**4
            + 1.330274429 * t**5
        )
    )
    p_value = 2.0 * p_tail  # two-tailed

    # Confidence level interpretation
    if p_value < 0.01:
        confidence_level = "Very High (p < 0.01)"
    elif p_value < 0.05:
        confidence_level = "High (p < 0.05)"
    elif p_value < 0.10:
        confidence_level = "Moderate (p < 0.10)"
    else:
        confidence_level = "Low (p >= 0.10)"

    return {
        "z_score": round(z_score, 4),
        "p_value": round(p_value, 6),
        "confidence_level": confidence_level,
        "sequence_avg": round(seq_avg, 4),
        "baseline_avg": round(baseline_avg, 4),
        "baseline_std": round(baseline_std, 4),
        "sample_size": occurrences,
        "is_significant": p_value < 0.05,
    }


# ============================================================================
# SEQUENCE HEATMAP DATA  —  density of matches over time
# ============================================================================


def sequence_heatmap_data(
    df: pd.DataFrame,
    seq_strs: List[str],
    bucket_size: int = 10,
) -> dict:
    """Compute pattern match density over time for heatmap visualisation.

    Divides the candle data into buckets and counts matches per bucket
    per sequence.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.
    seq_strs : list[str]
        List of sequence strings to analyse.
    bucket_size : int
        Number of candles per time bucket.

    Returns
    -------
    dict
        ``{"buckets": [...], "sequences": {"3R -> 2G": [count_per_bucket, ...], ...},
        "timestamps": [bucket_start_timestamps]}``
    """
    n = len(df)
    num_buckets = max(1, (n + bucket_size - 1) // bucket_size)

    result = {
        "buckets": list(range(num_buckets)),
        "timestamps": [],
        "sequences": {},
    }

    # Compute bucket start timestamps
    for b in range(num_buckets):
        idx = b * bucket_size
        if idx < n:
            result["timestamps"].append(str(df.iloc[idx]["timestamp"]))
        else:
            result["timestamps"].append("")

    for seq_str in seq_strs:
        counts = [0] * num_buckets
        try:
            if "*" in seq_str:
                hits = find_wildcard_sequence(df, seq_str)
                for h in hits:
                    bucket = h["start_idx"] // bucket_size
                    if bucket < num_buckets:
                        counts[bucket] += 1
            else:
                ends = find_sequence_occurrences(df, seq_str)
                seq_len = sequence_length(seq_str)
                for end_idx in ends:
                    start = end_idx - seq_len + 1
                    bucket = max(0, start) // bucket_size
                    if bucket < num_buckets:
                        counts[bucket] += 1
        except Exception:
            pass
        result["sequences"][seq_str] = counts

    return result
