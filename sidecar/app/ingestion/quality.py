"""Deciding whether a passage carries enough meaning to be worth indexing.

Some extracted text has no retrievable content: a patent's list of cited
numbers, a figure's callout labels, a diagram whose OCR is one word repeated.
Embedding it is worse than useless. Such passages land near the centre of the
embedding space, which in high dimensions makes them a near neighbour of
*every* query -- so they surface against unrelated searches at scores
indistinguishable from real matches.

Observed: searching a library for "EEIM notes" returned, at essentially the
same score as the correct document, this chunk from a patent PDF:

    US 11,321,312 B2 Page 2 (56) 2014/0004027 2016/0012044 2016/0012045 ...

400 characters containing five words.

Measured across a real 1131-chunk library, the rules below reject about 1% of
chunks, all of them noise of that kind.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Fraction of a passage's characters that belong to actual words.
#
# This is a better discriminator than "how many characters are letters":
# a reference list partly passes that test, because the patent numbers are
# interleaved with words like "References Cited U.S. PATENT DOCUMENTS". The
# chunk that triggered all this reads as 1170 characters containing 29 words,
# and scores 0.197 here against 0.44-0.75 for the real pages of the same PDF.
#
# Measured across a real library: everything below 0.35 was page furniture,
# figure callouts, reference lists or raw diagnostic output; ordinary prose is
# above 0.55.
MIN_WORD_DENSITY = 0.35

# A passage needs at least a couple of distinct words to mean anything. Kept
# low so short but genuine OCR survives -- "SCANNED INVOICE / Total due:
# 4820.00" has four.
MIN_DISTINCT_WORDS = 2

# Below this many words, repetition is not evidence of junk: a short label may
# legitimately repeat a term.
REPETITION_MIN_WORDS = 5

# Distinct words as a fraction of all words. "Mircroservice Clients
# Mircroservice Mircroservice Mircroservice Mircroservice" scores 0.33; real
# prose is above 0.7.
MIN_LEXICAL_DIVERSITY = 0.40

_WORD = re.compile(r"[A-Za-z]{3,}")


@dataclass(frozen=True)
class QualityVerdict:
    searchable: bool
    reason: str = ""


def assess(text: str) -> QualityVerdict:
    """Judge whether a passage is worth putting in the search index."""
    stripped = text.strip()
    if not stripped:
        return QualityVerdict(False, "empty")

    words = _WORD.findall(stripped)
    density = sum(len(word) for word in words) / len(stripped)
    if density < MIN_WORD_DENSITY:
        return QualityVerdict(False, f"little actual text ({density:.0%} words)")

    distinct = {word.lower() for word in words}
    if len(distinct) < MIN_DISTINCT_WORDS:
        return QualityVerdict(False, f"only {len(distinct)} distinct word(s)")

    if len(words) >= REPETITION_MIN_WORDS:
        diversity = len(distinct) / len(words)
        if diversity < MIN_LEXICAL_DIVERSITY:
            return QualityVerdict(False, f"one phrase repeated ({diversity:.0%} distinct)")

    return QualityVerdict(True)


def is_searchable(text: str) -> bool:
    return assess(text).searchable
