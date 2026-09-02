# Search design

How a natural-language query becomes a ranked list of files.

## The pipeline

```
query ──► bge text encoder ────► text_chunks collection ─┐
      └─► CLIP text encoder ──► images collection ───────┴─► RRF ─► per-file collapse ─► results
```

Both collections live in one embedded Qdrant store
(`QdrantClient(path=...)`) — in-process, no server, no Docker, no port.

## Why two collections

bge and OpenCLIP both emit 1024-dimensional vectors, which makes it tempting to
put them in one collection. They are **different spaces**: a bge vector and a
CLIP vector of the same sentence point nowhere near each other. Nearest-neighbour
search across a mixture of the two returns noise, and nothing about the result
would look wrong — just subtly bad. So text and images are stored separately and
each is queried with the encoder that produced it.

## Why fuse by rank rather than score

This is the part that decides whether multimodal search works.

A good text match scores about **0.7** with bge. An equally good image match
scores about **0.25** with CLIP. The numbers are not on the same scale and never
will be: they come from different models trained with different objectives and
temperatures. Comparing or averaging them would put every text result above
every image result regardless of relevance.

**Reciprocal Rank Fusion** throws the magnitudes away and keeps only each item's
position within its own list:

```
score(item) = Σ over lists  1 / (k + rank)
```

Each retriever votes with its ordering, which *is* comparable across models.
`k = 60` (the value from the original paper) damps the advantage of rank 1 so a
single list cannot monopolise the output.

`tests/test_ranking.py::test_raw_scores_do_not_decide_the_order` pins this: a
top-ranked image must outrank the second-place text hit despite scoring 0.25
against 0.68.

## Exact lookup needs a keyword index, not a better threshold

Dense vectors cannot find an identifier. Measured on a real library:

| Query: `Outward : 42595078122` | Cosine |
| --- | --- |
| The chunk that **literally contains** that number | **0.45** |
| Unrelated lab reports | 0.57 |

The right answer scored *below* the noise. bge tokenises a long digit string
into subword fragments carrying no identity, so the ranking is not merely
weak — it is inverted, and no threshold or reranking repairs an inversion.

`search/lexical.py` adds a **third retriever**: SQLite FTS5 over chunk text,
already in the database, offline, and indexing `42595078122` as a single token.
Its results are fused into the same RRF as the two vector spaces.

Three rules make it behave:

1. **Identifier tokens are required, not preferred.** A token with a digit and
   at least four characters triggers an AND search. Someone typing a reference
   number wants *that* document, not everything sharing the word beside it.
2. **Exact hits outrank semantic ones.** Without this they merely tie: a
   keyword hit at rank 1 scores `1/(60+1)`, exactly what the top dense hit
   scores, and the tie breaks arbitrarily.
3. **Semantic noise is suppressed beneath an exact hit.** A document that does
   not contain the number is not a worse answer, it is not an answer. Only
   results clearing `confident_match_score` survive alongside an exact match.

Two further details:

- **Non-searchable chunks are still searched here.** A passage is kept out of
  the *semantic* index when its embedding would be meaningless (a page of
  patent numbers). That says nothing about its literal text — an exact lookup
  should still reach it.
- **The similarity shown is real.** Keyword hits have no score of their own, so
  the stored vector is retrieved and the true cosine computed. The UI then
  ignores it and prints "exact match", because describing a literal find by its
  cosine would be actively misleading.

FTS5 uses an external-content table with triggers, so the index tracks inserts,
updates and deletes automatically; the migration backfills an existing library
without re-importing.

## The relevance floor

RRF fixes *ordering* across spaces, but it deliberately throws away magnitude —
and magnitude is the only thing that can answer "is this a match at all?".
Left alone, the store returns its N nearest neighbours however far away they
are, and every document in the library comes back for every query.

Measured on a real mixed library (recipes, resumes, a patent, field notes):

| Query | Best match | Next | Verdict |
| --- | --- | --- | --- |
| "recipe of smoked paprika" | **0.785** | 0.470 | clear winner |
| "recipe of smoked peparika" *(typo)* | **0.685** | 0.467 | still clear |
| "how far do sea birds travel" | **0.738** | 0.579 | clear |
| "quantum chromodynamics…" | 0.523 | 0.521 | nothing relevant |

bge-large does **not** spread its vectors over `[0,1]`: unrelated text scores
0.42–0.52 against any query, while a genuine match scores 0.62–0.79.
`min_text_score = 0.55` sits in that gap. A query the library cannot answer now
returns **nothing**, which is the honest response.

The threshold is applied by Qdrant itself (`score_threshold`), not after the
fact. Post-filtering would silently shrink the candidate pool — the store would
still return its N nearest, most of which get discarded, so genuinely relevant
results beyond position N would never appear.

CLIP needs its own floor (`min_image_score = 0.22`): a good image match is
~0.25–0.38 on that scale, so one shared threshold would either flood the
results with weak image hits or suppress every good one.

### Two scores, one of them for humans

`SearchResult` carries both:

- `score` — the RRF value, used for **ordering only**. Rank 1 is always
  `1/(60+1) = 0.0164` and rank 2 always `0.0161`, regardless of match quality.
  Showing it to a user is actively misleading, and doing so was the original
  symptom: an excellent match and a barely-passing one looked identical.
- `similarity` — the real cosine from the model that found the hit. This is
  what the UI displays, calibrated in words ("strong" ≥ 0.72, "good" ≥ 0.62,
  otherwise "weak") because a bare `0.56` invites the reader to assume it means
  56%.

### Low-information passages are never indexed

Some extracted text has no retrievable content: a patent's list of cited
numbers, a figure's callout labels, a diagram OCR'd as one repeated word.
Embedding it is worse than useless — such vectors sit near the centre of the
embedding space, which in high dimensions makes them a near neighbour of
*every* query.

Concretely, searching for "EEIM notes" returned this at effectively the same
score as the correct document:

> `US 11,321,312 B2 Page 2 (56) 2014/0004027 2016/0012044 2016/0012045 …`

1170 characters containing 29 words.

`ingestion/quality.py` rejects a passage on three signals:

| Signal | Threshold | Catches |
| --- | --- | --- |
| word density (characters belonging to words ÷ total) | < 0.35 | reference lists, figure sheets, diagnostic dumps |
| distinct words | < 2 | page numbers, bare labels |
| lexical diversity (distinct ÷ total words) | < 0.40 over 5+ words | `Mircroservice Clients Mircroservice Mircroservice…` indexed once per page |

Word density beats "fraction of characters that are letters": a reference list
partly passes the latter because its numbers are interleaved with words like
"References Cited U.S. PATENT DOCUMENTS". The offending chunk scores **0.197**
on density against 0.44–0.75 for the real pages of the same PDF.

Rejected chunks stay in the database — they are part of the document, and
visible in the file view — but carry `searchable = False` and never reach the
index. On a real 1131-chunk library this rejected about 1%.

`ml/maintenance.reclassify_chunks` re-applies the current rules at every
startup and deletes anything that no longer qualifies from the vector store, so
changing a threshold does not require re-importing a library (which would mean
re-running OCR and captioning — hours on CPU).

## A bare noun is not a caption

CLIP is trained on image-*caption* pairs -- "a photo of a person standing in a
park" -- and never on bare nouns. A one-word query is therefore out of
distribution: it scores lower against *every* image, not just the wrong ones,
and the relevance floor then removes the entire result set. Measured on 794
photographs:

| Query | Images above the floor |
| --- | --- |
| `human` | **0** |
| `person` | **1** |
| `black guy in yellow tshirt` | 2 (correct, rank 1) |

Nothing was wrong with the index. A full description worked perfectly while the
category word it belongs to returned nothing.

`embeddings_clip.embed_query` now encodes the query under several caption
templates and averages them. Only the query side changes, so this took effect
with no re-indexing.

| Query | raw | `"a photo of {}."` | ensemble (chosen) |
| --- | --- | --- | --- |
| person | 1 | 46 | 27 |
| human | 0 | 21 | 15 |
| vehicle | 4 | **1** | 6 |
| car | 8 | 10 | 13 |
| gibberish | 1 | 7 | 4 |

A single template finds the most people, but makes `vehicle` *worse* than no
template at all and pulls in twice the noise. The ensemble keeps the raw query
in the mix -- so a phrase that already reads like a caption is not distorted by
being wrapped in another one -- and is the only form that improves every real
query.

**This is retrieval, not classification.** "Show me every photo containing a
human" is a different question from "rank these photos by how well they match
*human*", and only the second is what a similarity search can answer. Recall
improves enormously; it does not become exhaustive.

## Two relative floors, because the two models have different spreads

`relative_score_floor` trims results far weaker than the best one. One ratio
cannot serve both spaces:

| | Typical range | 0.95 of a top hit |
| --- | --- | --- |
| bge | 0.42 - 0.79 | a wide, useful band |
| CLIP | 0.25 - 0.36 | 0.266 against a 0.280 best -- a band **0.014** wide |

Searching `person` retrieved 27 matching images and displayed **4**: the
absolute floor had already removed the non-answers, and then the relative floor
removed most of the answers. `image_relative_score_floor` (0.90) trims the
image tail on its own scale, and the same search now shows 25.

## The relevance floor is per space, not per query

RRF fixes ordering across the two spaces and then `drop_weak_tail` very nearly
undid it. The floor was a single cutoff, `best.similarity * 0.95`, applied to
every result whatever model produced it.

That re-introduces the exact bug RRF exists to prevent. Measured:

| Query: `a red Mercedes car` | Similarity | Fate |
| --- | --- | --- |
| The photograph of a red Mercedes (CLIP) | 0.308 | **dropped** |
| Unrelated notes on bioluminescence (bge) | 0.466 | kept |

A good CLIP match is ~0.30; bge *noise* is ~0.47. One cutoff taken from
whichever space happened to rank first deletes the other space wholesale, so
image results could essentially never appear in an "Everything" search --
image search looked completely broken while working perfectly.

The floor now takes the best hit **within each collection** and trims against
that, so each model's tail is judged on its own scale.
`tests/test_ranking.py::test_a_good_image_match_survives_alongside_unrelated_text`
pins it.

**Keyword hits on ordinary words go through the same floor.** A query with no
identifier in it becomes an OR over its words, and those hits never pass
through the vector store, so they carried no score floor at all: "a red
Mercedes car" returned every document containing "car" at 0.33, which the dense
retriever had already rejected at 0.55. They then *set* the relative floor and
buried the real answers. Exact identifier hits stay exempt -- the number was
found, and the model's opinion of the number is not evidence.

## There is no clean relevance threshold

Worth stating plainly, because it invites re-litigation: **a single similarity
cutoff cannot separate "no answer" from "weak answer"** with this model.

Measured over 20 queries on a real library:

| | |
| --- | --- |
| Best score for a query the library **cannot** answer | 0.572 |
| Worst score for a query it **can** answer | 0.581 |
| Usable gap | **0.009** |

Short queries score high against everything — "quantum chromodynamics" reaches
0.572 while its longer form reaches only 0.523. Tuning a threshold into a
0.009 gap is overfitting to whichever queries happened to be tested.

So the app does not pretend:

- `min_text_score` (0.55) removes what is clearly nothing.
- `relative_score_floor` (0.95 of the best hit) removes the flat tail.
- `confident_match_score` (0.62) decides whether results are presented as
  answers or as "the nearest thing I have". Below it the UI adds a caveat
  rather than suppressing the result, because a weak score frequently still
  belongs to the *correct* document — "EEIM notes" scores 0.605 and is right.

## Candidate depth

`search_candidates = 200`, well above the display limit. Results are collapsed
per file afterwards, so a single 200-page document can otherwise occupy every
candidate slot with its own chunks and crowd every other file out of the running
before grouping ever happens.

## Collapsing multiple passages from one file

A 200-page report mentioning the query on every page would otherwise fill the
entire first page of results and hide everything else. Results are grouped by
file, the strongest passage becomes the headline, and the rest are attached as
`supporting` (capped at five) so the evidence is not lost.

Scoring a grouped file is **the sum of its best hit in each space** — not the
maximum, and not the sum of everything:

| Approach | Problem |
| --- | --- |
| max | Throws away the best signal available: a file matching in *both* text and images is corroborated by two independent retrievers and should outrank one that matched once. |
| sum of all hits | Re-introduces the long-document problem — a long report accumulates hits by sheer length. |
| **sum of best-per-space** | Cross-modal agreement helps; repetition within one space does not. No tuning constant. |

## Point identity

A text point's id is its `chunks.id`; an image point's is its `assets.id`. SQLite
stays the single source of truth about what should exist, and deleting a file
addresses its points directly instead of scanning. Deleting a file removes its
vectors too — otherwise deleted documents keep turning up in results.

## Dimension safety

`ensure_collections` reads the widths from the **loaded models**, not from
config, and refuses to open a collection built at a different size. A collection
created for a 384-dim model and then written with 1024-dim vectors is a
corruption that only shows up much later, as quietly worse results.

## Queue priority

Indexing is prioritised above extraction, and both above ML processing:

| Stage | Priority | Cost |
| --- | --- | --- |
| index | 10 | ~1s — makes a file findable at all |
| extract | 20 | milliseconds |
| process (OCR + captions) | 50 | ~390s per image on CPU |

Under plain FIFO, dropping six documents in left five of them fully extracted
but **unsearchable for over six minutes**, queued behind one scanned page's
captioning. With priorities the library was searchable **15 seconds** after
upload, with captions filling in behind.

## Degrading rather than failing

| Situation | Behaviour |
| --- | --- |
| Image understanding disabled | Text results only; the UI says "documents only". |
| CLIP not downloaded | Text results only; logged, `searched_images=False`. |
| Worker holds a model (low-memory mode) | Image search skipped after a short wait; text results returned immediately. |
| Text encoder missing | 503 — there is no useful answer without it, so it surfaces. |
| Collection does not exist yet | Empty result, not an error: a text-only library has no image collection. |

bge is deliberately kept resident even under `ml_low_memory`, so a query never
waits behind a captioning pass. It is the smallest of the large models (1.3GB)
and the only one every search needs.
