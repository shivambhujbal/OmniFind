# A3 Poster Brief — Local Multimodal File Search

A generation brief for an image model. It describes **one** A3 portrait academic
project poster for the *finalized* system design — the complete target
architecture, not the current development state.

Every quoted string below is **verbatim poster copy**. Reproduce the spelling
exactly. Do not paraphrase, translate, expand or invent text.

---

## 1. Canvas

| Property | Value |
| --- | --- |
| Size | A3 portrait, 297 × 420 mm |
| Pixels | 3508 × 4961 px at 300 DPI |
| Margin | 18 mm on all sides, nothing critical outside it |
| Output | One single page. No spreads, no mockups, no poster-on-a-wall scenes. |

## 2. Visual direction

**Palette** — cool technical neutrals, three accent hues that each mean one
thing and are used *only* for that meaning.

| Role | Hex | Used for |
| --- | --- | --- |
| Ground | `#F2F3EF` | Poster background (a soft green-grey, not pure white) |
| Panel | `#FFFFFF` | Card and diagram fills |
| Ink | `#1B1E22` | Headings and body text |
| Muted ink | `#4A5057` | Captions, secondary text |
| Rule | `#D3D7CD` | Hairline borders, 1 pt |
| Indigo | `#3F53A0` | The **text / semantic** path, everywhere it appears |
| Violet | `#7A5399` | The **image / visual** path, everywhere it appears |
| Green | `#2C7458` | The **exact keyword** path, and "complete" states |
| Burnt orange | `#A8501F` | Reserved for measured problem figures only |

**Typography** — IBM Plex family throughout.

- Headings: IBM Plex Serif SemiBold
- Body and labels: IBM Plex Sans Regular / Medium
- All numbers, code and technical labels: IBM Plex Mono

**Style** — flat vector, clean engineering-documentation look. Thin hairline
rules, rectangular boxes with square corners, generous white space.

**Do not use:** gradients, drop shadows, glows, 3D, isometric illustration,
neon, circuit-board or "AI brain" imagery, stock photography, emoji, robot
icons, rounded pill-shaped cards, or a dark background.

## 3. Layout — eight zones on a three-column grid

Read top to bottom. Column widths are of the live area inside the margins.

```
+----------------------------------------------+
|  1  TITLE BAND                    full width |
+---------------+------------------------------+
| 2  PROBLEM    |  3  SYSTEM ARCHITECTURE      |
|    1 col      |     2 cols                   |
+---------------+------------------------------+
|  4  INGESTION PIPELINE            full width |
+----------------------------------------------+
|  5  SEARCH PIPELINE               full width |
+---------------+------------------------------+
| 6  TECH STACK |  7  DESIGN DECISIONS         |
|    1 col      |     2 cols                   |
+---------------+------------------------------+
|  8  PROJECT PHASES                full width |
+----------------------------------------------+
```

Approximate vertical share, top to bottom: 10 / 20 / 18 / 20 / 16 / 8 percent.

---

## 4. Zone content

### Zone 1 — Title band

Left-aligned, white panel across the full width.

- Eyebrow, mono uppercase, letter-spaced, muted: `FINAL YEAR PROJECT`
- Title, IBM Plex Serif, the largest type on the poster:
  **Local Multimodal File Search**
- Subtitle, sans, ink: *Find your files by what is inside them — not by their name*
- A row of four small mono badges with hairline borders. The first has a green
  border and green text; the rest are neutral:
  `100% OFFLINE` · `NO SERVER` · `NO ACCOUNT` · `ONE WINDOWS INSTALLER`

### Zone 2 — The problem

Heading: **The problem**

> Windows Search finds filenames. It cannot find meaning.
>
> A scanned certificate saved as `IMG_20230417.pdf` is invisible to every search
> tool on the machine — the page is a picture, so there is no text in the file
> at all.

Heading: **The objective**

> Search personal PDFs, Word documents and images by their content, using
> ordinary language, entirely on one Windows laptop — with no internet
> connection, no account and no cloud service.

### Zone 3 — System architecture

A vertical stack of full-width boxes, white with hairline borders. Boxes touch;
no gaps between them.

1. Label `DESKTOP SHELL` — title **Tauri (Rust)** — caption *Window and process
   lifecycle · spawns the backend · builds the installer*
2. Label `USER INTERFACE` — title **React + TypeScript (Vite)** — caption
   *Search, library, indexing progress · self-hosted fonts, no CDN*
3. A thin **indigo** connector strip with a downward arrow, mono text centred:
   `HTTP over 127.0.0.1 only — never exposed to the network`
4. Label `BACKEND SIDECAR` — title **FastAPI, packaged with PyInstaller** —
   caption *Folder scan · upload · search · job progress · health*
5. Label `BACKGROUND WORK` — title **In-process queue and worker thread** —
   caption *No Celery, no Redis, no message broker*
6. A row of **four equal boxes** side by side underneath:
   - **SQLite** — *Files, passages, status, plus a full-text keyword index*
   - **Qdrant, embedded** — *Two vector collections, in-process — no server, no Docker*
   - **Filesystem** — *Copies of originals, page images, thumbnails*
   - **Model weights** — *About 9 GB on disk, loaded locally, never downloaded*

### Zone 4 — Ingestion pipeline

Heading: **How a file gets in**

Six boxes left to right, joined by thin arrows. Each carries a mono step number
`01`–`06`, a short bold title, and one caption line.

| # | Title | Caption |
| --- | --- | --- |
| 01 | Scan folder | Supported types only. Originals are **copied**, never moved. |
| 02 | Extract | Text and page images from PDF, DOCX and image files |
| 03 | Chunk and filter | Split into passages; discard those with no retrievable content |
| 04 | Embed and index | bge-large-en-v1.5 → 1024-dim vectors → Qdrant + FTS5 |
| 05 | Read and describe | PaddleOCR on GPU · moondream2 captions · CLIP image vectors |
| 06 | Ready | Searchable by text, by picture, and by exact number |

Colour the boxes by the path they feed: box 04 indigo, box 05 violet, and the
FTS5 mention inside 04 green. Boxes 01–03 stay neutral.

Underneath, a small mono lifecycle strip with the last two states in green:

`pending → extracting → extracted → processing → ready`

And one caption line:

> Work is prioritised so a file becomes findable in about a second, while
> slower image processing continues in the background.

### Zone 5 — Search pipeline

Heading: **How a question gets answered**

At the top, one wide box holding a mono example query:

`"outward reference 42595078122"`

Below it, **three columns fanning out**, each a white box with a 3 pt coloured
top border:

| Column | Top border | Title | Caption | Footer (mono) |
| --- | --- | --- | --- | --- |
| A | Indigo | Meaning, in text | Matches passages by sense, not by keyword | `bge-large · cut-off 0.55` |
| B | Violet | Meaning, in pictures | Matches words directly against photographs | `OpenCLIP ViT-H/14 · cut-off 0.22` |
| C | Green | Exact words and numbers | Finds reference numbers and codes literally | `SQLite FTS5 · identifiers required` |

The three columns converge with thin lines into a single box:

- **Reciprocal Rank Fusion** — *Merged by position, not by score* — mono: `1 / (60 + rank)`

Then two boxes stacked below:

- **Group by file** — *Strongest passage becomes the headline; the rest attach as supporting evidence*
- **Relevance floor** — *Weak results are dropped; when the best match is only marginal, the app says so*

Final box, wider, green outline:

- **Ranked results** — *Title, snippet, thumbnail, page number, match strength — ten per page*

### Zone 6 — Technology stack

Heading: **Built with**

A compact two-column table, mono in the right column.

| Layer | Choice |
| --- | --- |
| Desktop shell | Tauri (Rust) |
| Interface | React · TypeScript · Vite |
| Backend | Python · FastAPI · PyInstaller |
| Database | SQLite · SQLAlchemy · Alembic |
| Keyword index | SQLite FTS5 |
| Vector store | Qdrant (embedded) |
| Text embeddings | BAAI/bge-large-en-v1.5 |
| Image embeddings | OpenCLIP ViT-H/14 |
| Captioning | moondream2 |
| OCR | PaddleOCR (GPU) · Tesseract (CPU) |

### Zone 7 — Design decisions

Heading: **Three decisions, each measured**

Three stacked panels. Each has a small coloured mono label, a bold one-line
claim, and a short supporting line. Figures are set in mono; the two problem
figures are burnt orange.

**Panel 1** — label indigo `SEPARATE SPACES`

> **Text and image vectors are stored in two collections, never one.**
> Both are 1024 numbers wide, but they are different spaces. Searching across a
> mixture returns noise — and nothing about the results looks wrong.

**Panel 2** — label violet `RANK, NOT SCORE`

> **Results are merged by position, because the two models' scores are not comparable.**
> A good text match scores `0.70`. An equally good image match scores `0.25`.
> Comparing them directly puts every text result above every image result.

**Panel 3** — label green `KEYWORDS STILL MATTER`

> **Meaning-based search cannot find a reference number.**
> The passage literally containing the number scored `0.45`. An unrelated
> document scored `0.57`. The correct answer ranked *below* the noise — an
> inversion no threshold can repair. A keyword index solves it.

### Zone 8 — Project phases

One horizontal strip of six equal segments with mono labels and hairline
borders. Segments 1–4 green border with a green `COMPLETE` marker; segment 5
indigo, `IN PROGRESS`; segment 6 neutral, `PLANNED`.

| Label | Title | State |
| --- | --- | --- |
| `PHASE 1` | Backend and round trip | COMPLETE |
| `PHASE 2` | Storage and ingestion | COMPLETE |
| `PHASE 3` | Models, OCR, queue | COMPLETE |
| `PHASE 4` | Vector search and ranking | COMPLETE |
| `PHASE 5` | Desktop interface | IN PROGRESS |
| `PHASE 6` | Windows installer | PLANNED |

Footer line, small, muted, centred:

> All figures measured against a real document library. No network connection is
> used at any point after installation.

---

## 5. Hard requirements

- Reproduce every quoted string **exactly**. No invented words, no filler text,
  no lorem ipsum, no misspellings.
- All content fits on **one A3 page**. Nothing is cropped; there is no second
  page or continuation.
- Text stays inside its panel. Nothing overlaps anything else.
- Body text no smaller than 11 pt at A3 scale — readable at arm's length.
- Each accent colour keeps one meaning across the whole poster: indigo means
  text, violet means image, green means exact keyword.
- Arrows are thin and straight, horizontal or vertical only. No curves.

---

## 6. Fallback single-paragraph prompt

For a model that accepts only one prompt block:

> A single A3 portrait academic project poster, 297×420mm, flat vector
> engineering-documentation style on a soft green-grey #F2F3EF background with
> white panels and thin hairline borders. Title in serif: "Local Multimodal File
> Search", subtitle "Find your files by what is inside them — not by their
> name". Below it a vertical architecture stack labelled Tauri desktop shell,
> React interface, an indigo strip reading "HTTP over 127.0.0.1 only", a FastAPI
> backend, and a row of four store boxes: SQLite, Qdrant embedded, Filesystem,
> Model weights. Then a horizontal six-step ingestion flow: Scan folder,
> Extract, Chunk and filter, Embed and index, Read and describe, Ready. Then a
> search flow where one query fans out into three columns — indigo "Meaning, in
> text", violet "Meaning, in pictures", green "Exact words and numbers" — which
> converge into "Reciprocal Rank Fusion", then "Group by file", then "Ranked
> results". A technology stack table and three design-decision panels at the
> bottom, and a six-segment project phase strip along the footer. IBM Plex Serif
> headings, IBM Plex Sans body, IBM Plex Mono for all numbers. Accent colours
> only indigo #3F53A0, violet #7A5399, green #2C7458. No gradients, no shadows,
> no 3D, no neon, no circuit-board imagery, no emoji, no photographs. Clean,
> legible, print-ready.
