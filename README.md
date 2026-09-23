# psaurus-pcr-analyzer

Access the tool here: https://psaurus-pcr-analyzer.streamlit.app/

Analysis pipeline for **Plasmidsaurus long-read (Oxford Nanopore) PCR amplicon
data**: quality-filter the reads, pull out the insert sitting between two
user-supplied flanking sequences, collapse and count the unique inserts, and
(optionally) compare them against an intended library.

Point it at one FASTQ, at a list of them, at a whole delivery folder, or at a
`.zip`/`.tar.gz` of one. Every FASTQ found is analysed **separately** with
identical settings, and the batch additionally gets cross-sample comparison
tables.

It ships as three layers that all run the same code:

| Layer | Entry point | Use it for |
|---|---|---|
| Python API | `psaurus_pcr.run_analysis(...)` / `run_batch(...)` | notebooks, pipelines, other tools |
| CLI | `psaurus-pcr` / `python -m psaurus_pcr` | cluster runs, pipelines |
| Web app | `streamlit run streamlit_app.py` | interactive exploration, sharing with the bench |

The core never reads `sys.argv`, never requires a file path (bytes and upload
buffers work), and never writes intermediate files that a downstream step needs
to read back. Writing outputs is a separate, optional step.

---

## Contents

- [Install](#install)
- [Quickstart](#quickstart)
- [What the pipeline does](#what-the-pipeline-does)
- [Analysing many samples at once](#analysing-many-samples-at-once)
- [CLI reference](#cli-reference)
- [Output files](#output-files)
- [PDF run report](#pdf-run-report)
- [Streamlit app](#streamlit-app)
- [Deploying to Streamlit Community Cloud](#deploying-to-streamlit-community-cloud)
- [Python API](#python-api)
- [Design decisions](#design-decisions-and-the-biology-behind-them)
- [Project layout](#project-layout)
- [Tests](#tests)

---

## Install

### conda / mamba (recommended on a cluster)

```bash
micromamba create -n psaurus -c conda-forge -c bioconda \
    python=3.11 biopython pandas numpy matplotlib streamlit pytest python-edlib
micromamba activate psaurus
pip install -e .          # from the repo root; adds the `psaurus-pcr` command
```

### pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

You can also skip installing the package and just run from a checkout — both
the CLI (`PYTHONPATH=src python -m psaurus_pcr ...`) and `streamlit_app.py`
add `src/` to the path themselves.

Dependencies: [Biopython](https://biopython.org) (FASTQ parsing and Phred
decoding), [edlib](https://github.com/Martinsos/edlib) (fast Levenshtein
alignment for fuzzy flank matching), pandas, matplotlib, and streamlit for the
web app.

---

## Quickstart

A small synthetic dataset is included (538 reads, an 8-member barcode library,
~1.2% simulated error, both strands, plus concatemers / primer dimers /
read-through artefacts):

```bash
psaurus-pcr examples/example_reads.fastq \
    -F CAGTTCGGACTTAGCCATGACT \
    -R TGGACCAATCGTTACGGTCAAG \
    -l examples/example_library.txt \
    -o results/
```

```
Step 1 - quality filtering
  total reads       : 538
  passing Q>=20 (mean_phred)    : 498 (92.57%)

Step 2 - flank extraction
  extracted                           : 482 (96.79%)
  reverse_flank_not_found             : 6 (1.20%)
  insert_empty                        : 5 (1.00%)
  insert_too_long                     : 5 (1.00%)
  reads with an ambiguous flank hit   : 8

Step 3 - quantification
  extracted reads   : 482
  unique sequences  : 117
  singletons        : 98 (83.76% of unique)

Step 4 - library comparison (exact matching)
  library members   : 8
  detected          : 7 (87.50%)
  unexpected unique : 110 (94.02%)
  unexpected reads  : 139 (28.84%)
```

That 94% "unexpected unique" figure is the expected behaviour of **exact**
matching on nanopore data, not a bug: each residual basecall error creates its
own unique sequence. Re-run with `--fuzzy-library` and those error variants
collapse onto their intended library member, leaving only the true
contaminant:

```
Step 4 - library comparison (fuzzy matching)
  detected          : 7 (87.50%)
  unexpected unique : 8 (6.84%)
  unexpected reads  : 25 (5.19%)
```

### …and a four-sample batch

The same generator writes a mock delivery folder, `examples/batch/`
(`barcode01`–`barcode04`, deliberately differing: one clean, one shallow, one
heavily contaminated, one with two library dropouts), plus
`examples/example_batch.zip` containing the same four files.

```bash
psaurus-pcr examples/batch \
    -F CAGTTCGGACTTAGCCATGACT -R TGGACCAATCGTTACGGTCAAG \
    -l examples/example_library.txt -o batch_results/

# identical result from the zip
psaurus-pcr examples/example_batch.zip -F ... -R ... -l ... -o batch_results/
```

```
  sample                       reads  passQC%  extract%  unique   top%   lib%  unexp%
  -----------------------------------------------------------------------------------
  barcode01                      340     94.1      97.8      91   21.7   87.5    28.8
  barcode02                      138     92.8      98.4      34   26.2   87.5    23.0
  barcode03                      259     94.2      98.4      67   29.2   87.5    55.8
  barcode04                      240     95.0      98.7      64   30.7   62.5    28.4
```

The contamination in `barcode03` (55.8% of reads unexpected) and the dropouts in
`barcode04` (62.5% library coverage against 87.5% elsewhere) are exactly what
the overview table is for.

Regenerate both datasets with `python examples/make_example_data.py`
(deterministic, fixed seed).

---

## What the pipeline does

### Step 1 — quality filtering

Each read's quality is summarised and compared against a threshold
(`--min-quality`, default **Q20**).

Two statistics are available via `--quality-metric`:

- `mean_phred` *(default)* — the arithmetic mean of the per-base Phred scores.
  This is the conventional "mean read Q" and what most filtering tools report.
- `error_prob` — `-10·log10(mean per-base error probability)`. This is how ONT
  itself defines a read qscore, and it is the more defensible statistic, since
  Phred is a log scale and averaging logs over-weights the good bases. It reads
  a few Q units lower than `mean_phred` on typical reads, so state which one
  you used when quoting a threshold.

**On the default of Q20.** Older ONT chemistry (R9.4.1, 1D) could not do much
better than Q9–Q12, so a great deal of legacy documentation quotes Q9 or Q10 as
"the" nanopore cutoff. Modern R10.4.1 flowcells basecalled with the
super-accurate (`sup`) model — which is what Plasmidsaurus runs — routinely
produce modal read qualities above Q20, so Q20 (99% accuracy) is a moderate
"keep the good reads" threshold rather than an aggressive one. Lower it for an
older or rescued run; the QC plot shows immediately whether your cutoff is
trimming a separate low-quality tail (good) or slicing through the main
population (too aggressive).

Reported before *and* after filtering: total reads, reads passing, % passing,
and the read-length and quality distributions.

### Step 2 — flank-based extraction

For every read that passes QC, F and R are located by **approximate matching**
(`edlib`, infix/"HW" mode, true Levenshtein distance so substitutions *and*
indels count), and the insert strictly between the end of F and the start of R
is extracted.

- **Fuzzy, not exact.** Even Q20 reads carry ~1% residual error, concentrated
  in homopolymers. Requiring an exact 20–25 bp flank match would silently drop
  a large and *sequence-biased* fraction of perfectly good reads.
- **Edit budget.** `--flank-edit-fraction` (default `0.12`) sets the allowed
  edits as a fraction of each flank's length: 2 edits for a 20 bp flank, 3 for
  25 bp — the usual 10–15% rule of thumb. Override per flank with
  `--forward-max-edits` / `--reverse-max-edits`.
  **The tradeoff:** too loose and a short flank will match random sequence
  somewhere in a multi-kb read, filling the count table with garbage inserts;
  too tight and you lose real reads in proportion to how error-prone the run
  was. Tune it empirically — sweep the value and watch where the extraction
  rate plateaus.
- **Both strands are searched.** A double-stranded PCR product enters the pore
  from either end, so a run contains both orientations in roughly equal
  proportion. Each read is searched as given *and* reverse-complemented; the
  orientation producing a valid ordered F…R pair wins (lower total edit
  distance breaks a tie). The extracted insert is always reported in the F→R
  orientation, which is what makes exact-match collapsing in step 3 meaningful.
- **Anchoring.** F is anchored first and R searched downstream of it; then the
  reverse (R anchored, F searched upstream). Whichever pairing has the lower
  combined edit distance is used, so one spurious best hit cannot hide a
  genuine pair.

Every read gets exactly one status, and all of them are counted and reported:

| Status | Meaning | Counted? |
|---|---|---|
| `extracted` | valid F…R pair, insert within the length bounds | yes |
| `no_flanks_found` | neither flank matched in either orientation | no |
| `forward_flank_not_found` | only R matched | no |
| `reverse_flank_not_found` | only F matched (e.g. a truncated product) | no |
| `flanks_wrong_order_or_overlapping` | both matched, never as F…R | no |
| `insert_empty` | F immediately abuts R (primer dimer) | no |
| `insert_too_short` | shorter than `--min-insert-length` | no |
| `insert_too_long` | longer than the max insert length | no |
| `ambiguous_flank_excluded` | only when `--exclude-ambiguous` is set | no |

**Ambiguous flank hits.** If a second, non-overlapping placement of a flank
scores within `--ambiguity-margin` edits (default 1) of the best one, the read
is flagged. That pattern usually means a concatemer, a tandem duplication or a
read-through product. By default such reads are **counted using their
best-scoring hit** but carry an `ambiguous_flank` flag in the per-read table
and are reported as their own line in the QC summary — nothing is silently
collapsed. Pass `--exclude-ambiguous` to drop them from quantification instead.

**Max insert length.** Inserts longer than the cutoff are flagged and excluded
from quantification but kept in the QC counts (and their sequence is retained
in the per-read table so you can inspect them). The cutoff is, in order of
precedence: `--max-insert-length`, else **3× the longest library sequence**
when `-l` is given, else **5000 bp**.

### Step 3 — quantification

Extracted inserts are collapsed by **exact string identity** and counted. The
output table is sorted by descending count (ties broken by sequence, so the
table is stable between runs) and reports each sequence's count, percentage of
extracted reads, and length. Summary statistics include the number of unique
sequences, the singleton count and fraction, and the insert-length
distribution.

Exact collapsing is the right default for a defined-library experiment: a
residual basecall error shows up as its own low-count sequence, which is
informative, rather than being quietly merged into a neighbour. It also makes
the singleton fraction a useful diagnostic.

### Step 4 — library comparison *(optional)*

Given `-l/--library` (one sequence per line; `#` comments and `>` FASTA headers
ignored), the pipeline reports:

- per library member: detected or not, read count, % of extracted reads, and
  the best observed supporting sequence;
- **library coverage** — how many / what % of library members were detected;
- **unexpected sequences** — how many / what % of the detected unique sequences
  are *not* in the library, and how many reads they account for. This is the
  number to watch for contamination or synthesis errors.

**Matching is exact by default.** Both sides are normalised first (all
whitespace stripped, upper-cased), so case and formatting differences between a
design spreadsheet and a basecall never cause a spurious miss — but that is
robustness to *formatting*, not to sequence. A library is a set of *designed*
sequences, so "close but not identical" is a finding, not a match.

`--fuzzy-library` turns on optional inexact matching: global (Needleman–Wunsch)
edit distance with a budget of `--library-edit-fraction` of each member's
length (default 5%, minimum 1 edit) or an absolute `--library-max-edits`. Each
observed sequence is assigned to its single best in-budget member; exact
matches always win over fuzzy ones. Global alignment means a truncated product
is *not* treated as a match.

Reverse-complement matching is deliberately not attempted: step 2 already
orients every insert F→R, so a library that only matches in reverse complement
means the library file is written on the opposite strand — worth surfacing
rather than papering over.

---

## Analysing many samples at once

Pass any number of inputs; they are expanded into a flat list of FASTQ files
and each one is analysed on its own with the same parameters.

| Input | Behaviour |
|---|---|
| `reads.fastq`, `reads.fastq.gz` | one sample (streamed from disk, never slurped) |
| several files | one sample each, in the order given |
| a **directory** | searched recursively for `*.fastq` / `*.fq` (± `.gz`); narrow it with `--pattern`, flatten it with `--no-recursive` |
| a **`.zip`** | FASTQ members read out in memory; `__MACOSX` and dotfiles skipped |
| a **`.tar` / `.tar.gz` / `.tgz`** | same |
| any mix of the above | combined, in the order given |

Nothing is ever extracted to disk — archive members are read into memory and
handed to the pipeline as bytes, the same route a Streamlit upload takes.

**Sample names** come from the filename with the FASTQ and compression
extensions stripped: `barcode07.fastq.gz` → `barcode07`. Two files that reduce
to the same name (say `sample.fastq` in two subfolders) are disambiguated as
`sample` and `sample__2` rather than one silently overwriting the other; the
`origin` column in `sample_overview.tsv` tells you which is which.

**Check before you commit** to a long run:

```bash
psaurus-pcr /path/to/order/ -F ... -R ... --list-inputs
```

```
4 FASTQ file(s) would be analysed:
  barcode01                      examples/batch/barcode01.fastq.gz
  barcode02                      examples/batch/barcode02.fastq.gz
  ...
```

**Output layout** is flat for a single sample (unchanged from before) and
nested for a batch:

```
results/
├── barcode01/            # the full single-sample output set, per sample
│   ├── unique_sequences.tsv
│   ├── qc_summary.tsv
│   ├── run_summary.json
│   └── *.png
├── barcode02/ …
├── sample_overview.tsv          one row per sample
├── sequence_count_matrix.tsv    unique sequence x sample counts
├── combined_unique_sequences.tsv
├── combined_qc_summary.tsv      metrics as rows, samples as columns
├── combined_library_comparison.tsv
├── library_detection_matrix.tsv
├── batch_summary.json / .txt
├── batch_read_fate.png
└── batch_library_detection.png
```

Force per-sample directories for a single input with `--sample-dirs`.

**A failing sample does not sink the batch.** If one file is truncated or in
the wrong format, it is recorded in `sample_overview.tsv` with
`status = failed` and its error message, the rest of the batch completes, and
the CLI exits `1` so a pipeline still notices. `--fail-fast` restores
abort-on-first-error.

**The two tables to look at first:**

- `sample_overview.tsv` — depth, QC pass rate, extraction rate, unique count,
  library coverage and unexpected-read fraction, one row per sample. Outliers
  in any of those columns are the samples worth investigating.
- `sequence_count_matrix.tsv` — every unique insert seen anywhere in the batch
  against every sample, with `total_count` and `n_samples_detected`. A
  high-abundance sequence with `n_samples_detected = 1` is usually a
  sample-specific artefact; a low-abundance one present in *every* sample is
  the signature of index hopping or cross-contamination.

---

## CLI reference

```
psaurus-pcr INPUT [INPUT ...] -F SEQ -R SEQ [-l FILE] [-o DIR] [options]
```

Run `psaurus-pcr --help` for the authoritative list. Summary:

| Option | Default | What it does |
|---|---|---|
| `INPUT ...` | — | one or more FASTQ files (plain or gzipped — detected by magic bytes, not extension), directories, and/or `.zip`/`.tar.gz` archives |
| `-F, --forward` | *required* | forward flank, 5'→3' |
| `-R, --reverse` | *required* | reverse flank, 5'→3' (see `--r-convention`) |
| `-l, --library` | none | intended-library text file, one sequence per line |
| `-o, --outdir` | `psaurus_pcr_results` | output directory |
| `--no-recursive` | off | do not descend into subdirectories |
| `--pattern` | none | glob applied when walking a directory, e.g. `barcode*.fastq.gz` |
| `--list-inputs` | off | list the FASTQ files that would be analysed, then exit |
| `--fail-fast` | off | abort the batch on the first sample that fails |
| `-q, --min-quality` | `20.0` | minimum mean read quality |
| `--quality-metric` | `mean_phred` | `mean_phred` or `error_prob` |
| `--flank-edit-fraction` | `0.12` | allowed edits per flank, as a fraction of its length |
| `--forward-max-edits` / `--reverse-max-edits` | derived | absolute edit budgets |
| `--ambiguity-margin` | `1` | edits within which a rival flank hit counts as ambiguous |
| `--exclude-ambiguous` | off | drop ambiguous reads from the counts |
| `--r-convention` | `auto` | `auto`, `literal`, or `revcomp` |
| `--orientation-probe-reads` | `2000` | reads used by `auto` (0 = all) |
| `--max-insert-length` | derived | "absurdly long" cutoff |
| `--min-insert-length` | `1` | only truly empty inserts are dropped by default |
| `--library-length-multiplier` | `3.0` | multiplier used to derive the cutoff from the library |
| `--fuzzy-library` | off | enable fuzzy library matching |
| `--library-edit-fraction` | `0.05` | fuzzy tolerance, fraction of length |
| `--library-max-edits` | none | absolute fuzzy tolerance |
| `--format` | `tsv` | `tsv` or `csv` for the tabular outputs |
| `--per-read` | off | also write the per-read classification table |
| `--no-plots` | off | skip the PNGs |
| `--no-pdf` | off | skip the PDF run report |
| `--top-n` | `20` | sequences shown in the abundance bar chart |
| `--sample-dirs` / `--no-sample-dirs` | auto | force per-sample subdirectories on / off (auto = off for one sample, on for many) |
| `--quiet` | off | suppress the stdout report |

### `--r-convention`: how your R sequence is interpreted

"R, given 5'→3'" is genuinely ambiguous in the wild. People write a *reverse
PCR primer* 5'→3' on the **bottom** strand, but they write the flanking
sequence of a construct 5'→3' on the **top** strand. The two differ by a
reverse complement, and getting it wrong means extracting nothing.

- `literal` — R lies on the same strand as F; the amplicon reads `5'-F…R-3'`.
  R is searched exactly as typed.
- `revcomp` — R is a conventional reverse primer, so `revcomp(R)` is what
  appears downstream of F, and that is what gets searched.
- `auto` *(default)* — both are tried on the first `--orientation-probe-reads`
  quality-passing reads, and whichever explains more reads is used. The
  decision, the counts behind it, and the sequence actually searched are all
  recorded in the run summary. If the extraction rate is low and the probe
  reports a tie, set the convention explicitly.

### More examples

```bash
# a whole delivery folder, or a zip of one
psaurus-pcr /path/to/plasmidsaurus_order/ -F $F -R $R -l library.txt -o out/
psaurus-pcr order.zip -F $F -R $R -o out/

# several named files, and a mixed set of inputs
psaurus-pcr barcode01.fastq.gz barcode02.fastq.gz -F $F -R $R -o out/
psaurus-pcr run1.zip run2/ extra.fastq -F $F -R $R -o out/ --pattern "barcode*.fastq.gz"

# older / rescued run: relax quality and allow more flank error
psaurus-pcr reads.fastq.gz -F $F -R $R -o out/ \
    --min-quality 12 --flank-edit-fraction 0.20

# strict: exact flanks only, stricter quality statistic, drop ambiguous reads
psaurus-pcr reads.fastq.gz -F $F -R $R -o out/ \
    --forward-max-edits 0 --reverse-max-edits 0 \
    --quality-metric error_prob --exclude-ambiguous

# debugging a low extraction rate: per-read table + explicit R orientation
psaurus-pcr reads.fastq.gz -F $F -R $R -o out/ --per-read --r-convention revcomp

# CSV for Excel, no plots
psaurus-pcr reads.fastq.gz -F $F -R $R -o out/ --format csv --no-plots
```

---

## Output files

Written to `--outdir`. Tabular files are TSV by default (`--format csv` for
comma-separated). With one sample these land directly in `--outdir`; with
several, each sample gets `--outdir/<sample>/` and the batch tables are written
alongside.

### Per sample

| File | Contents |
|---|---|
| `unique_sequences.tsv` | one row per unique extracted insert: `sequence`, `count`, `percent_of_extracted`, `length`. With `-l`, also `library_match_type` (`exact`/`fuzzy`/`none`), `library_id`, `library_sequence`, `library_edit_distance`, `in_library`. Sorted by descending count. |
| `qc_summary.tsv` | every run statistic flattened to `metric`/`value` rows (`quality.total_reads`, `extraction.status_counts.extracted`, `library.percent_library_detected`, …). |
| `library_comparison.tsv` | *(with `-l`)* one row per library member: `library_id`, `library_sequence`, `library_length`, `detected`, `count`, `percent_of_extracted`, `match_type`, `n_observed_variants`, `best_observed_sequence`, `best_edit_distance`. |
| `per_read_classification.tsv` | *(with `--per-read`)* one row per QC-passing read: `read_id`, `status`, `orientation`, `read_length`, `mean_quality`, `insert_length`, `forward_edit_distance`, `reverse_edit_distance`, `ambiguous_flank`, `counted`, `insert`. The place to look when the extraction rate disappoints. |
| `run_summary.json` | machine-readable record of the run: every parameter (including the derived edit budgets and max insert length), the resolved R convention and the probe counts behind it, read counts at every stage, all quantification and library statistics, the timestamp and the elapsed time. |
| `run_summary.txt` | the same report, human-readable — what the CLI prints. |
| `quality_distribution.png` | per-read quality, stacked kept vs removed, with the threshold marked. |
| `read_length_distribution.png` | raw read length, stacked kept vs removed. |
| `insert_length_distribution.png` | length of the extracted inserts. |
| `top_sequences.png` | the `--top-n` most abundant inserts; coloured by library membership when `-l` is given. |
| `report.pdf` | self-contained PDF run report — see [below](#pdf-run-report). Only for a single-sample run; a batch gets one combined report instead. |

### Per batch (two or more samples)

| File | Contents |
|---|---|
| `sample_overview.tsv` | one row per sample: `status`, `origin`, depth, QC pass rate, extraction rate, unique count, singletons, median insert length, ambiguous-flank count, resolved R convention, runtime, a `reads_<status>` column for every read fate, and — with `-l` — library coverage and unexpected-read fraction. Failed samples appear here with `status = failed` and an `error` message. |
| `sequence_count_matrix.tsv` | unique sequence × sample counts, plus `length`, `total_count`, `n_samples_detected`, and library annotation when `-l` is given. Sorted by total abundance. |
| `combined_unique_sequences.tsv` | every sample's count table stacked long-format, with a leading `sample` column. |
| `combined_qc_summary.tsv` | QC metrics as rows, samples as columns. |
| `combined_library_comparison.tsv` | *(with `-l`)* every sample's library table stacked long-format. |
| `library_detection_matrix.tsv` | *(with `-l`)* library member × sample counts, plus `total_count` and `n_samples_detected`. |
| `batch_summary.json` | batch-level record: parameters, the discovered sample list with origins, per-sample run summaries, totals, failures, timestamp. |
| `batch_summary.txt` | the per-sample overview table as printed by the CLI. |
| `batch_read_fate.png` | 100% stacked read fate per sample (depth on the tick labels). |
| `batch_library_detection.png` | *(with `-l`)* % of the library detected per sample. |
| `batch_report.pdf` | self-contained PDF report covering the batch and every sample — see [below](#pdf-run-report). |

---

## PDF run report

Every run writes a PDF alongside the tables, for record keeping — the point is
that the PDF **alone** is enough to reconstruct what was run and what came out,
without the TSVs. Skip it with `--no-pdf`.

A single-sample `report.pdf` contains:

1. **Cover** — inputs (FASTQ, library, both flanks as given *and* as searched),
   every parameter including the derived edit budgets and max insert length,
   the headline results, tool version and timestamps.
2. **QC page** — read accounting, the full read-fate table, read length and
   quality distributions before and after filtering, extracted insert lengths,
   and the R-orientation decision with the probe counts behind it.
3. **Plots** — the same four figures written as PNGs.
4. **Tables** — the top-N sequences and, if a library was given, the library
   comparison summary and every library member.

A batch `batch_report.pdf` puts the cross-sample view first — the sample list
with origins, any failed samples, the per-sample summary, read-fate counts, the
cross-sample plots, and the sequence × sample and library × sample matrices —
then a full section per sample.

Practical notes:

- **Nothing is silently dropped.** A table too wide for the page is split
  across pages by column group with the key column repeated, and a table
  truncated by row count says so and points at the TSV.
- Per-sample detail is capped at the first 24 samples (`max_detailed_samples`);
  the overview and every combined table still cover all of them, and the cap is
  stated in the document.
- Tables with 12 or fewer samples are transposed (metrics down the side,
  samples across), which is far more compact at that size.
- From Python: `psaurus_pcr.report.write_report(result_or_batch, "report.pdf")`,
  or `report_bytes(...)` to get it in memory.

---

## Streamlit app

```bash
streamlit run streamlit_app.py
# on a cluster node, to reach it from your laptop:
streamlit run streamlit_app.py --server.port 8899 --server.headless true
# then: ssh -N -L 8899:<node>:8899 you@cluster   and open http://localhost:8899
```

The app has:

- **a multi-file uploader** for FASTQs (plain or gzipped) and/or `.zip` /
  `.tar.gz` archives — nothing is written to disk, the bytes go straight into
  the same `run_batch` the CLI calls;
- **a server-path box** for a file, folder or archive already on the machine
  running the app. Browsers cannot upload a directory, and on a cluster the
  data is usually already on the filesystem, so this is the route to use for a
  whole delivery folder. Comma-separate several paths; folder search options
  (recursive, filename glob) sit just below it.
  **By default these paths are confined to the app's own directory**, so a
  deployed app cannot be used to read the rest of the server — see
  [deployment](#deploying-to-streamlit-community-cloud) for how to open that up
  on a machine you control;
- **a live preview** of exactly which FASTQ files would be analysed, shown
  before you press Run;
- **a file uploader** for the optional library text file, shared across all
  samples;
- **text inputs** for the F and R flanks;
- **a sidebar** with every parameter: quality threshold and statistic, flank
  edit tolerance (fractional or absolute), R orientation, ambiguity margin and
  whether to exclude ambiguous reads, max/min insert length, the fuzzy library
  toggle and its tolerance, output format, and top-N;
- **a Batch overview tab** (shown as soon as there are two or more samples):
  the per-sample summary table, the read-fate and library-coverage charts, the
  filterable sequence × sample count matrix, the library-member × sample
  matrix, and downloads for every combined table;
- **a sample selector** driving the per-sample tabs — QC summary (read fate
  plus all statistics), the unique-sequence table (sortable by clicking a
  header, with a substring filter and a minimum-count filter), the library
  comparison (with an "undetected members only" toggle), the plots, and the
  full run summary;
- **download buttons** for every output, per sample and for the batch,
  including the **PDF report** (rendered on request rather than on every
  interaction — press *Prepare PDF report*, then download).

Results are cached on the input bytes/paths plus parameters, so sorting a table
or switching samples does not re-run the analysis; file size and mtime are part
of the cache key, so editing data on disk does invalidate it. Samples that fail
are reported at the top of the page and the rest of the batch still renders.

---

## Deploying to Streamlit Community Cloud

**Entry point: `streamlit_app.py`, at the repository root.** That is the path
to give Community Cloud when it asks for the "Main file path".

### Deploying

1. Push this repository to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io), *Create app* → pick the
   repo and branch, set **Main file path** to `streamlit_app.py`.
3. Open **Advanced settings** and set the **Python version to 3.11, 3.12 or
   3.13**. This matters: `pandas` 3.x and `numpy` 2.4 need ≥ 3.11, and `edlib`
   publishes Linux wheels only up to `cp313` — on 3.14+ pip falls back to
   compiling it from source, which is slow and can fail on the builder.
4. Deploy. Nothing else is needed: every pinned dependency ships a manylinux
   wheel, so there is no `packages.txt` and no compiler involved.

`requirements.txt` holds the **runtime** dependencies, pinned exactly to the
versions this project is tested against. Test-only packages live in
`requirements-dev.txt` (`pip install -r requirements-dev.txt`) so they are
never installed on the server.

### Configuration and secrets

There are no credentials in this app — it talks to nothing but its own
filesystem. The settings below are read from `st.secrets` first (Community
Cloud's **Secrets** box, or a local `.streamlit/secrets.toml`, which is
git-ignored) and then from the environment:

| Secret | Env var | Default | What it does |
|---|---|---|---|
| `allow_any_path` | `PSAURUS_ALLOW_ANY_PATH` | `false` | Lets the server-path box reach the whole filesystem. **Leave off for any deployment strangers can reach.** |
| `data_root` | `PSAURUS_DATA_ROOT` | the app directory | The directory server-side paths are confined to when `allow_any_path` is off. |
| `max_total_upload_mb` | `PSAURUS_MAX_UPLOAD_MB` | `200` | Refuses to start on more uploaded data than the host can hold. |

**Why the server-path box is confined by default.** It exists so you can point
the app at a delivery folder on a lab machine, but on a public deployment an
unconstrained "type a path" box is a file-read primitive. Paths are therefore
resolved *inside* the app directory and may not escape it — `..` segments and
symlinks that point outward are rejected, not followed. A visitor to the
deployed app can still type `examples/batch` and run the bundled demo data.

On your own machine or a lab server, open it up with:

```bash
PSAURUS_ALLOW_ANY_PATH=1 streamlit run streamlit_app.py
# or confine it to your data instead of the whole disk:
PSAURUS_DATA_ROOT=/fh/fast/mylab/nanopore streamlit run streamlit_app.py
```

### The ephemeral filesystem

Community Cloud's disk resets on every restart and is not shared between
users or sessions. This app never depends on it:

- **Every download is built in memory at the moment you press the button** —
  tables via `DataFrame.to_csv(...).encode()`, the PDF via
  `report.report_bytes(...)`. Nothing is written to disk and read back, and
  `write_outputs()` (which does write files) is used only by the CLI.
- **Uploads are processed from the in-memory upload object.** The bytes go
  straight into `run_batch`; no path is assumed to exist afterwards.
- **Caches are disposable.** `st.cache_data` entries carry a TTL and an entry
  cap, and losing one — restart, eviction, expiry — only costs a recompute on
  the next *Run analysis*. Nothing reads a cached *file*.

### Memory, and when to use the CLI instead

The real ceiling is RAM, not disk. A run holds the read-length and quality
distributions plus a per-read table (one row per QC-passing read, including
the extracted insert) in memory, and the cached result keeps them for the
session. Community Cloud gives each app a modest shared memory budget, so:

- the app refuses to start on more than `max_total_upload_mb` of input;
- the batch cache keeps **one** result (`max_entries=1`) and the PDF cache two;
- for a deep run — millions of reads, or a whole plate of them — **run the
  `psaurus-pcr` CLI locally.** It streams the FASTQ rather than holding it,
  and it writes the same outputs plus the PDF report.

If the deployed app shows *"This app has gone over its resource limits"*, that
is the symptom: analyse fewer or smaller samples per run, or self-host.

---

## Python API

```python
from psaurus_pcr import AnalysisParams, run_analysis

result = run_analysis(
    fastq="reads.fastq.gz",                 # path, bytes, or any file-like object
    forward_flank="CAGTTCGGACTTAGCCATGACT",
    reverse_flank="TGGACCAATCGTTACGGTCAAG",
    library="library.txt",                  # path, bytes, file-like, or a list[str]
    params=AnalysisParams(
        min_mean_quality=20.0,
        flank_edit_fraction=0.12,
        r_convention="auto",
        fuzzy_library=False,
    ),
)

result.unique_sequences      # pandas DataFrame
result.library_comparison    # pandas DataFrame or None
result.qc_summary            # pandas DataFrame (metric/value)
result.per_read              # pandas DataFrame
result.run_summary           # plain dict, JSON-serialisable
print(result.text_summary())

# writing files is opt-in and entirely separate from the analysis
result.write_outputs("results/", output_format="tsv", write_per_read=True)
```

For several samples, `run_batch` takes the same arguments plus anything
`discover_inputs` understands:

```python
from psaurus_pcr import AnalysisParams, run_batch

batch = run_batch(
    sources=["run1.zip", "/data/plasmidsaurus_order/", "extra.fastq.gz"],
    forward_flank="CAGTTCGGACTTAGCCATGACT",
    reverse_flank="TGGACCAATCGTTACGGTCAAG",
    library="library.txt",
    params=AnalysisParams(fuzzy_library=True),
    pattern="barcode*.fastq.gz",   # applied when walking directories
)

batch.sample_names                # ['barcode01', 'barcode02', ...]
batch["barcode01"]                # the full AnalysisResult for one sample
batch.sample_overview             # one row per sample
batch.sequence_count_matrix       # unique sequence x sample
batch.library_detection_matrix    # library member x sample
batch.failures                    # samples that could not be analysed
print(batch.text_summary())

batch.write_outputs("results/", output_format="tsv")
```

A PDF report for either a single result or a whole batch:

```python
from psaurus_pcr.report import report_bytes, write_report

write_report(result, "report.pdf")                 # one sample
write_report(batch, "batch_report.pdf")            # batch overview + every sample
write_report(batch, "overview.pdf", per_sample_detail=False)
pdf = report_bytes(batch, top_n=30)                # in memory, for a web download
```

To see what would be analysed without running anything:

```python
from psaurus_pcr import discover_inputs

for sample in discover_inputs("/data/order/", recursive=True):
    print(sample.name, sample.describe())
```

The individual steps are importable and pure, so you can use one without the
others:

```python
from psaurus_pcr.inputs import discover_inputs
from psaurus_pcr.fastq_io import read_fastq, read_library
from psaurus_pcr.qc import filter_reads
from psaurus_pcr.flanks import extract_insert, find_flank
from psaurus_pcr.quantify import quantify
from psaurus_pcr.library import compare_to_library
from psaurus_pcr.plots import build_all_figures, build_batch_figures
from psaurus_pcr.batch import build_sequence_count_matrix
from psaurus_pcr.report import write_report
```

---

## Design decisions and the biology behind them

- **Both strands are searched** because a duplex PCR product enters the pore
  from either end; ignoring one orientation throws away ~half the data. Inserts
  are normalised to the F→R orientation so both halves collapse onto the same
  count.
- **Fuzzy flank matching with true Levenshtein distance**, because the
  characteristic nanopore error is an indel, not a substitution — a Hamming
  distance would systematically under-match.
- **Quality filtering before extraction**, because a poor read is more likely
  to produce a *wrong* insert than none at all, and a wrong insert is
  indistinguishable from a real rare variant once it is in the count table.
- **Exact collapsing in step 3, exact library matching in step 4**, with fuzzy
  as an explicit opt-in: the error variants are data, and the gap between the
  exact and fuzzy numbers is itself a measure of run quality.
- **Nothing is silently dropped.** Every read that does not make it into the
  count table is assigned a specific status, counted, and reported — including
  ambiguous hits, empty inserts and oversized inserts.
- **Analysis and output are separate.** `run_analysis` returns in-memory
  structures; only `write_outputs` touches the filesystem. That is what lets the
  CLI and the web app share the whole pipeline.
- **Samples in a batch are independent.** No counts are pooled, no thresholds
  are derived across samples, and the R convention is resolved per sample — a
  batch is N runs plus a comparison layer, never a merged analysis. Pooling
  would hide exactly the per-sample differences you are looking for.
- **Degenerate flanks are understood.** IUPAC ambiguity codes in a flank
  (`N`, `R`, `Y`, …) are given to the aligner as equivalences, so a degenerate
  primer matches rather than burning an edit at every ambiguous position.
- **Library entries that are not DNA are rejected, not counted.** A stray
  header row pasted from a spreadsheet would otherwise become a library member
  that can never be detected, quietly deflating "% of library detected"; the
  rejected entries are reported instead.
- **One bad file does not lose the run.** A sample that fails is recorded with
  its error and the batch continues; on a 96-barcode plate, aborting on the
  first truncated file would be the wrong trade.

---

## Project layout

```
psaurus-pcr-analyzer/
├── src/psaurus_pcr/
│   ├── config.py        AnalysisParams — every tunable, with the defaults documented
│   ├── inputs.py        discovery of samples from files, folders and archives
│   ├── fastq_io.py      FASTQ / library input from paths, bytes or file-like objects
│   ├── sequtils.py      reverse complement, normalisation, Phred maths
│   ├── qc.py            step 1: quality filtering and read statistics
│   ├── flanks.py        step 2: fuzzy flank matching, strand handling, extraction
│   ├── quantify.py      step 3: collapsing and counting
│   ├── library.py       step 4: intended-library comparison
│   ├── pipeline.py      one sample: orchestration, R-convention auto-detection, results
│   ├── batch.py         many samples: per-sample runs + cross-sample tables
│   ├── plots.py         matplotlib figures (returned as Figure objects)
│   ├── report.py        the PDF run report
│   └── cli.py           argparse layer — parsing, writing and printing only
├── streamlit_app.py     web interface (no analysis logic) - Community Cloud entry point
├── .streamlit/config.toml   committed Streamlit settings (no secrets)
├── requirements.txt     pinned runtime dependencies (what the cloud installs)
├── requirements-dev.txt pinned test/lint dependencies
├── examples/            synthetic dataset + the deterministic generator
├── tests/               pytest suite
├── pyproject.toml
└── requirements.txt
```

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
pyflakes src/psaurus_pcr/*.py streamlit_app.py tests/*.py examples/*.py
```

The suite uses synthetic FASTQ data built in memory — deterministic, no
external files. It covers a clean match, a match that needs fuzzy tolerance
(substitutions and an indel), a reverse-complement read, reads missing one or
both flanks, flanks in the wrong order and overlapping flanks, empty /
oversized / undersized inserts, a concatemer with ambiguous flank hits, a
low-quality read that must be filtered out, quantification and library
comparison (exact and fuzzy), R-convention auto-detection for both conventions,
path-vs-bytes input equivalence, and output file writing.

Batch mode is covered separately: sample discovery from files, nested
directories, globs, zip and tar.gz archives, raw bytes and upload-shaped
buffers; duplicate sample-name disambiguation; per-sample independence; the
combined overview, sequence × sample and library × sample tables; a failing
sample not sinking the batch; and the flat-vs-nested output layouts.

The CLI and the Streamlit app are tested end to end, including a real
two-sample batch driven through the app's own widgets, and the PDF report is
checked for structure and content (valid PDF, expected sections, wide tables
split rather than truncated, per-sample detail capped).

`tests/test_regressions.py` holds one test per defect found in the audit of
this pipeline, each naming the bug it pins down.

`tests/test_deployment.py` encodes the hosted environment's assumptions: that
every third-party import is declared in `requirements.txt` and pinned, that no
secrets file is committed, that the app never round-trips an output through
disk, that its caches are bounded, and that server-side paths stay confined.
