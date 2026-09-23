"""Command-line interface.

This module is *only* argument parsing, file writing and printing.  All the
science lives in the importable core (:mod:`psaurus_pcr.pipeline` and friends),
so the Streamlit app can do exactly the same work without going near argparse.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from psaurus_pcr import __version__
from psaurus_pcr.config import (
    DEFAULT_AMBIGUITY_MARGIN,
    DEFAULT_FLANK_EDIT_FRACTION,
    DEFAULT_LIBRARY_LENGTH_MULTIPLIER,
    DEFAULT_MAX_INSERT_LENGTH,
    DEFAULT_MIN_MEAN_QUALITY,
    DEFAULT_ORIENTATION_PROBE_READS,
    AnalysisParams,
)
from psaurus_pcr.batch import run_batch
from psaurus_pcr.inputs import discover_inputs

EPILOG = """\
examples
--------
  # one FASTQ
  psaurus-pcr reads.fastq.gz -F GCTAGCATGCTAGCAT -R TTGACCTGCAGTTAAC -o results/

  # several FASTQs at once -- each analysed separately with the same settings
  psaurus-pcr barcode01.fastq.gz barcode02.fastq.gz -F ... -R ... -o results/

  # a whole delivery folder (searched recursively), or a zip/tar of one
  psaurus-pcr /path/to/plasmidsaurus_order/ -F ... -R ... -o results/
  psaurus-pcr order.zip -F ... -R ... -o results/

  # mix them, and narrow a folder search with a glob
  psaurus-pcr run1.zip run2/ extra.fastq -F ... -R ... -o results/ --pattern "barcode*.fastq.gz"

  # see what would be analysed, without running anything
  psaurus-pcr /path/to/order/ -F ... -R ... --list-inputs

  # relax the quality cutoff for an older run, and allow 3 edits per flank
  psaurus-pcr reads.fastq.gz -F ... -R ... -o results/ \\
      --min-quality 12 --forward-max-edits 3 --reverse-max-edits 3

  # compare against an intended library, with fuzzy matching on
  psaurus-pcr reads.fastq.gz -F ... -R ... -l library.txt -o results/ --fuzzy-library

outputs written to --outdir
---------------------------
  one sample  -> the files below are written straight into --outdir
  many samples-> each sample gets its own <outdir>/<sample>/ directory, plus
                 the combined batch tables listed at the end

  per sample:
    unique_sequences.tsv          unique inserts, counts, % of extracted reads
                                  (+ library match columns when -l is given)
    qc_summary.tsv                every run statistic as metric/value rows
    library_comparison.tsv        one row per library member (only with -l)
    per_read_classification.tsv   one row per QC-passing read (only with --per-read)
    run_summary.json / .txt       parameters, counts at each stage, timestamp
    *.png                         quality / read-length / insert-length / top-N plots
    report.pdf                    self-contained PDF run report (single sample)

  per batch (two or more samples):
    sample_overview.tsv           one row per sample: depth, pass rate, extraction
                                  rate, unique count, library coverage
    sequence_count_matrix.tsv     unique sequence x sample counts, the main
                                  cross-sample comparison table
    combined_unique_sequences.tsv every sample's count table, stacked long-format
    combined_qc_summary.tsv       QC metrics as rows, samples as columns
    combined_library_comparison.tsv / library_detection_matrix.tsv  (only with -l)
    batch_summary.json / .txt     batch-level report
    batch_report.pdf              self-contained PDF report: batch overview
                                  plus a section per sample
    batch_read_fate.png           read fate per sample
    batch_library_detection.png   library coverage per sample (only with -l)
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="psaurus-pcr",
        description=(
            "Analyse Plasmidsaurus long-read (Oxford Nanopore) PCR amplicon data: "
            "quality-filter reads, extract the insert between two fuzzy-matched "
            "flanking sequences, count unique inserts, and optionally compare "
            "them against an intended library."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs", nargs="+", metavar="INPUT",
        help="one or more inputs: FASTQ files (plain or gzipped), directories "
             "to search, and/or .zip/.tar.gz archives. Every FASTQ found is "
             "analysed separately with the same settings.",
    )
    parser.add_argument(
        "-F", "--forward", required=True, metavar="SEQ",
        help="forward flanking sequence, 5'->3'",
    )
    parser.add_argument(
        "-R", "--reverse", required=True, metavar="SEQ",
        help="reverse flanking sequence, 5'->3' (see --r-convention)",
    )
    parser.add_argument(
        "-l", "--library", metavar="FILE", default=None,
        help="optional intended-library text file, one sequence per line "
             "('#' comments and '>' FASTA headers are ignored)",
    )
    parser.add_argument(
        "-o", "--outdir", metavar="DIR", default="psaurus_pcr_results",
        help="output directory (created if needed) [default: %(default)s]",
    )

    disc = parser.add_argument_group("input discovery")
    disc.add_argument(
        "--no-recursive", dest="recursive", action="store_false",
        help="when an input is a directory, do not search it recursively "
             "(recursive by default, since delivery folders are nested)",
    )
    disc.add_argument(
        "--pattern", metavar="GLOB", default=None,
        help="glob applied when walking a directory, e.g. 'barcode*.fastq.gz'. "
             "Ignored for archives and for files named directly",
    )
    disc.add_argument(
        "--list-inputs", action="store_true",
        help="list the FASTQ files that would be analysed, then exit",
    )
    disc.add_argument(
        "--fail-fast", action="store_true",
        help="abort the batch if any sample fails; by default a failing sample "
             "is recorded in the overview table and the rest continue",
    )

    qc = parser.add_argument_group("step 1: quality filtering")
    qc.add_argument(
        "-q", "--min-quality", type=float, default=DEFAULT_MIN_MEAN_QUALITY,
        metavar="Q",
        help="minimum mean read quality to keep a read. Q20 = 99%% accuracy, "
             "appropriate for R10.4.1 + 'sup' basecalling as used by "
             "Plasmidsaurus; older chemistries were commonly filtered at Q9-Q10 "
             "[default: %(default)s]",
    )
    qc.add_argument(
        "--quality-metric", choices=("mean_phred", "error_prob"), default="mean_phred",
        help="how to summarise a read's quality: 'mean_phred' = arithmetic mean "
             "of per-base Phred scores (the conventional definition); "
             "'error_prob' = -10*log10(mean per-base error probability), which "
             "is how ONT computes a read qscore and is stricter "
             "[default: %(default)s]",
    )

    fl = parser.add_argument_group("step 2: flank matching")
    fl.add_argument(
        "--flank-edit-fraction", type=float, default=DEFAULT_FLANK_EDIT_FRACTION,
        metavar="F",
        help="allowed edits per flank as a fraction of that flank's length "
             "(0.12 gives 2 edits for a 20 bp flank, 3 for 25 bp). Too loose and "
             "short flanks hit random sequence; too tight and real reads are "
             "lost to residual nanopore error [default: %(default)s]",
    )
    fl.add_argument(
        "--forward-max-edits", type=int, default=None, metavar="N",
        help="absolute edit budget for F, overriding --flank-edit-fraction",
    )
    fl.add_argument(
        "--reverse-max-edits", type=int, default=None, metavar="N",
        help="absolute edit budget for R, overriding --flank-edit-fraction",
    )
    fl.add_argument(
        "--ambiguity-margin", type=int, default=DEFAULT_AMBIGUITY_MARGIN, metavar="N",
        help="a second, non-overlapping flank hit within this many edits of the "
             "best one flags the read as ambiguous (possible concatemer or "
             "read-through) [default: %(default)s]",
    )
    fl.add_argument(
        "--exclude-ambiguous", action="store_true",
        help="drop ambiguous reads from the count table instead of counting "
             "them with the best-scoring hit (they are reported either way)",
    )
    fl.add_argument(
        "--r-convention", choices=("auto", "literal", "revcomp"), default="auto",
        help="how to read the R sequence: 'literal' = R lies on the same strand "
             "as F (amplicon is 5'-F...R-3'); 'revcomp' = R is a conventional "
             "reverse PCR primer, so revcomp(R) is searched; 'auto' tries both "
             "on a subsample and keeps whichever explains more reads "
             "[default: %(default)s]",
    )
    fl.add_argument(
        "--orientation-probe-reads", type=int,
        default=DEFAULT_ORIENTATION_PROBE_READS, metavar="N",
        help="reads used by --r-convention auto (0 = all) [default: %(default)s]",
    )

    ins = parser.add_argument_group("step 2b: insert sanity checks")
    ins.add_argument(
        "--max-insert-length", type=int, default=None, metavar="BP",
        help="inserts longer than this are flagged and excluded from counting. "
             f"Default: {DEFAULT_LIBRARY_LENGTH_MULTIPLIER:g}x the longest library "
             f"sequence when --library is given, otherwise {DEFAULT_MAX_INSERT_LENGTH} bp",
    )
    ins.add_argument(
        "--min-insert-length", type=int, default=1, metavar="BP",
        help="inserts shorter than this are flagged and excluded; the default of "
             "1 means only truly empty inserts are dropped [default: %(default)s]",
    )
    ins.add_argument(
        "--library-length-multiplier", type=float,
        default=DEFAULT_LIBRARY_LENGTH_MULTIPLIER, metavar="X",
        help="multiplier on the longest library sequence used to derive the max "
             "insert length [default: %(default)s]",
    )

    lib = parser.add_argument_group("step 4: library comparison")
    lib.add_argument(
        "--fuzzy-library", action="store_true",
        help="also match observed sequences to library members within an edit "
             "budget. Off by default: library members are defined sequences, so "
             "a near-miss is a finding (synthesis or basecall error), not a match",
    )
    lib.add_argument(
        "--library-edit-fraction", type=float, default=0.05, metavar="F",
        help="fuzzy tolerance as a fraction of the library sequence's length "
             "[default: %(default)s]",
    )
    lib.add_argument(
        "--library-max-edits", type=int, default=None, metavar="N",
        help="absolute fuzzy tolerance, overriding --library-edit-fraction",
    )

    out = parser.add_argument_group("output")
    out.add_argument(
        "--format", choices=("tsv", "csv"), default="tsv", dest="output_format",
        help="delimiter for the tabular outputs [default: %(default)s]",
    )
    out.add_argument(
        "--per-read", action="store_true",
        help="also write a per-read classification table (one row per "
             "QC-passing read; large for deep runs)",
    )
    out.add_argument("--no-plots", action="store_true", help="skip PNG plots")
    out.add_argument(
        "--no-pdf", action="store_true",
        help="skip the PDF run report (report.pdf for one sample, "
             "batch_report.pdf for a batch)",
    )
    out.add_argument(
        "--top-n", type=int, default=20, metavar="N",
        help="number of sequences in the abundance bar chart [default: %(default)s]",
    )
    layout = out.add_mutually_exclusive_group()
    layout.add_argument(
        "--sample-dirs", dest="sample_dirs", action="store_true", default=None,
        help="always write each sample into its own <outdir>/<sample>/ directory, "
             "even for a single input (default: only when there are 2+ samples)",
    )
    layout.add_argument(
        "--no-sample-dirs", dest="sample_dirs", action="store_false",
        help="write a single sample's files straight into --outdir "
             "(only valid when there is exactly one sample)",
    )
    out.add_argument("--quiet", action="store_true", help="suppress the stdout report")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def params_from_args(args: argparse.Namespace) -> AnalysisParams:
    """Translate parsed CLI args into the core config object."""
    return AnalysisParams(
        min_mean_quality=args.min_quality,
        quality_metric=args.quality_metric,
        flank_edit_fraction=args.flank_edit_fraction,
        forward_max_edits=args.forward_max_edits,
        reverse_max_edits=args.reverse_max_edits,
        ambiguity_margin=args.ambiguity_margin,
        r_convention=args.r_convention,
        orientation_probe_reads=args.orientation_probe_reads,
        max_insert_length=args.max_insert_length,
        min_insert_length=args.min_insert_length,
        library_length_multiplier=args.library_length_multiplier,
        exclude_ambiguous=args.exclude_ambiguous,
        fuzzy_library=args.fuzzy_library,
        library_edit_fraction=args.library_edit_fraction,
        library_max_edits=args.library_max_edits,
        top_n_plot=args.top_n,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    for raw in args.inputs:
        if not Path(raw).exists():
            parser.error(f"input not found: {raw}")
    if args.library is not None and not Path(args.library).exists():
        parser.error(f"library file not found: {args.library}")

    # --list-inputs answers "what would you actually run on?" before committing
    # to a long batch -- the cheapest way to catch a wrong glob or a nested
    # delivery folder.
    try:
        samples = discover_inputs(
            args.inputs, recursive=args.recursive, pattern=args.pattern
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not samples:
        print(
            "error: no FASTQ files found in the given input(s). Expected files "
            "ending in .fastq/.fq (optionally .gz), a directory containing them, "
            "or a zip/tar archive of them.",
            file=sys.stderr,
        )
        return 2

    if args.list_inputs:
        print(f"{len(samples)} FASTQ file(s) would be analysed:")
        for sample in samples:
            print(f"  {sample.name:<30} {sample.describe()}")
        return 0

    if args.sample_dirs is False and len(samples) > 1:
        parser.error(
            f"--no-sample-dirs needs exactly one sample, but {len(samples)} were "
            "found; without per-sample directories they would overwrite each other"
        )

    try:
        params = params_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover (parser.error exits)

    try:
        batch = run_batch(
            sources=samples,
            forward_flank=args.forward,
            reverse_flank=args.reverse,
            library=args.library,
            params=params,
            continue_on_error=not args.fail_fast,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    written = batch.write_outputs(
        output_dir=args.outdir,
        output_format=args.output_format,
        write_per_read=args.per_read,
        make_plots=not args.no_plots,
        make_pdf=not args.no_pdf,
        top_n=args.top_n,
        sample_dirs=args.sample_dirs,
    )

    single = len(batch.results) == 1 and args.sample_dirs is not True
    if not args.quiet:
        if single:
            print(next(iter(batch.results.values())).text_summary())
        else:
            print(batch.text_summary())
        print("\noutputs:")
        for label, path in written.items():
            print(f"  {label:<40} {path}")

    # Warn per sample: in a batch, one dead barcode among twenty is easy to miss.
    barren = [
        name for name, result in batch.results.items()
        if result.run_summary["quantification"]["total_extracted_reads"] == 0
    ]
    if barren:
        listed = ", ".join(barren[:10]) + (" ..." if len(barren) > 10 else "")
        print(
            f"\nwarning: no inserts were extracted for {len(barren)} of "
            f"{len(batch.results)} sample(s): {listed}. Check the flank sequences, "
            "try --r-convention literal/revcomp explicitly, relax --min-quality, "
            "or raise --flank-edit-fraction.",
            file=sys.stderr,
        )
    if batch.failures:
        print(
            f"\nwarning: {len(batch.failures)} sample(s) failed to analyse:",
            file=sys.stderr,
        )
        for failure in batch.failures:
            print(f"  {failure.name}: {failure.error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
