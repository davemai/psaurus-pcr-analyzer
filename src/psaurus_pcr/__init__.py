"""psaurus_pcr: analysis of Plasmidsaurus long-read (ONT) PCR amplicon data.

The package is deliberately split so that *every* analysis step is a pure,
importable function that works on in-memory data as happily as on file paths.
The CLI (:mod:`psaurus_pcr.cli`) and the Streamlit app (``streamlit_app.py``)
are thin wrappers around :func:`psaurus_pcr.pipeline.run_analysis` (one FASTQ)
and :func:`psaurus_pcr.batch.run_batch` (many).
"""

from psaurus_pcr.batch import BatchResult, run_batch
from psaurus_pcr.config import AnalysisParams
from psaurus_pcr.inputs import SampleInput, discover_inputs
from psaurus_pcr.pipeline import AnalysisResult, run_analysis

__all__ = [
    "AnalysisParams",
    "AnalysisResult",
    "BatchResult",
    "SampleInput",
    "discover_inputs",
    "run_analysis",
    "run_batch",
]
__version__ = "0.3.0"
