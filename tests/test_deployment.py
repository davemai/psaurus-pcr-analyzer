"""Deployment readiness for Streamlit Community Cloud.

These tests encode the assumptions the hosted environment makes, so a future
change that quietly breaks the deployed build fails here first:

* the environment is rebuilt from requirements.txt alone -- anything the app
  imports but does not declare will break the build, not this machine;
* the filesystem is ephemeral and not shared between users, so no output may
  depend on a file persisting between interactions;
* a visitor is untrusted, so a server-side path must stay confined.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

import pytest

from psaurus_pcr.inputs import PathNotAllowed, resolve_under_root

REPO = Path(__file__).resolve().parents[1]
ENTRYPOINT = REPO / "streamlit_app.py"

#: Modules that ship with Python, so they need no requirement line.
STDLIB = set(sys.stdlib_module_names)

#: Imported by tests/tooling only; never imported by the shipped app.
DEV_ONLY = {"pytest", "pypdf", "pyflakes", "conftest"}


def _requirement_names(path: Path) -> dict:
    names = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = re.match(r"^([A-Za-z0-9_.\-]+)\s*(==|>=|~=)\s*([^\s;]+)", line)
        if match:
            names[match.group(1).lower().replace("-", "_")] = (match.group(2), match.group(3))
    return names


def _top_level_imports(path: Path) -> set:
    tree = ast.parse(path.read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


#: Import name -> distribution name, where they differ.
DISTRIBUTION = {"bio": "biopython", "psaurus_pcr": None}


# --- 1: the dependency file must be complete and pinned --------------------


def test_every_third_party_import_is_declared_in_requirements():
    """Community Cloud installs only requirements.txt.

    An import that happens to work here because something else pulled it in
    (numpy, once) would fail the deployed build, so the declaration is checked
    against what the shipped code actually imports.
    """
    declared = _requirement_names(REPO / "requirements.txt")
    shipped = [ENTRYPOINT] + sorted((REPO / "src" / "psaurus_pcr").glob("*.py"))

    missing = {}
    for path in shipped:
        for module in _top_level_imports(path):
            name = module.lower()
            if name in STDLIB or name in DEV_ONLY:
                continue
            distribution = DISTRIBUTION.get(name, name)
            if distribution is None:      # the package itself
                continue
            if distribution.lower() not in declared:
                missing.setdefault(distribution, []).append(path.name)
    assert not missing, f"imported but not in requirements.txt: {missing}"


def test_runtime_requirements_are_pinned_exactly():
    """A fresh cloud build must resolve to the versions this was tested against."""
    declared = _requirement_names(REPO / "requirements.txt")
    assert declared, "requirements.txt parsed as empty"
    loose = {name: spec for name, (spec, _) in declared.items() if spec != "=="}
    assert not loose, f"not pinned with ==: {loose}"


def test_test_only_packages_are_not_in_the_runtime_requirements():
    """pytest/pypdf on the server would only slow the build down."""
    declared = _requirement_names(REPO / "requirements.txt")
    assert not (DEV_ONLY & set(declared)), f"dev packages leaked: {DEV_ONLY & set(declared)}"

    dev = _requirement_names(REPO / "requirements-dev.txt")
    assert "pytest" in dev and "pypdf" in dev
    assert "-r requirements.txt" in (REPO / "requirements-dev.txt").read_text()


# --- 2: entry point ---------------------------------------------------------


def test_entrypoint_exists_at_the_repo_root():
    """Community Cloud is pointed at one file; it must be unambiguous."""
    assert ENTRYPOINT.is_file()
    assert ENTRYPOINT.name == "streamlit_app.py"
    assert list(REPO.glob("*.py")) == [ENTRYPOINT], "more than one candidate at the root"


def test_entrypoint_imports_the_package_without_installation():
    """The cloud installs requirements.txt, not this repo, so src/ must be
    put on sys.path by the app itself."""
    source = ENTRYPOINT.read_text()
    assert 'sys.path.insert' in source
    assert (REPO / "src" / "psaurus_pcr" / "__init__.py").is_file()


# --- 3: no secrets, and config is sane -------------------------------------


def test_no_secrets_file_is_committed():
    assert not (REPO / ".streamlit" / "secrets.toml").exists()
    ignored = (REPO / ".gitignore").read_text()
    assert ".streamlit/secrets.toml" in ignored


def test_streamlit_config_is_valid_and_bounded():
    config = tomllib.loads((REPO / ".streamlit" / "config.toml").read_text())
    upload = config["server"]["maxUploadSize"]
    assert isinstance(upload, int) and 0 < upload <= 500, upload
    assert config["browser"]["gatherUsageStats"] is False


def test_gitignore_covers_python_and_output_cruft():
    ignored = (REPO / ".gitignore").read_text()
    for pattern in ("__pycache__/", ".pytest_cache/", "*.egg-info/", ".venv/",
                    "psaurus_pcr_results/"):
        assert pattern in ignored, pattern


# --- 4: repo stays small enough to deploy ----------------------------------


def test_checked_in_data_files_stay_small():
    """Community Cloud clones the repo on every build; fixtures must stay light."""
    data = list((REPO / "examples").rglob("*"))
    files = [p for p in data if p.is_file()]
    total_mb = sum(p.stat().st_size for p in files) / 1e6
    assert total_mb < 5, f"examples/ is {total_mb:.1f} MB"
    for path in files:
        size_mb = path.stat().st_size / 1e6
        assert size_mb < 2, f"{path.name} is {size_mb:.1f} MB; consider trimming or LFS"


# --- 5: server-side paths are confined -------------------------------------


def test_resolve_under_root_allows_paths_inside_the_root(tmp_path):
    (tmp_path / "data").mkdir()
    target = tmp_path / "data" / "reads.fastq"
    target.write_text("@r\nA\n+\nI\n")
    assert resolve_under_root("data/reads.fastq", tmp_path) == target.resolve()
    assert resolve_under_root(target, tmp_path) == target.resolve()


def test_resolve_under_root_rejects_traversal_and_absolute_escapes(tmp_path):
    (tmp_path / "data").mkdir()
    for candidate in ("../outside", "data/../../outside", "/etc"):
        with pytest.raises(PathNotAllowed):
            resolve_under_root(candidate, tmp_path)


def test_resolve_under_root_rejects_a_symlink_that_escapes(tmp_path):
    """resolve() follows links, so a link out of the root is caught."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.fastq").write_text("@r\nA\n+\nI\n")
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(outside)
    with pytest.raises(PathNotAllowed):
        resolve_under_root("link/secret.fastq", root)


def test_resolve_under_root_reports_a_missing_path_distinctly(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_under_root("nope.fastq", tmp_path)


def test_no_root_means_no_confinement(tmp_path):
    target = tmp_path / "reads.fastq"
    target.write_text("@r\nA\n+\nI\n")
    assert resolve_under_root(target, None) == target.resolve()


# --- 6: outputs are produced in memory, never via disk ---------------------


def test_the_app_never_writes_outputs_to_disk():
    """The cloud filesystem is ephemeral and per-session; every download must
    be built from bytes held in memory at generation time."""
    source = ENTRYPOINT.read_text()
    for forbidden in ("write_outputs(", ".to_csv(\"", "open(", "mkdir(", "savefig("):
        assert forbidden not in source, f"{forbidden} suggests a disk round-trip"
    # Downloads are served from in-memory bytes.
    assert "st.download_button" in source
    assert "report_bytes" in source


def test_caches_are_bounded_and_expire():
    """An unbounded st.cache_data is the documented way to exhaust the memory
    budget; losing a bounded cache only costs a recompute."""
    source = ENTRYPOINT.read_text()
    decorators = re.findall(r"@st\.cache_data\(([^)]*)\)", source)
    assert decorators, "no cached functions found"
    for decorator in decorators:
        assert "ttl=" in decorator, decorator
        assert "max_entries=" in decorator, decorator


# --- 7: large tables must not be re-serialised on every rerun --------------


def test_large_tables_are_serialised_lazily():
    """st.download_button takes bytes, not a callback.

    A direct button therefore re-encodes its whole frame on every script
    re-run; for the per-read table (one row per read) that is a large CSV
    rebuilt every time a widget moves.
    """
    source = ENTRYPOINT.read_text()
    assert "LAZY_DOWNLOAD_ROWS" in source
    assert "MAX_DISPLAY_ROWS" in source
    # The download helper must branch on size rather than always encoding.
    helper = source[source.index("def _download("):source.index("def _show(")]
    assert "if len(frame) <= LAZY_DOWNLOAD_ROWS" in helper
    assert "st.session_state" in helper


def test_session_state_payloads_use_a_single_slot():
    """Session state lives for the whole session; one key per rendered file
    would grow without bound."""
    source = ENTRYPOINT.read_text()
    for slot in ('st.session_state["pdf_payload"]', 'st.session_state["csv_payload"]'):
        assert slot in source, slot
