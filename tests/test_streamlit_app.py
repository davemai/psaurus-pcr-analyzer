"""Smoke test for the Streamlit wrapper.

The point is not to re-test the science (that is covered by the core tests) but
to catch the class of breakage a web front end is prone to: a widget API
change, a bad import, or a code path that raises before the user has supplied
any input.
"""

from __future__ import annotations

from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip(
    "streamlit.testing.v1", reason="streamlit is an optional extra"
)

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def test_app_renders_and_waits_for_input():
    app = streamlit_testing.AppTest.from_file(APP, default_timeout=120).run()
    assert not app.exception
    assert any("Run analysis" in message.value for message in app.info)


def test_run_button_validates_inputs_instead_of_crashing():
    app = streamlit_testing.AppTest.from_file(APP, default_timeout=120).run()
    app.sidebar.button[0].click().run()
    assert not app.exception
    assert any("upload at least one FASTQ file" in message.value for message in app.error)


def test_app_runs_a_real_batch_from_a_server_path(write_fastq_dir, monkeypatch):
    """Drive the whole app end to end via the server-path input.

    A browser cannot upload a directory, so the server-path box is the route
    people will actually use on a cluster -- and it is the only input AppTest
    can set, which makes it the natural integration test.

    The directory pytest hands us is outside the app root, so this also
    exercises the opt-in that a self-hosted deployment would set.
    """
    from conftest import F, HIGH_Q, LIB_A, LIB_B, R, amplicon

    monkeypatch.setenv("PSAURUS_ALLOW_ANY_PATH", "1")
    root = write_fastq_dir({
        "barcode01.fastq": [(f"a{i}", amplicon(LIB_A), HIGH_Q) for i in range(4)],
        "barcode02.fastq": [(f"b{i}", amplicon(LIB_B), HIGH_Q) for i in range(2)],
    })

    app = streamlit_testing.AppTest.from_file(APP, default_timeout=180).run()
    app.text_input("server_path").set_value(str(root))
    app.text_input("forward_flank").set_value(F)
    app.text_input("reverse_flank").set_value(R)
    app.button("run").click().run()

    assert not app.exception
    assert not app.error

    batch = app.session_state["batch"]
    assert batch.sample_names == ["barcode01", "barcode02"]
    assert int(batch["barcode01"].unique_sequences.loc[0, "count"]) == 4
    assert int(batch["barcode02"].unique_sequences.loc[0, "count"]) == 2
    assert int(batch.sample_overview["total_reads"].sum()) == 6
    # The batch-only tab appears once there is more than one sample.
    assert any("Batch overview" in str(tab.label) for tab in app.tabs)


def test_server_paths_are_confined_to_the_app_root_by_default(write_fastq_dir, monkeypatch):
    """A public deployment must not turn the path box into a file-read primitive.

    Without the explicit opt-in, a path outside the app's own directory is
    refused before it ever reaches the filesystem.
    """
    monkeypatch.delenv("PSAURUS_ALLOW_ANY_PATH", raising=False)
    from conftest import HIGH_Q, LIB_A, amplicon

    outside = write_fastq_dir({"barcode01.fastq": [("r1", amplicon(LIB_A), HIGH_Q)]})

    app = streamlit_testing.AppTest.from_file(APP, default_timeout=120).run()
    app.text_input("server_path").set_value(str(outside))
    app.run()

    assert not app.exception
    messages = [e.value for e in app.error]
    assert any("outside the directory this app is allowed to read" in m for m in messages)
    assert app.session_state.get("batch") is None


def test_a_relative_path_inside_the_app_root_still_works(monkeypatch):
    """The bundled demo data must stay reachable under the default confinement.

    `examples/batch` ships in the repo, so it exists on a Community Cloud
    container too -- this is the path a visitor to the deployed app can type.
    """
    monkeypatch.delenv("PSAURUS_ALLOW_ANY_PATH", raising=False)

    app = streamlit_testing.AppTest.from_file(APP, default_timeout=300).run()
    app.text_input("server_path").set_value("examples/batch")
    app.text_input("forward_flank").set_value("CAGTTCGGACTTAGCCATGACT")
    app.text_input("reverse_flank").set_value("TGGACCAATCGTTACGGTCAAG")
    app.button("run").click().run()

    assert not app.exception
    assert not app.error
    batch = app.session_state["batch"]
    assert batch.sample_names == ["barcode01", "barcode02", "barcode03", "barcode04"]


def test_traversal_out_of_the_app_root_is_refused(monkeypatch):
    monkeypatch.delenv("PSAURUS_ALLOW_ANY_PATH", raising=False)
    app = streamlit_testing.AppTest.from_file(APP, default_timeout=120).run()
    app.text_input("server_path").set_value("examples/../../../../etc")
    app.run()
    assert not app.exception
    assert any("outside the directory" in e.value for e in app.error)
