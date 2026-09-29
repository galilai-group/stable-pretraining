"""Exported tables and figures must preserve values, labels, and image geometry."""

import copy
import io

import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_pretraining.utils import visualization as viz

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize(
    "text,expected",
    [
        (r"a_b&c%", r"a\_b\&c\%"),
        ("\\", r"\textbackslash{}"),
        ("~^", r"\textasciitilde{}\textasciicircum{}"),
        ("{$#}", r"\{\$\#\}"),
        (7, 7),
    ],
)
def test_latex_escaping_does_not_reescape_inserted_commands(text, expected):
    assert viz.latex_escape(text) == expected


@pytest.mark.parametrize("bold", ["row", "col", "overall", None])
@pytest.mark.parametrize("percent", [False, True])
def test_latex_table_formats_values_without_mutating_dataframe(bold, percent):
    frame = pd.DataFrame(
        [[0.25, 0.75], [np.nan, 0.5]],
        index=pd.Index(["b_2", "a_1"], name="model_name"),
        columns=pd.Index(["z", "a"], name="metric_name"),
    )
    before = frame.copy(deep=True)
    output = viz.format_df_to_latex(
        frame,
        caption="Scores",
        label="tab:scores",
        bold=bold,
        show_percent_symbol=percent,
        sort_index=True,
        sort_columns=True,
    )
    assert "25.00" in output and "75.00" in output and "50.00" in output
    assert "b\\_2" in output and "model\\_name" in output
    assert "tab:scores" in output and "Scores" in output
    assert "\\%" in output if percent else "All values are percentages" in output
    if bold:
        assert "\\bfseries" in output or "\\textbf" in output
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("multilevel", [False, True])
def test_latex_unit_annotations_and_non_numeric_cells(multilevel):
    columns = (
        pd.MultiIndex.from_tuples(
            [("eval", "a_b"), ("eval", "name")], names=["split", None]
        )
        if multilevel
        else ["a_b", "name"]
    )
    frame = pd.DataFrame([[0.5, "model"]], columns=columns)
    output = viz.format_df_to_latex(frame, unit_annotation="columns", bold=None)
    assert "50.00" in output and "model" in output and "\\%" in output
    assert "a\\_b" in output
    assert "model" in viz.format_df_to_latex(frame, escape_headers=False, bold=None)


def test_multiindex_labels_preserve_levels_and_missing_names():
    index = pd.MultiIndex.from_tuples(
        [("a_b", 1), ("x&y", 2)], names=["group_name", None]
    )
    escaped = viz.escape_labels(index)
    assert list(escaped) == [(r"a\_b", 1), (r"x\&y", 2)]
    assert escaped.names == [r"group\_name", None]


@pytest.mark.parametrize("constant", [False, True])
def test_display_image_conversion_is_bounded_and_preserves_geometry(constant):
    image = torch.ones(3, 2, 4) if constant else torch.arange(24.0).reshape(3, 2, 4)
    converted = viz._make_image(image)
    assert converted.shape == (2, 4, 3)
    assert ((converted >= 0) & (converted <= 255)).all()
    if constant:
        assert (converted == 0).all()
    else:
        assert converted.min() == 0 and converted.max() == 255


def test_grid_plot_and_bars_leave_input_options_unchanged():
    figure, axes = plt.subplots()
    bars = [(0, 2), (1, 3, {"thickness": 0.5, "facecolor": "red"})]
    before = copy.deepcopy(bars)
    matrix = np.arange(16).reshape(4, 4)
    artist = viz.imshow_with_grid(axes, matrix, bars=bars, extent=[0, 4, 0, 4])
    np.testing.assert_array_equal(artist.get_array(), matrix)
    assert len(axes.collections) == 2 and len(axes.patches) == 4
    assert bars == before
    destination = io.BytesIO()
    figure.savefig(destination, format="png")
    assert destination.getvalue().startswith(b"\x89PNG")


def test_similarity_figure_contains_source_images_and_graph():
    images = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    graph = torch.tensor([[1.0, 0.2], [0.2, 1.0]])
    viz.visualize_images_graph(images, graph, zoom_on=2)
    figure = plt.gcf()
    rendered = [artist for axes in figure.axes for artist in axes.images]
    assert len(rendered) == 4
    np.testing.assert_array_equal(rendered[0].get_array(), graph.numpy())
    np.testing.assert_array_equal(
        rendered[2].get_array(), viz._make_image(images[0]).numpy()
    )
    destination = io.BytesIO()
    figure.savefig(destination, format="png")
    assert destination.getvalue().startswith(b"\x89PNG")
