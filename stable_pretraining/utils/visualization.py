"""Visualization utilities for SSL experiments."""

from typing import Optional, Union

import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import pandas as pd
from pandas.api.types import is_number


def latex_escape(s):
    """Escape LaTeX special characters in a string."""
    if not isinstance(s, str):
        return s
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    # Escape original characters once, leaving generated LaTeX commands intact.
    return "".join(replacements.get(character, character) for character in s)


def escape_labels(idx_or_cols):
    """Recursively escape all labels in a pandas Index or MultiIndex."""
    if isinstance(idx_or_cols, pd.MultiIndex):
        new_tuples = []
        for tup in idx_or_cols:
            new_tuples.append(tuple(latex_escape(x) for x in tup))
        new_names = [
            latex_escape(n) if n is not None else None for n in idx_or_cols.names
        ]
        return pd.MultiIndex.from_tuples(new_tuples, names=new_names)
    else:
        return pd.Index(
            [latex_escape(x) for x in idx_or_cols],
            name=latex_escape(idx_or_cols.name) if idx_or_cols.name else None,
        )


def format_df_to_latex(
    df,
    caption=None,
    label=None,
    bold="row",  # 'row', 'col', 'overall', or None
    na_rep="–",
    sort_index=False,
    sort_columns=False,
    column_format=None,
    position="htbp",
    escape_headers=True,
    show_percent_symbol=False,
    unit_annotation="caption",  # 'caption', 'columns', or None
):
    """Format a MultiIndex DataFrame for LaTeX export with percent formatting (no % symbol).

    Escapes LaTeX special characters in all headers if escape_headers=True.
    """
    df = df.copy()
    if sort_index:
        df = df.sort_index()
    if sort_columns:
        df = df.sort_index(axis=1)
    # Optionally annotate units in caption or columns
    cap = caption
    if not show_percent_symbol:
        if unit_annotation == "caption":
            cap = (caption or "") + " (All values are percentages)"
        elif unit_annotation == "columns":
            # Add " (%)" to the last column level name
            if isinstance(df.columns, pd.MultiIndex):
                new_names = list(df.columns.names)
                new_names[-1] = (new_names[-1] or "") + " (%)"
                df.columns = pd.MultiIndex.from_tuples(df.columns, names=new_names)
            else:
                df.columns = [str(c) + " (%)" for c in df.columns]
    # Escape headers if requested
    if escape_headers:
        df.index = escape_labels(df.index)
        df.columns = escape_labels(df.columns)
        styler_escape = None
    else:
        styler_escape = "latex"

    # Formatter: percent with 2 decimals, handle NaN, with or without %
    def percent_or_plain(x, show_symbol=show_percent_symbol):
        if pd.isna(x):
            return na_rep
        if not is_number(x):
            # It's a string or other non-numeric type - return as-is
            return str(x)
        # It's a number - format as percentage
        val = f"{x * 100:.2f}"
        return f"{val}\\%" if show_symbol else val

    styler = df.style.format(
        lambda x: percent_or_plain(x, show_percent_symbol),
        na_rep=na_rep,
        escape=styler_escape,
    )
    # Bolding logic (using LaTeX property mapping, not literal \textbf{})
    if bold == "row":
        styler = styler.highlight_max(axis=1, props="font-weight: bold;")
    elif bold == "col":
        styler = styler.highlight_max(axis=0, props="font-weight: bold;")
    elif bold == "overall":
        max_val = np.nanmax(df.values)

        def bold_overall(val):
            return (
                "font-weight: bold;"
                if np.isclose(val, max_val, equal_nan=False)
                else ""
            )

        styler = styler.apply(lambda column: column.map(bold_overall))
    # else: no bolding
    latex = styler.to_latex(
        hrules=True,
        caption=cap,
        label=label,
        column_format=column_format,
        position=position,
        multicol_align="c",
        environment=None,
        convert_css=True,
    )
    return latex


def _make_image(x):
    """Convert tensor to displayable image format.

    Args:
        x: Image tensor in CHW format

    Returns:
        Image array in HWC format with values in [0, 255]
    """
    x = x.float()
    span = (x.max() - x.min()).clamp_min(torch.finfo(x.dtype).eps)
    return (255 * (x - x.min()) / span).int().permute(1, 2, 0)


def imshow_with_grid(
    ax,
    G: Union[np.ndarray, torch.Tensor],
    linewidth: Optional[float] = 0.4,
    color: Optional[Union[str, tuple]] = "black",
    bars=[],
    **kwargs,
):
    """Display a matrix with grid lines overlaid.

    Args:
        ax: Matplotlib axes to plot on
        G: Matrix to display
        linewidth: Width of grid lines
        color: Color of grid lines
        bars: List of bar specifications for highlighting regions
        **kwargs: Additional arguments for imshow

    Returns:
        The image object from imshow
    """
    extent = [0, 1, 0, 1]
    if "extent" in kwargs:
        del kwargs["extent"]
    im = ax.imshow(G, extent=extent, **kwargs)
    shape = G.shape
    line_segments = []

    # horizontal lines
    for y in np.linspace(extent[2], extent[3], shape[1] + 1):
        line_segments.append([(extent[0], y), (extent[1], y)])
    # vertical lines
    for x in np.linspace(extent[0], extent[1], shape[0] + 1):
        line_segments.append([(x, extent[2]), (x, extent[3])])
    collection = LineCollection(line_segments, color=color, linewidth=linewidth)
    ax.add_collection(collection)

    # border line
    line_segments = [
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 1), (1, 1)],
        [(1, 0), (1, 1)],
    ]
    collection = LineCollection(line_segments, color="black", linewidth=linewidth * 3)
    ax.add_collection(collection)
    step = 1 / len(G)
    for bar in bars:
        if len(bar) == 2:
            barkwargs = {}
        else:
            barkwargs = bar[2]
        if "thickness" in barkwargs:
            thickness = barkwargs["thickness"]
            del barkwargs["thickness"]
        else:
            thickness = 1

        rect = Rectangle(
            xy=(bar[0] / len(G), 1),
            width=(bar[1] - bar[0]) / len(G),
            height=step * thickness,
            **barkwargs,
        )
        ax.add_patch(rect)
        rect = Rectangle(
            xy=(-step * thickness, 1 - bar[1] / len(G)),
            width=step * thickness,
            height=(bar[1] - bar[0]) / len(G),
            **barkwargs,
        )
        barkwargs["thickness"] = thickness
        ax.add_patch(rect)
        ax.set_xlim(-step * thickness, 1 + step)
        ax.set_ylim(-step, 1 + step * thickness)
    return im


def _plot_square(fig, x0, y0, x1, y1):
    """Plot a square outline on a figure.

    Args:
        fig: Matplotlib figure
        x0: Bottom-left corner x-coordinate
        y0: Bottom-left corner y-coordinate
        x1: Top-right corner x-coordinate
        y1: Top-right corner y-coordinate
    """
    fig.patches.append(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=None,
            edgecolor="tab:blue",
            linewidth=3,
            transform=fig.transFigure,
            figure=fig,
        )
    )


def visualize_images_graph(x, G, zoom_on=8):
    """Visualize images and their similarity graph with zoom detail.

    Creates a visualization showing:
    - A grid of sample images
    - The full similarity matrix
    - A zoomed-in view of the top-left portion of the matrix
    - Connection lines between the views

    Args:
        x: List or tensor of images
        G: Similarity/adjacency matrix
        zoom_on: Number of rows/columns to show in zoomed view
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.set_axis_off()

    # add in the overall graph
    inset = fig.add_axes([0.5, 0.005, 0.495, 0.8])
    imshow_with_grid(inset, G, linewidth=0.01)
    plt.setp(inset, xticks=[], yticks=[])
    bboxr = inset.get_position()

    # add in the zoomed one
    inset = fig.add_axes([0.04, 0.02, 0.42, 0.68])
    imshow_with_grid(inset, G[:zoom_on, :zoom_on], vmin=G.min(), vmax=G.max())
    plt.setp(inset, xticks=[], yticks=[])
    bboxl = inset.get_position()

    # add in the number of rows/columns
    dx = (np.max(bboxl.intervalx) - np.min(bboxl.intervalx)) / zoom_on
    dy = (np.max(bboxl.intervaly) - np.min(bboxl.intervaly)) / zoom_on
    for i in range(zoom_on):
        fig.text(
            np.min(bboxl.intervalx) + dx / 2 + dx * i,
            np.max(bboxl.intervaly) + dy / 3,
            str(i + 1),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )
        fig.text(
            np.min(bboxl.intervalx) - dx / 3,
            np.max(bboxl.intervaly) - dy / 2 - dy * i,
            str(i + 1),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )

    # add the inset zoom lines
    _plot_square(
        fig,
        np.min(bboxl.intervalx),
        np.min(bboxl.intervaly),
        np.max(bboxl.intervalx),
        np.max(bboxl.intervaly),
    )
    pct = zoom_on / G.size(0)
    delta_x = (np.max(bboxr.intervalx) - np.min(bboxr.intervalx)) * pct
    delta_y = (np.max(bboxr.intervaly) - np.min(bboxr.intervaly)) * pct
    _plot_square(
        fig,
        np.min(bboxr.intervalx),
        np.max(bboxr.intervaly) - delta_y,
        np.min(bboxr.intervalx) + delta_x,
        np.max(bboxr.intervaly),
    )
    fig.add_artist(
        lines.Line2D(
            [np.min(bboxl.intervalx), np.min(bboxr.intervalx)],
            [np.max(bboxl.intervaly), np.max(bboxr.intervaly)],
            linewidth=1,
        )
    )
    fig.add_artist(
        lines.Line2D(
            [np.max(bboxl.intervalx), np.min(bboxr.intervalx) + delta_x],
            [np.min(bboxl.intervaly), np.max(bboxr.intervaly) - delta_y],
            linewidth=1,
        )
    )

    # adding the images
    for i in range(zoom_on):
        inset = fig.add_axes(
            [0.002 + i / zoom_on, 0.815, 1 / (zoom_on + 1), 5.5 / 4 / zoom_on]
        )
        fig.text(
            i * 1 / zoom_on + 0.5 / zoom_on,
            1.003,
            rf"$x_{i + 1}$",
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=12,
        )
        inset.imshow(_make_image(x[i]), aspect="auto", interpolation="nearest")
        plt.setp(inset, xticks=[], yticks=[])

    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
