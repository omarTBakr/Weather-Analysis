"""Shared matplotlib style (validated categorical palette, recessive chrome) for report figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from weather.enums import Palette

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def apply_style() -> None:
    """Set the project-wide matplotlib rcParams (colours, fonts, grid, 600-dpi export)."""
    plt.rcParams.update(
        {
            "figure.facecolor": Palette.SURFACE,
            "axes.facecolor": Palette.SURFACE,
            "savefig.facecolor": Palette.SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 600,  # high-res rasters: figures sit in one 2-column slot, readers zoom in
            "pdf.fonttype": 42,  # embed TrueType so text stays selectable/searchable in the PDF
            "font.family": "sans-serif",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9.5,
            "axes.labelcolor": Palette.INK_2,
            "axes.edgecolor": Palette.AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": Palette.GRID,
            "grid.linewidth": 0.6,
            "xtick.color": Palette.MUTED,
            "ytick.color": Palette.MUTED,
            "xtick.labelcolor": Palette.INK_2,
            "ytick.labelcolor": Palette.INK_2,
            "text.color": Palette.INK,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 1.8,
            "axes.prop_cycle": matplotlib.cycler(color=Palette.SERIES),
        }
    )


def legend_below(fig, axes=None, ncol: int | None = None, **kwargs):
    """Figure-level legend placed *underneath* the plot area, so it can never cover data.

    Handles are collected from `axes` (an Axes, a list of Axes, or all axes of the figure) and de-duplicated
    by label. The legend's top edge sits at the bottom of the figure (below tick labels and x-labels, which
    `tight_layout` keeps inside the figure); `bbox_inches="tight"` in `save` then grows the canvas to include it.
    """
    if axes is None:
        axes = fig.axes
    elif not isinstance(axes, list | tuple):
        axes = [axes]
    handles, labels = [], []
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            if lab and not lab.startswith("_") and lab not in labels:
                handles.append(h)
                labels.append(lab)
    if not handles:
        return None
    ncol = ncol or min(len(labels), 4)
    fig._legend_below = dict(handles=handles, labels=labels, ncol=ncol, **kwargs)  # placed in save()
    return fig._legend_below


def _place_legend_below(fig) -> None:
    """Draw the legend requested by `legend_below` underneath the axes (called by `save` after layout)."""
    spec = getattr(fig, "_legend_below", None)
    if spec:
        fig.legend(
            spec["handles"],
            spec["labels"],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=spec["ncol"],
            frameon=False,
            **{k: v for k, v in spec.items() if k not in {"handles", "labels", "ncol"}},
        )


def annotate_heatmap(
    ax, values: np.ndarray, fmt: str, fontsize: float, light_text, lower_triangle: bool = False
) -> None:
    """Write each cell's value (formatted with `fmt`) at its centre on an `imshow` heatmap.

    `light_text(v)` decides whether a cell is dark enough to need white text (default ink otherwise);
    `lower_triangle=True` annotates only cells on or below the diagonal.
    """
    for i in range(values.shape[0]):
        for j in range(i + 1 if lower_triangle else values.shape[1]):
            v = values[i, j]
            ax.text(
                j,
                i,
                format(v, fmt),
                ha="center",
                va="center",
                fontsize=fontsize,
                color="#ffffff" if light_text(v) else Palette.INK,
            )


def save(fig, path: Path, tight: bool = True) -> Path:
    """Write the figure as a 600-dpi PNG (README / MLflow) and a vector PDF with the same stem (LaTeX report).

    `\\includegraphics{name}` without an extension makes pdflatex pick the PDF.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    _place_legend_below(fig)  # after tight_layout so the legend is laid out below the finished axes
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path
