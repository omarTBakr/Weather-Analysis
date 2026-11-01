"""Shared matplotlib style (validated categorical palette, recessive chrome) for report figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical order (fixed, never cycled) and ink/chrome tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIV_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a5a4", "#e34948", "#a32727"]
)
CRITICAL = "#d03b3b"

CONTINENT_ORDER = ["Africa", "Asia", "Europe", "North America", "South America", "Oceania"]
CONTINENT_COLORS = dict(zip(CONTINENT_ORDER, SERIES, strict=False))


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 600,  # high-res rasters: figures sit in one 2-column slot, readers zoom in
            "pdf.fonttype": 42,  # embed TrueType so text stays selectable/searchable in the PDF
            "font.family": "sans-serif",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9.5,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 1.8,
            "axes.prop_cycle": matplotlib.cycler(color=SERIES),
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
