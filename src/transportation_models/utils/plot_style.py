"""Shared matplotlib styling for the figures that go into the paper.

The figures are authored at their final printed size -- IEEE Transactions
single-column width, 3.5 in -- and included at 1:1::

    \\includegraphics[width=\\columnwidth]{fig.pdf}

That is what makes the text come out at the size it says: LaTeX applies no
scaling, so an 8 pt label in the figure is an 8 pt label on the page. The
older approach of drawing a 20 in canvas with 32 pt text and letting
``width=\\columnwidth`` shrink it 5.7x rendered that label at 5.6 pt --
smaller than the caption. Anything authored here should therefore keep
``COLUMN_W`` as its width and leave the point sizes alone; make a figure
taller, never wider.

Apply with ``plt.rc_context(PAPER_RC)`` (not ``plt.rcParams.update``) so
the sizes do not leak into plots rendered later in the same process, and
save at ``PAPER_DPI``.
"""

from __future__ import annotations

# IEEE Transactions single-column text width, inches. Full text width
# (a figure* spanning both columns) is 7.16 in.
COLUMN_W = 3.5

PAPER_DPI = 300

PAPER_RC = {
    # Body text in the IEEE Trans template is 10 pt; figure text sits a
    # couple of steps below it, with the ticks one step below the labels.
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.title_fontsize": 7,
    # Defaults thick enough to survive printing; individual plots that set
    # their own lw/ms override these.
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.6,
}
