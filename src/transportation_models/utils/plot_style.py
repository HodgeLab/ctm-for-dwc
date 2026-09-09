"""Shared matplotlib styling for the figures that get shown at full size.

``PRESENTATION_RC`` is the font block for plots destined for talks and the
paper, where the default 10 pt text is unreadable once the figure is scaled
down onto a slide. Apply it with ``plt.rc_context(PRESENTATION_RC)`` (not
``plt.rcParams.update``) so the sizes do not leak into the inline-sized
plots rendered later in the same process, and scale the figure up with the
text so the labels still fit.
"""

from __future__ import annotations

PRESENTATION_RC = {
    "font.size": 32,
    "axes.titlesize": 32,
    "axes.labelsize": 28,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 20,
}
