"""
A vertically-scrollable container for ttk content.

Tkinter has no built-in "scrollable frame" widget -- the standard trick is a
Canvas + Scrollbar with a plain Frame embedded inside the Canvas. This
wraps that pattern into one reusable class so every tab can just build its
widgets onto `.body` instead of onto itself, and get a scrollbar for free
once the content grows taller than the visible area.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class ScrollableFrame(ttk.Frame):
    """
    Usage:
        scrollable = ScrollableFrame(parent)
        scrollable.pack(fill="both", expand=True)
        ttk.Label(scrollable.body, text="...").pack(...)   # build ON .body, not on scrollable itself
    """

    def __init__(self, parent, bg: str = "#f4f6f8"):
        super().__init__(parent)

        canvas = tk.Canvas(self, highlightthickness=0, bg=bg)
        vscroll = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)

        # This is the frame tabs actually build their widgets onto. It lives
        # inside the canvas, which is what makes it scrollable.
        self.body = ttk.Frame(canvas)
        body_window = canvas.create_window((0, 0), window=self.body, anchor="nw")

        # Whenever the body's contents change size, tell the canvas how far
        # it's allowed to scroll (its "scrollregion").
        self.body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        # Whenever the visible canvas area is resized (e.g. the window is
        # resized), stretch the body to match its width so widgets using
        # fill="x" still expand correctly instead of staying a fixed width.
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(body_window, width=e.width))

        canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        # Mouse-wheel scrolling. bind_all() is deliberately scoped to only
        # be active while the cursor is over THIS canvas (bound/unbound on
        # Enter/Leave) -- binding it permanently would break scrolling on
        # every other tab, since bind_all() is global to the whole app, not
        # per-widget.
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_wheel(_event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)  # Windows / Mac
            canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))  # Linux scroll up
            canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))   # Linux scroll down

        def _unbind_wheel(_event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        canvas.bind("<Enter>", _bind_wheel)
        canvas.bind("<Leave>", _unbind_wheel)
