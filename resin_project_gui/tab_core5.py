"""
GUI tab standing in for Core Function #5 -- Web UI / API Service Layer.

The original table specifies a React.js frontend + FastAPI backend + AWS
cloud infrastructure. This whole desktop app IS the simplified local
equivalent of that layer (it's what calls into Core 1/2/3/4 and displays
their results); this tab specifically covers the "visualization dashboard"
sub-function -- a recommended-recipe list and a scatter plot -- using the
Pareto front already computed in the Core 1 tab, since matplotlib/Tkinter
can do this locally without standing up a real web server.
"""
from __future__ import annotations

from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import style
from scrollable import ScrollableFrame


class Core5Tab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        # Scrollable body so the recipe list + scatter plot remain reachable
        # if the window is narrower/shorter than the two-column layout wants.
        scrollable = ScrollableFrame(self)
        scrollable.pack(fill="both", expand=True)
        self._build(scrollable.body)

    def _build(self, parent):
        parent.configure(padding=12)
        style.section_title(
            parent,
            "Core Function #5: Dashboard (local stand-in for the Web UI / API service layer)",
            "The original design calls for a React.js + FastAPI + AWS web layer; this tab is the same "
            "visualization dashboard sub-function (recipe list + scatter plot) running locally.",
        ).pack(fill="x", pady=(0, 12))

        ttk.Button(parent, text="Refresh from Core 1's Pareto front", style="Primary.TButton",
                   command=self.on_refresh).pack(anchor="w", pady=4)

        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True, pady=6)

        # Left half: a plain table of recommended recipes, ranked by strength.
        list_frame = ttk.LabelFrame(body, text="Recommended Recipe List", padding=8)
        list_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))
        columns = ("rank", "strength", "cost", "carbon", "method")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=14)
        for col, w in zip(columns, (50, 80, 80, 80, 90)):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=4, pady=4)

        # Right half: the same data as a scatter plot, so trade-offs across
        # the Pareto front are visible at a glance rather than row-by-row.
        plot_frame = ttk.LabelFrame(body, text="Cost vs. Strength (colored by carbon emission)", padding=8)
        plot_frame.pack(side="left", fill="both", expand=True)
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self._draw_empty()

    def _draw_empty(self):
        self.ax.clear()
        self.ax.set_xlabel("Cost")
        self.ax.set_ylabel("Compressive strength")
        self.ax.set_title("Run Core 1's Pareto front, then refresh")
        self.canvas.draw()

    def on_refresh(self):
        """Synchronous by design (no run_async) -- this just reads data
        Core 1 already computed (app.core1_tab._pareto_rows) and redraws;
        there's no slow work here, so a background thread would be overkill."""
        front = getattr(self.app.core1_tab, "_pareto_rows", None)
        if not front:
            self.app.log("No Pareto front yet -- compute it in the Core 1 tab first.")
            return

        self.tree.delete(*self.tree.get_children())
        for i, r in enumerate(front[:30], 1):
            self.tree.insert("", "end", values=(i, f"{r['compressive_strength']:.2f}", f"{r['cost']:.2f}",
                                                  f"{r['carbon_emission']:.2f}", r["curing_method"]))

        self.ax.clear()
        costs = [r["cost"] for r in front]
        strengths = [r["compressive_strength"] for r in front]
        carbons = [r["carbon_emission"] for r in front]
        scatter = self.ax.scatter(costs, strengths, c=carbons, cmap="viridis")
        self.ax.set_xlabel("Cost")
        self.ax.set_ylabel("Compressive strength")
        self.ax.set_title(f"{len(front)} Pareto-optimal designs")
        self.figure.colorbar(scatter, ax=self.ax, label="Carbon emission")
        self.canvas.draw()
        self.app.log(f"Dashboard refreshed with {len(front)} Pareto-optimal designs.")
