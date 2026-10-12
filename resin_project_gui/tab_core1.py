"""GUI tab wrapping Core Function #1 -- AI Reverse Design Core."""
from __future__ import annotations

import os

import tkinter as tk
from tkinter import ttk

import paths
import style
from scrollable import ScrollableFrame
from add_cost_data import synthesize_cost
from data_prep import load_dataset
from forward_model import ForwardDNN
from pareto_optimization import pareto_front
from reverse_design import reverse_design


class Core1Tab(ttk.Frame):
    """
    Wraps core1_reverse_design/*.py: trains the forward property-prediction
    DNN, then uses it two ways -- reverse design (invert it to hit one target
    value) and Pareto optimization (sample many candidates and keep the
    non-dominated set across strength/cost/carbon).

    Run order: Add Cost Data -> Train Forward DNN -> then either Reverse
    Design or Pareto Front (both need a trained model in self.model).
    """

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        # These three are set once training finishes (see on_train's done()
        # callback) and read by every other button in this tab.
        self.model = None
        self.additive_vocab = None
        self.target_names = None
        # Built onto a scrollable body (see scrollable.py) so this tab gets a
        # vertical scrollbar once its content -- setup, reverse design,
        # Pareto table -- grows taller than the window.
        scrollable = ScrollableFrame(self)
        scrollable.pack(fill="both", expand=True)
        self._build(scrollable.body)

    def _build(self, parent):
        parent.configure(padding=12)
        style.section_title(
            parent,
            "Core Function #1: AI Reverse Design Core",
            "Trains a forward property-prediction model, then inverts it to design recipes.",
        ).pack(fill="x", pady=(0, 12))

        # --- Setup: cost data + training ------------------------------------
        setup_frame = ttk.LabelFrame(parent, text="Setup", padding=10)
        setup_frame.pack(fill="x", pady=4)
        ttk.Button(setup_frame, text="1. Add Cost Data (from Core 2)", command=self.on_add_cost).grid(
            row=0, column=0, padx=4, pady=4, sticky="ew"
        )
        ttk.Label(setup_frame, text="Architecture:").grid(row=0, column=1, padx=(16, 4))
        self.arch_var = tk.StringVar(value="large")
        ttk.Combobox(setup_frame, textvariable=self.arch_var, values=["basic", "large", "wide"],
                     width=8, state="readonly").grid(row=0, column=2, padx=4)
        ttk.Button(setup_frame, text="2. Train Forward DNN", style="Primary.TButton",
                   command=self.on_train).grid(row=0, column=3, padx=4, pady=4, sticky="ew")
        for i in range(4):
            setup_frame.columnconfigure(i, weight=1)

        self.metrics_var = tk.StringVar(value="Train the model to see R2 / RMSE / MAE.")
        ttk.Label(parent, textvariable=self.metrics_var, wraplength=1050, justify="left",
                  foreground=style.MUTED_TEXT).pack(anchor="w", pady=(8, 12))

        # --- Reverse design: invert the trained model for one target ------
        rd_frame = ttk.LabelFrame(parent, text="Reverse Design: hit a target property value", padding=10)
        rd_frame.pack(fill="x", pady=6)
        ttk.Label(rd_frame, text="Target property:").grid(row=0, column=0, padx=4, pady=6)
        self.target_prop_var = tk.StringVar(value="compressive_strength")
        # Populated with real target names once training finishes (see on_train).
        self.target_prop_combo = ttk.Combobox(rd_frame, textvariable=self.target_prop_var, width=20, state="readonly")
        self.target_prop_combo.grid(row=0, column=1, padx=4)
        ttk.Label(rd_frame, text="Target value:").grid(row=0, column=2, padx=4)
        self.target_value_var = tk.StringVar(value="30.0")
        ttk.Entry(rd_frame, textvariable=self.target_value_var, width=10).grid(row=0, column=3, padx=4)
        ttk.Button(rd_frame, text="Run Reverse Design", style="Primary.TButton",
                   command=self.on_reverse_design).grid(row=0, column=4, padx=8)

        self.rd_result_var = tk.StringVar(value="")
        ttk.Label(rd_frame, textvariable=self.rd_result_var, wraplength=1050, justify="left").grid(
            row=1, column=0, columnspan=5, sticky="w", padx=4, pady=(4, 8)
        )

        # --- Pareto front: strength vs. cost vs. carbon emission ----------
        pf_frame = ttk.LabelFrame(parent, text="Multi-objective Pareto front: strength vs. cost vs. carbon emission", padding=10)
        pf_frame.pack(fill="both", expand=True, pady=6)
        ttk.Button(pf_frame, text="Compute Pareto Front (1500 samples)", style="Primary.TButton",
                   command=self.on_pareto).pack(anchor="w", pady=4)
        columns = ("strength", "cost", "carbon", "method", "additives", "voc", "ratio", "curing")
        self.pareto_tree = ttk.Treeview(pf_frame, columns=columns, show="headings", height=8)
        widths = (70, 70, 70, 80, 160, 60, 60, 60)
        for col, w in zip(columns, widths):
            self.pareto_tree.heading(col, text=col)
            self.pareto_tree.column(col, width=w, anchor="center")
        self.pareto_tree.pack(fill="both", expand=True, pady=4)
        ttk.Button(pf_frame, text="Send selected recipe to Core 3 (Validation Protocol)",
                   command=self.on_send_to_core3).pack(anchor="w", pady=(4, 0))

        # Keeps the full dict for each row shown in pareto_tree, since the
        # Treeview only stores display strings -- on_send_to_core3 needs the
        # original numeric values, not their formatted text.
        self._pareto_rows: list[dict] = []

    # -------------------------------------------------------------- steps
    def on_add_cost(self):
        """Reads Core 2's materials_wide.csv and adds a synthesized 'cost'
        column (add_cost_data.py) -- Core 2 doesn't track cost, but Core 1's
        Pareto optimization needs it as one of its three objectives."""
        def work():
            import pandas as pd

            if not os.path.exists(paths.MATERIALS_WIDE_CSV):
                raise RuntimeError("materials_wide.csv not found -- run Core 2 steps 1-3 first.")
            df = pd.read_csv(paths.MATERIALS_WIDE_CSV)
            df["cost"] = synthesize_cost(df)
            df.to_csv(paths.MATERIALS_WIDE_WITH_COST_CSV, index=False)
            return len(df)

        self.app.run_async(
            work,
            on_done=lambda n: self.app.log(f"Synthesized cost column for {n} materials -> {paths.MATERIALS_WIDE_WITH_COST_CSV}"),
            busy_message="Synthesizing cost data...",
        )

    def on_train(self):
        """Trains the forward DNN (ForwardDNN in forward_model.py) on the
        cost-augmented data. Cross-validates its L2 regularization internally
        -- see forward_model.py's docstring for why that matters."""
        architecture = self.arch_var.get()  # read the Tk variable now, on the main thread

        def work():
            if not os.path.exists(paths.MATERIALS_WIDE_WITH_COST_CSV):
                raise RuntimeError("Run step 1 (Add Cost Data) first.")
            X, Y, feature_names, target_names, additive_vocab, methods = load_dataset(paths.MATERIALS_WIDE_WITH_COST_CSV)
            model = ForwardDNN(architecture=architecture)
            metrics = model.fit(X, Y)
            return model, additive_vocab, target_names, metrics

        def done(result):
            model, additive_vocab, target_names, metrics = result
            # Store on self so Reverse Design and Pareto Front (below) can
            # use the trained model without retraining.
            self.model, self.additive_vocab, self.target_names = model, additive_vocab, target_names
            self.target_prop_combo["values"] = target_names
            per_target = "  ".join(f"{n}: R2={metrics['per_target_r2'][i]:.3f}" for i, n in enumerate(target_names))
            self.metrics_var.set(
                f"Overall R2={metrics['r2']:.4f}  RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  "
                f"(alpha={metrics['selected_alpha']}, n_train={metrics['n_train']}, n_test={metrics['n_test']})\n{per_target}"
            )
            self.app.log(f"Forward DNN trained. Overall R2={metrics['r2']:.4f}")

        self.app.run_async(work, on_done=done, busy_message="Training forward DNN (cross-validating regularization)...")

    def on_reverse_design(self):
        """Inverts the trained model with L-BFGS-B (reverse_design.py) to
        find the recipe that best hits one target property value. Stores the
        result in app.shared_state so the Core 3 tab can load it directly."""
        if self.model is None:
            self.app.log("Train the model first (step 2).")
            return
        # Read both Tk variables now -- work() runs on a background thread
        # and must never touch them directly (see app.py's threading note).
        target_prop = self.target_prop_var.get()
        try:
            target_value = float(self.target_value_var.get())
        except ValueError:
            self.app.log("Target value must be a number.")
            return

        def work():
            target = {target_prop: target_value}
            return reverse_design(self.model, target, self.target_names, paths.CONTINUOUS_BOUNDS, self.additive_vocab)

        def done(result):
            cv = result["continuous_values"]
            recipe = {
                "resin_voc_ratio": cv["resin_voc_ratio"],
                "sio2_na2o_ratio": cv["sio2_na2o_ratio"],
                "curing_time": cv["curing_time"],
                "curing_method": result["curing_method"],
                "additives": result["additives"],
            }
            self.app.shared_state["last_recipe"] = recipe
            self.app.shared_state["last_predicted_properties"] = result["predicted_properties"]
            self.rd_result_var.set(
                f"Recipe: resin_voc_ratio={cv['resin_voc_ratio']:.3f}, sio2_na2o_ratio={cv['sio2_na2o_ratio']:.3f}, "
                f"curing_time={cv['curing_time']:.2f}hr, method={result['curing_method']}, additives={result['additives']}\n"
                f"Predicted: {result['predicted_properties']}\n"
                f"Response time: {result['response_time_s']*1000:.0f} ms"
            )
            self.app.log(f"Reverse design complete in {result['response_time_s']*1000:.0f} ms. Recipe stored for Core 3.")

        self.app.run_async(work, on_done=done, busy_message="Running L-BFGS-B reverse design search...")

    def on_pareto(self):
        """Samples 1500 random designs, predicts their properties with the
        trained model, and keeps only the non-dominated (Pareto-optimal)
        subset -- see pareto_optimization.py."""
        if self.model is None:
            self.app.log("Train the model first (step 2).")
            return

        def work():
            return pareto_front(self.model, self.target_names, paths.CONTINUOUS_BOUNDS, self.additive_vocab, n_samples=1500)

        def done(front):
            self._pareto_rows = front  # keep the raw dicts for on_send_to_core3
            self.pareto_tree.delete(*self.pareto_tree.get_children())
            for r in front[:30]:
                self.pareto_tree.insert(
                    "", "end",
                    values=(f"{r['compressive_strength']:.2f}", f"{r['cost']:.2f}", f"{r['carbon_emission']:.2f}",
                            r["curing_method"], ", ".join(r["additives"]) or "(none)",
                            f"{r['resin_voc_ratio']:.3f}", f"{r['sio2_na2o_ratio']:.2f}", f"{r['curing_time']:.1f}"),
                )
            self.app.log(f"Pareto front computed: {len(front)} non-dominated designs found.")

        self.app.run_async(work, on_done=done, busy_message="Sampling design space and computing Pareto front...")

    def on_send_to_core3(self):
        """Synchronous (no run_async needed) -- just copies the selected
        Pareto row into app.shared_state for the Core 3 tab to pick up."""
        selection = self.pareto_tree.selection()
        if not selection:
            self.app.log("Select a row in the Pareto front table first.")
            return
        idx = self.pareto_tree.index(selection[0])
        r = self._pareto_rows[idx]
        recipe = {
            "resin_voc_ratio": r["resin_voc_ratio"],
            "sio2_na2o_ratio": r["sio2_na2o_ratio"],
            "curing_time": r["curing_time"],
            "curing_method": r["curing_method"],
            "additives": r["additives"],
        }
        self.app.shared_state["last_recipe"] = recipe
        self.app.shared_state["last_predicted_properties"] = {
            "compressive_strength": r["compressive_strength"],
            "cost": r["cost"],
            "carbon_emission": r["carbon_emission"],
        }
        self.app.log("Sent selected Pareto recipe to Core 3 (Validation Protocol tab).")
