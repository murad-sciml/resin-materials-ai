"""GUI tab wrapping Core Function #4 -- Software-Based Autonomous Experiment Module."""
from __future__ import annotations

import warnings

import tkinter as tk
from tkinter import ttk

import paths
import style
from scrollable import ScrollableFrame
from bootstrap_from_core2 import DESIGN_VARS, TARGET_CHOICES, bootstrap_pipeline, load_real_observations


class Core4Tab(ttk.Frame):
    """
    Wraps core4_autonomous_experiment/*.py. Bootstraps a Bayesian-Optimization
    surrogate model directly from Core 2's real data (bootstrap_from_core2.py),
    then supports the same recommend -> test -> feedback -> retrain loop as
    that module's example_usage.py, driven from the GUI instead of a script.
    """

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.pipeline = None  # set once Bootstrap succeeds; a RetrainingPipeline instance
        # Scrollable body so the DOE table + feedback form stay reachable
        # even with a full-width design-variable row of entry fields.
        scrollable = ScrollableFrame(self)
        scrollable.pack(fill="both", expand=True)
        self._build(scrollable.body)

    def _build(self, parent):
        parent.configure(padding=12)
        style.section_title(
            parent,
            "Core Function #4: Software-Based Autonomous Experiment Module",
            "Bootstraps a Bayesian Optimizer from Core 2's real data, then recommends the next experiments to run.",
        ).pack(fill="x", pady=(0, 12))

        # --- Bootstrap: fit the surrogate on real historical data ----------
        setup_frame = ttk.LabelFrame(parent, text="Bootstrap", padding=10)
        setup_frame.pack(fill="x", pady=4)
        ttk.Label(setup_frame, text="Optimize target:").grid(row=0, column=0, padx=4)
        self.target_var = tk.StringVar(value="compressive_strength")
        ttk.Combobox(setup_frame, textvariable=self.target_var, values=TARGET_CHOICES, width=20,
                     state="readonly").grid(row=0, column=1, padx=4)
        self.minimize_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(setup_frame, text="minimize (default: maximize)", variable=self.minimize_var).grid(
            row=0, column=2, padx=8
        )
        ttk.Button(setup_frame, text="Bootstrap from Core 2 Data", style="Primary.TButton",
                   command=self.on_bootstrap).grid(row=0, column=3, padx=8)

        self.status_var = tk.StringVar(value="Bootstrap the optimizer to begin.")
        ttk.Label(parent, textvariable=self.status_var, wraplength=1050, justify="left",
                  foreground=style.MUTED_TEXT).pack(anchor="w", pady=(8, 12))

        # --- next DOE: exploitation (Bayesian Opt.) + exploration (uncertainty) --
        doe_frame = ttk.LabelFrame(parent, text="Next-priority Design of Experiments (DOE)", padding=10)
        doe_frame.pack(fill="both", expand=True, pady=6)
        controls = ttk.Frame(doe_frame)
        controls.pack(fill="x", pady=4)
        ttk.Label(controls, text="n_bo (exploit):").pack(side="left")
        self.n_bo_var = tk.StringVar(value="3")
        ttk.Entry(controls, textvariable=self.n_bo_var, width=5).pack(side="left", padx=(2, 10))
        ttk.Label(controls, text="n_uncertain (explore):").pack(side="left")
        self.n_uncertain_var = tk.StringVar(value="2")
        ttk.Entry(controls, textvariable=self.n_uncertain_var, width=5).pack(side="left", padx=(2, 10))
        ttk.Button(controls, text="Get Next DOE", style="Primary.TButton",
                   command=self.on_next_doe).pack(side="left", padx=10)

        columns = ("rank",) + tuple(DESIGN_VARS) + ("source",)
        self.doe_tree = ttk.Treeview(doe_frame, columns=columns, show="headings", height=6)
        for col, w in zip(columns, (50, 130, 130, 100, 150)):
            self.doe_tree.heading(col, text=col)
            self.doe_tree.column(col, width=w, anchor="center")
        self.doe_tree.pack(fill="both", expand=True, pady=4)

        # --- feedback: retrains the surrogate with a real measured result --
        fb_frame = ttk.LabelFrame(parent, text="Submit experimental feedback (retrains the surrogate model)", padding=10)
        fb_frame.pack(fill="x", pady=6)
        self.fb_vars = {}
        for i, name in enumerate(DESIGN_VARS):
            ttk.Label(fb_frame, text=f"{name}:").grid(row=0, column=i * 2, padx=4, pady=4, sticky="e")
            var = tk.StringVar(value="")
            ttk.Entry(fb_frame, textvariable=var, width=8).grid(row=0, column=i * 2 + 1, padx=4)
            self.fb_vars[name] = var
        ttk.Label(fb_frame, text="measured_value:").grid(row=0, column=len(DESIGN_VARS) * 2, padx=4, sticky="e")
        self.fb_measured_var = tk.StringVar(value="")
        ttk.Entry(fb_frame, textvariable=self.fb_measured_var, width=8).grid(
            row=0, column=len(DESIGN_VARS) * 2 + 1, padx=4
        )
        ttk.Button(fb_frame, text="Submit Feedback && Retrain", style="Primary.TButton",
                   command=self.on_submit_feedback).grid(row=0, column=len(DESIGN_VARS) * 2 + 2, padx=8)
        self.fb_result_var = tk.StringVar(value="")
        ttk.Label(fb_frame, textvariable=self.fb_result_var, wraplength=1050, justify="left").grid(
            row=1, column=0, columnspan=len(DESIGN_VARS) * 2 + 3, sticky="w", padx=4, pady=(4, 6)
        )

    # -------------------------------------------------------------- steps
    def on_bootstrap(self):
        """Fits the Gaussian Process surrogate on ALL of Core 2's real
        materials_wide.csv rows in one batched call -- see
        bootstrap_from_core2.py's comment on why this must be a single
        fit rather than one-row-at-a-time (the latter used to make the GUI
        hang once the dataset grew past ~200 rows)."""
        target = self.target_var.get()
        minimize = self.minimize_var.get()

        def work():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # sklearn's GP hyperparameter convergence warnings are expected & harmless here
                df = load_real_observations(paths.MATERIALS_WIDE_CSV, target)
                pipeline = bootstrap_pipeline(df, target, minimize=minimize)
            return pipeline, len(df)

        def done(result):
            pipeline, n = result
            self.pipeline = pipeline
            metrics = pipeline.convergence_metrics()[-1]
            self.status_var.set(
                f"Bootstrapped with {n} real observations targeting "
                f"{'min' if minimize else 'max'} {target}. "
                f"Surrogate fit: R2={metrics['r2']:.4f}  RMSE={metrics['rmse']:.4f}"
            )
            self.app.log(f"Core 4 bootstrapped from real data: R2={metrics['r2']:.4f}")

        self.app.run_async(work, on_done=done, busy_message="Bootstrapping Bayesian Optimizer from real Core 2 data...")

    def on_next_doe(self):
        """Blends Bayesian-Optimization exploitation with uncertainty-sampling
        exploration -- see retrain_pipeline.py's next_doe()."""
        if self.pipeline is None:
            self.app.log("Bootstrap the optimizer first.")
            return
        try:
            n_bo = int(self.n_bo_var.get())
            n_uncertain = int(self.n_uncertain_var.get())
        except ValueError:
            self.app.log("n_bo / n_uncertain must be integers.")
            return

        def work():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self.pipeline.next_doe(n_bo=n_bo, n_uncertain=n_uncertain)

        def done(result):
            self.doe_tree.delete(*self.doe_tree.get_children())
            for c in result["doe"]:
                vals = [c["rank"]] + [f"{c['design_variables'][n]:.3f}" for n in DESIGN_VARS] + [c["source"]]
                self.doe_tree.insert("", "end", values=vals)
            self.app.log(f"Next DOE computed ({len(result['doe'])} candidates).")

        self.app.run_async(work, on_done=done, busy_message="Computing next-priority DOE...")

    def on_submit_feedback(self):
        """Feeds one real experimental result back into the pipeline, which
        retrains the surrogate immediately (closed-loop optimization)."""
        if self.pipeline is None:
            self.app.log("Bootstrap the optimizer first.")
            return
        try:
            x = [float(self.fb_vars[n].get()) for n in DESIGN_VARS]
            y = float(self.fb_measured_var.get())
        except ValueError:
            self.app.log("Design variables and measured value must be numbers.")
            return

        def work():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self.pipeline.submit_feedback(x, y)

        def done(metrics):
            self.fb_result_var.set(
                f"Retrained. n_observations={metrics['n_observations']}  R2={metrics['r2']}  RMSE={metrics['rmse']}"
            )
            self.app.log("Core 4 surrogate retrained with new feedback.")

        self.app.run_async(work, on_done=done, busy_message="Submitting feedback and retraining...")
