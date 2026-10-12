"""GUI tab wrapping Core Function #2 -- Data Integration & Standardization."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import paths
import style
from scrollable import ScrollableFrame
import export_to_csv
import generate_simulated_data
import ingest_simulated_sources as ingest_mod
import schema as c2_schema


class Core2Tab(ttk.Frame):
    """
    Wraps core2_data_integration/*.py. This tab is meant to be run first --
    it produces materials_wide.csv, which Core 1 and Core 4 both read.

    Each button below maps to one function from the real core2 modules;
    nothing here re-implements any logic, it just calls into schema.py,
    generate_simulated_data.py, ingest_simulated_sources.py, etc. and shows
    the result. See the threading-model note in app.py before editing any
    on_*() handler below -- Tk variables must be read on the main thread.
    """

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        # All content is built onto scrollable.body rather than directly on
        # this frame, so the tab gets a vertical scrollbar once its content
        # grows taller than the window (more Pareto rows, a smaller window,
        # etc.) instead of silently clipping anything below the fold.
        scrollable = ScrollableFrame(self)
        scrollable.pack(fill="both", expand=True)
        self._build(scrollable.body)

    def _build(self, parent):
        parent.configure(padding=12)
        style.section_title(
            parent,
            "Core Function #2: Data Integration & Standardization",
            "Generates/ingests experimental, literature, and process-log data into one unified database.",
        ).pack(fill="x", pady=(0, 12))

        # --- Step buttons: run in order, top to bottom, left to right -----
        step_frame = ttk.LabelFrame(parent, text="Pipeline steps (run in order)", padding=10)
        step_frame.pack(fill="x", pady=4, padx=12)

        ttk.Button(step_frame, text="1. Generate Simulated Raw Data",
                   command=self.on_generate_data).grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(step_frame, text="2. Ingest && Unify Sources",
                   command=self.on_ingest).grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(step_frame, text="3. Export to CSV",
                   command=self.on_export).grid(row=0, column=2, padx=4, pady=4, sticky="ew")
        ttk.Button(step_frame, text="4. Train GAN && Augment", style="Primary.TButton",
                   command=self.on_augment).grid(row=0, column=3, padx=4, pady=4, sticky="ew")
        for i in range(4):
            step_frame.columnconfigure(i, weight=1)

        # --- Results table: filled in by on_ingest() ------------------------
        ttk.Label(parent, text="Ingestion quality report:", font=style.SECTION_FONT).pack(
            anchor="w", pady=(14, 4), padx=12
        )
        columns = ("source", "rows", "written", "completeness", "missing_rate", "outliers")
        self.tree = ttk.Treeview(parent, columns=columns, show="headings", height=6)
        for col, width in zip(columns, (110, 60, 70, 100, 100, 70)):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=width, anchor="center")
        self.tree.pack(fill="x", pady=4, padx=12)

        self.augment_summary_var = tk.StringVar(value="Run step 4 to see GAN augmentation quality.")
        ttk.Label(parent, textvariable=self.augment_summary_var, wraplength=1000, justify="left",
                  foreground=style.MUTED_TEXT).pack(anchor="w", pady=(10, 12), padx=12)

    # -------------------------------------------------------------- steps
    def on_generate_data(self):
        """Step 1: writes 3 simulated raw files (csv/json/xlsx) with
        deliberately different column names/units, to exercise the ETL
        harmonization logic in generate_simulated_data.py."""
        def work():
            # This runs on a background thread -- pure file I/O + numpy,
            # no Tkinter involved, so it's safe here (unlike the button
            # handlers, which must read Tk variables before this point).
            generate_simulated_data.main()
            return True

        self.app.run_async(
            work,
            on_done=lambda _: self.app.log(f"Simulated raw data written to {paths.CORE2_SAMPLE_DATA}"),
            busy_message="Generating simulated raw data (experimental / literature / process-log)...",
        )

    def on_ingest(self):
        """Step 2: loads the 3 raw sources, renames columns/converts units to
        canonical form, cleans missing values & outliers, and writes
        everything into one SQLite database (schema.py's Material /
        MaterialProperty tables)."""
        def work():
            # ingest_simulated_sources.py's loader functions reference a
            # module-level DATA_DIR constant rather than taking a path
            # argument; we point it at our absolute sample_data folder here
            # so this works no matter what directory the GUI was launched
            # from.
            ingest_mod.DATA_DIR = paths.CORE2_SAMPLE_DATA
            engine = c2_schema.init_db(c2_schema.get_engine(paths.CORE2_DB_URL))
            session = c2_schema.get_session(engine)

            sources = {
                "experimental": ingest_mod.load_experimental(),
                "literature": ingest_mod.load_literature(),
                "process_log": ingest_mod.load_process_log(),
            }
            rows = []
            for source_type, raw_df in sources.items():
                quality = ingest_mod.run_pipeline_for_source(raw_df, source_type, session)
                rows.append((source_type, len(raw_df), quality["materials_written"],
                             quality["data_completeness"], quality["missing_value_rate"], quality["outliers_detected"]))
            session.close()
            return rows

        def done(rows):
            # Runs on the main thread (via the task queue), so touching the
            # Treeview here is safe.
            self.tree.delete(*self.tree.get_children())
            for r in rows:
                self.tree.insert("", "end", values=(r[0], r[1], r[2], f"{r[3]:.3f}", f"{r[4]:.3f}", r[5]))
            self.app.log(f"Ingested {sum(r[1] for r in rows)} raw rows across 3 sources into {paths.CORE2_DB_FILE}")

        self.app.run_async(work, on_done=done, busy_message="Ingesting and unifying experimental/literature/process-log data...")

    def on_export(self):
        """Step 3: dumps the SQLite database to CSV (materials.csv,
        material_properties.csv, materials_wide.csv) -- materials_wide.csv
        is what Core 1 and Core 4 both consume."""
        def work():
            export_to_csv.export(paths.CORE2_DB_FILE, paths.CORE2_SAMPLE_DATA)
            return True

        self.app.run_async(
            work,
            on_done=lambda _: self.app.log(f"Exported materials.csv, material_properties.csv, materials_wide.csv to {paths.CORE2_SAMPLE_DATA}"),
            on_error=lambda e: self.app.log(f"Export failed: {e}. Did you run step 2 (Ingest) first?"),
            busy_message="Exporting database to CSV...",
        )

    def on_augment(self):
        """Step 4: trains the lightweight NumPy GAN (augmentation.py) on the
        real property data and reports how close the synthetic samples'
        distribution is to the real one (mean KS statistic -- lower is
        better)."""
        def work():
            import pandas as pd
            import augmentation as aug
            from schema import FIVE_MAJOR_PROPERTIES, MaterialProperty, get_engine, get_session

            engine = get_engine(paths.CORE2_DB_URL)
            session = get_session(engine)
            rows = (
                session.query(MaterialProperty)
                .filter(MaterialProperty.property_name.in_(FIVE_MAJOR_PROPERTIES))
                .filter(MaterialProperty.is_synthetic.is_(False))
                .all()
            )
            session.close()
            df = pd.DataFrame([{"material_id": r.material_id, "property_name": r.property_name, "value": r.property_value} for r in rows])
            if df.empty:
                raise RuntimeError("No real data available yet -- run steps 1 & 2 first.")
            # Pivot from long (one row per property) to wide (one row per
            # material) so each row is a complete feature vector for the GAN.
            wide = df.pivot_table(index="material_id", columns="property_name", values="value").dropna()
            if len(wide) < 10:
                raise RuntimeError(f"Need >=10 complete records to train the GAN, found {len(wide)}.")
            real = wide[FIVE_MAJOR_PROPERTIES].to_numpy()
            synthetic = aug.generate_synthetic(real, n_samples=30, epochs=500)
            similarity = aug.distribution_similarity(real, synthetic, feature_names=FIVE_MAJOR_PROPERTIES)
            return len(real), similarity

        def done(result):
            n_real, similarity = result
            self.augment_summary_var.set(
                f"Trained GAN on {n_real} real records; generated 30 synthetic samples. "
                f"Mean KS statistic: {similarity['mean_ks_statistic']:.3f} (lower = closer to real distribution)."
            )
            self.app.log("GAN augmentation complete.")

        self.app.run_async(work, on_done=done, busy_message="Training lightweight GAN for data augmentation...")
