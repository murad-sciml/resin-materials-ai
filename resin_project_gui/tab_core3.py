"""GUI tab wrapping Core Function #3 -- Validation Protocol Engine."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, scrolledtext

import paths
import style
from scrollable import ScrollableFrame
from feedback import FeedbackRecord, reproducibility_score, submit_feedback
from protocol_generator import DEFAULT_TARGET_PROPERTIES, generate_protocol
from standards_mapper import map_all, mapping_accuracy


class Core3Tab(ttk.Frame):
    """
    Wraps core3_validation_protocol/*.py. Consumes a recipe from Core 1 (via
    app.shared_state["last_recipe"]) or a manually-entered one, generates a
    synthesis + testing protocol, maps evaluation items to standards, and
    writes real experimental feedback back into Core 2's database.
    """

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.current_protocol = None  # set once Generate Protocol succeeds
        # Scrollable body so the protocol text box + all three sections
        # below it remain reachable even on a smaller window.
        scrollable = ScrollableFrame(self)
        scrollable.pack(fill="both", expand=True)
        self._build(scrollable.body)

    def _build(self, parent):
        parent.configure(padding=12)
        style.section_title(
            parent,
            "Core Function #3: Validation Protocol Engine",
            "Generates a synthesis + testing protocol for a recipe (from Core 1 or entered manually), "
            "maps evaluation items to standards, and records real experimental feedback.",
        ).pack(fill="x", pady=(0, 12))

        # --- Recipe under test: loaded from Core 1, or entered manually ---
        recipe_frame = ttk.LabelFrame(parent, text="Recipe under test", padding=10)
        recipe_frame.pack(fill="x", pady=4)
        ttk.Button(recipe_frame, text="Load recipe from Core 1", command=self.on_load_from_core1).grid(
            row=0, column=0, padx=4, pady=4, sticky="w"
        )
        self.recipe_var = tk.StringVar(value="(none loaded -- run Core 1 or fill in manually below)")
        ttk.Label(recipe_frame, textvariable=self.recipe_var, wraplength=1000, justify="left",
                  foreground=style.MUTED_TEXT).grid(row=0, column=1, columnspan=4, padx=4, sticky="w")

        fields = ["resin_voc_ratio", "sio2_na2o_ratio", "curing_time"]
        self.manual_vars = {f: tk.StringVar(value=v) for f, v in zip(fields, ["0.15", "2.5", "20"])}
        for i, f in enumerate(fields):
            ttk.Label(recipe_frame, text=f + ":").grid(row=1, column=i * 2, padx=4, pady=4, sticky="e")
            ttk.Entry(recipe_frame, textvariable=self.manual_vars[f], width=8).grid(row=1, column=i * 2 + 1, padx=4)
        ttk.Label(recipe_frame, text="curing_method:").grid(row=2, column=0, padx=4, sticky="e")
        self.method_var = tk.StringVar(value="steam")
        ttk.Combobox(recipe_frame, textvariable=self.method_var, values=["ambient", "steam", "autoclave"],
                     width=10, state="readonly").grid(row=2, column=1, padx=4, sticky="w")
        ttk.Label(recipe_frame, text="additives (comma-sep):").grid(row=2, column=2, padx=4, sticky="e")
        self.additives_var = tk.StringVar(value="")
        ttk.Entry(recipe_frame, textvariable=self.additives_var, width=30).grid(row=2, column=3, columnspan=2, padx=4, sticky="w")
        ttk.Button(recipe_frame, text="Use manual recipe instead", command=self.on_use_manual).grid(
            row=2, column=5, padx=8
        )

        ttk.Button(parent, text="Generate Protocol", style="Primary.TButton",
                   command=self.on_generate_protocol).pack(anchor="w", pady=(10, 4))
        # Monospace + dark-ish background makes the generated protocol read
        # like a document rather than plain form text.
        self.protocol_text = scrolledtext.ScrolledText(parent, height=14, font=("Consolas", 9),
                                                        bg="#fbfbfb", relief="flat", borderwidth=1)
        self.protocol_text.pack(fill="both", expand=True, pady=4)

        # --- standards mapping ---------------------------------------------
        map_frame = ttk.LabelFrame(parent, text="Standard-to-property mapping (try loosely-named inputs)", padding=10)
        map_frame.pack(fill="x", pady=6)
        self.loose_names_var = tk.StringVar(value="Compressive Strength, flexural strength, VISCOSITY, made_up_property")
        ttk.Entry(map_frame, textvariable=self.loose_names_var, width=70).grid(row=0, column=0, padx=4, pady=4)
        ttk.Button(map_frame, text="Map to Standards", command=self.on_map_standards).grid(row=0, column=1, padx=4)
        self.mapping_result_var = tk.StringVar(value="")
        ttk.Label(map_frame, textvariable=self.mapping_result_var, wraplength=1000, justify="left").grid(
            row=1, column=0, columnspan=2, sticky="w", padx=4, pady=(0, 6)
        )

        # --- feedback: writes into Core 2's database ------------------------
        fb_frame = ttk.LabelFrame(parent, text="Submit experimental feedback (writes into Core 2's database)", padding=10)
        fb_frame.pack(fill="x", pady=6)
        self.measured_vars = {}
        for i, prop in enumerate(DEFAULT_TARGET_PROPERTIES):
            ttk.Label(fb_frame, text=f"measured {prop}:").grid(row=0, column=i * 2, padx=4, pady=4, sticky="e")
            var = tk.StringVar(value="")
            ttk.Entry(fb_frame, textvariable=var, width=8).grid(row=0, column=i * 2 + 1, padx=4)
            self.measured_vars[prop] = var
        ttk.Label(fb_frame, text="expert rating (0-1):").grid(row=1, column=0, padx=4, sticky="e")
        self.expert_rating_var = tk.StringVar(value="0.9")
        ttk.Entry(fb_frame, textvariable=self.expert_rating_var, width=8).grid(row=1, column=1, padx=4, sticky="w")
        ttk.Button(fb_frame, text="Submit Feedback", style="Primary.TButton",
                   command=self.on_submit_feedback).grid(row=1, column=2, padx=8)
        self.feedback_result_var = tk.StringVar(value="")
        ttk.Label(fb_frame, textvariable=self.feedback_result_var, wraplength=1000, justify="left").grid(
            row=2, column=0, columnspan=6, sticky="w", padx=4, pady=(4, 6)
        )

    # ------------------------------------------------------------- recipe
    def _current_recipe(self) -> dict:
        """Returns whichever recipe should currently drive protocol
        generation: Core 1's handoff if present, otherwise the manual entry
        fields. MUST be called on the main thread (it reads Tk variables) --
        callers should capture its result before starting a background task."""
        recipe = self.app.shared_state.get("last_recipe")
        if recipe:
            return recipe
        additives = [a.strip() for a in self.additives_var.get().split(",") if a.strip()]
        return {
            "resin_voc_ratio": float(self.manual_vars["resin_voc_ratio"].get()),
            "sio2_na2o_ratio": float(self.manual_vars["sio2_na2o_ratio"].get()),
            "curing_time": float(self.manual_vars["curing_time"].get()),
            "curing_method": self.method_var.get(),
            "additives": additives,
        }

    def on_load_from_core1(self):
        recipe = self.app.shared_state.get("last_recipe")
        if not recipe:
            self.app.log("No recipe from Core 1 yet -- run reverse design or Pareto front there first.")
            return
        self.recipe_var.set(f"Loaded from Core 1: {recipe}")
        self.app.log("Loaded recipe from Core 1 into Core 3.")

    def on_use_manual(self):
        """Clears the Core 1 handoff so _current_recipe() falls back to the
        manual entry fields instead."""
        self.app.shared_state.pop("last_recipe", None)
        self.app.shared_state.pop("last_predicted_properties", None)
        self.recipe_var.set(f"Using manual recipe: {self._current_recipe()}")
        self.app.log("Switched Core 3 to the manually-entered recipe.")

    # -------------------------------------------------------------- steps
    def on_generate_protocol(self):
        """Rule-based generation (protocol_generator.py): synthesis steps +
        one evaluation step per target property, each mapped to a standard
        test method from standards_kb.py."""
        recipe = self._current_recipe()  # reads Tk variables on the main thread

        def work():
            return generate_protocol(recipe, target_properties=DEFAULT_TARGET_PROPERTIES)

        def done(protocol):
            self.current_protocol = protocol
            self.protocol_text.delete("1.0", "end")
            self.protocol_text.insert("1.0", protocol.as_text())
            self.app.log(f"Protocol {protocol.protocol_id} generated ({protocol.completeness_score:.0%} complete).")

        self.app.run_async(work, on_done=done, busy_message="Generating protocol...")

    def on_map_standards(self):
        """Demonstrates standards_mapper.py's fuzzy matching -- feeds it
        loosely/inconsistently-named property strings and shows the matched
        standard plus a confidence score for each."""
        names_str = self.loose_names_var.get()

        def work():
            names = [n.strip() for n in names_str.split(",") if n.strip()]
            mappings = map_all(names)
            return mappings, mapping_accuracy(mappings)

        def done(result):
            mappings, accuracy = result
            lines = [f"'{m['input']}' -> {m.get('matched_property')} (confidence={m['confidence']})" for m in mappings]
            self.mapping_result_var.set("\n".join(lines) + f"\nMean mapping confidence: {accuracy:.3f}")

        self.app.run_async(work, on_done=done, busy_message="Mapping properties to standards...")

    def on_submit_feedback(self):
        """Writes the measured results back into Core 2's database
        (feedback.py, reusing schema.py's Material/MaterialProperty tables)
        so a validated recipe becomes real training data for Core 1 and
        Core 4. Also scores reproducibility against Core 1's original
        prediction, if one is available in app.shared_state."""
        if self.current_protocol is None:
            self.app.log("Generate a protocol first.")
            return
        # Snapshot every Tk variable now, on the main thread.
        measured_raw = {k: v.get() for k, v in self.measured_vars.items()}
        rating_str = self.expert_rating_var.get().strip()

        def work():
            measured = {k: float(v) for k, v in measured_raw.items() if v.strip()}
            if not measured:
                raise ValueError("Enter at least one measured property value.")
            expert_rating = float(rating_str) if rating_str else None

            record = FeedbackRecord(
                protocol_id=self.current_protocol.protocol_id,
                recipe=self.current_protocol.recipe,
                measured_properties=measured,
                expert_rating=expert_rating,
            )
            material_id = submit_feedback(record, db_path=paths.CORE2_DB_URL)

            predicted = self.app.shared_state.get("last_predicted_properties", {})
            repro = reproducibility_score(predicted, measured) if predicted else None
            return material_id, repro

        def done(result):
            material_id, repro = result
            msg = f"Feedback written to Core 2 database as material_id={material_id}."
            if repro is not None:
                msg += f" Reproducibility score (predicted vs. measured): {repro:.3f}"
            self.feedback_result_var.set(msg)
            self.app.log(msg)

        self.app.run_async(work, on_done=done, busy_message="Submitting feedback...")
