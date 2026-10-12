"""
Resin-Coated Material Project -- desktop control panel.

A single Tkinter app with one tab per core function (Data Integration,
AI Reverse Design, Validation Protocol, Autonomous Experiment) plus a
Dashboard tab and a live Console log, so the whole pipeline can be run and
inspected without touching four separate terminals.

HOW THIS FILE FITS TOGETHER
----------------------------
- paths.py        adds core1-4's folders to sys.path and defines shared
                   absolute file paths, so every tab can import e.g.
                   `forward_model` directly regardless of where this app
                   was launched from.
- style.py         all colors/fonts/ttk theme live here, kept separate from
                   layout logic so the look can be tweaked in one place.
- tab_coreN.py     one file per tab; each imports the real core-function
                   modules (schema.py, forward_model.py, etc.) and wires
                   their functions up to buttons/tables.
- app.py (this)    the window shell: builds the tab strip, the console
                   panel, and the background-task runner every tab uses so
                   long operations (training a model, querying a database)
                   never freeze the window.

THREADING MODEL (read this before touching run_async)
--------------------------------------------------------
Tkinter widgets and variables (StringVar, etc.) may ONLY be read or written
from the main thread. Training a model or hitting a database can take
seconds, and doing that on the main thread would freeze the whole window.
run_async() below solves this with a simple pattern:

    1. The button handler reads any Tk variables it needs *right away*,
       on the main thread (e.g. `target = self.target_var.get()`).
    2. It defines work_fn() as a closure over those plain Python values
       (never over the Tk variables themselves) and calls run_async(work_fn,
       on_done=...).
    3. run_async spawns a background thread to run work_fn(). Tkinter is
       never touched from that thread.
    4. When work_fn() finishes (or raises), the result (or exception) is
       handed back to the main thread via a thread-safe queue, and on_done
       (or on_error) runs there -- safe to update labels/tables from there.

Every tab in this project follows that pattern. If you add a new button that
reads a Tk variable, read it BEFORE defining work_fn, not inside it.
"""
from __future__ import annotations

import datetime
import queue
import threading
import traceback
import tkinter as tk
from tkinter import ttk, scrolledtext

import paths  # noqa: F401  (import first: sets up sys.path for core1-4 modules)
import style

from tab_core1 import Core1Tab
from tab_core2 import Core2Tab
from tab_core3 import Core3Tab
from tab_core4 import Core4Tab
from tab_core5 import Core5Tab


class ResinProjectApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Resin-Coated Material Project -- Control Panel")
        self.geometry("1180x800")
        self.minsize(1000, 650)
        style.apply_theme(self)  # ttk colors/fonts; must run before building widgets

        # Shared state passed between tabs. Currently holds:
        #   "last_recipe"               -- dict from Core 1's reverse-design or
        #                                  Pareto selection, consumed by Core 3.
        #   "last_predicted_properties" -- Core 1's predicted values for that
        #                                  recipe, used by Core 3 to score
        #                                  reproducibility once real results
        #                                  come back from the lab.
        self.shared_state: dict = {}

        # Background threads never touch Tkinter directly; they drop a
        # zero-argument callable here, and _poll_queue runs it on the main
        # thread shortly after. See the threading-model note above.
        self.task_queue: "queue.Queue[callable]" = queue.Queue()

        self._build_layout()
        self.after(100, self._poll_queue)
        self.log("Ready. Start with the 'Core 2: Data Integration' tab.")

    # ------------------------------------------------------------------ UI
    def _build_layout(self):
        header = style.section_title(
            self,
            "Resin-Coated Material Project",
            "Data Integration -> AI Reverse Design -> Validation Protocol -> Autonomous Experiment -> Dashboard",
        )
        header.pack(fill="x")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=(8, 0))

        # Tabs are created in the order a person would actually use them,
        # not the order the core functions are numbered in the spec.
        self.core2_tab = Core2Tab(self.notebook, self)
        self.core1_tab = Core1Tab(self.notebook, self)
        self.core3_tab = Core3Tab(self.notebook, self)
        self.core4_tab = Core4Tab(self.notebook, self)
        self.core5_tab = Core5Tab(self.notebook, self)

        self.notebook.add(self.core2_tab, text="\U0001F9EA  Core 2: Data Integration")
        self.notebook.add(self.core1_tab, text="\U0001F916  Core 1: AI Reverse Design")
        self.notebook.add(self.core3_tab, text="\U0001F4CB  Core 3: Validation Protocol")
        self.notebook.add(self.core4_tab, text="\U0001F52C  Core 4: Autonomous Experiment")
        self.notebook.add(self.core5_tab, text="\U0001F4CA  Core 5: Dashboard")

        console_frame = ttk.LabelFrame(self, text="Console")
        console_frame.pack(fill="both", expand=False, padx=8, pady=8)
        # A dark background + light monospace text reads like an actual
        # terminal, which is the point of a "console" panel -- a plain white
        # Text widget would just look like another form field.
        self.console = scrolledtext.ScrolledText(
            console_frame, height=10, state="disabled", font=style.CONSOLE_FONT,
            bg=style.CONSOLE_BG, fg=style.CONSOLE_FG, insertbackground=style.CONSOLE_FG,
            borderwidth=0, padx=8, pady=6,
        )
        self.console.pack(fill="both", expand=True, padx=4, pady=4)

        self.status_var = tk.StringVar(value="Ready.")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w", padding=(8, 3))
        status_bar.pack(fill="x", side="bottom")

    # ------------------------------------------------------------ logging
    def log(self, msg: str):
        """Append a timestamped line to the Console panel and the status bar.
        Must be called from the main thread -- worker threads should go
        through enqueue() instead (see the threading-model note above)."""
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.console.configure(state="normal")
        self.console.insert("end", f"[{ts}] {msg}\n")
        self.console.see("end")  # auto-scroll to the newest line
        self.console.configure(state="disabled")
        first_line = str(msg).splitlines()[0] if msg else ""
        self.status_var.set(first_line[:140])

    # ------------------------------------------------------- async runner
    def enqueue(self, fn):
        """Schedule fn to run on the main thread on the next queue poll.
        Safe to call from any thread -- this is how background threads hand
        results (or log messages) back to the GUI."""
        self.task_queue.put(fn)

    def _poll_queue(self):
        """Runs every 100ms on the main thread (via `after`), draining
        anything worker threads have queued up. This is the only place
        background-thread results ever touch Tkinter."""
        try:
            while True:
                fn = self.task_queue.get_nowait()
                try:
                    fn()
                except Exception:
                    self.log("ERROR while updating UI:\n" + traceback.format_exc())
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def run_async(self, work_fn, on_done=None, on_error=None, busy_message: str | None = None):
        """
        Runs work_fn() in a background thread so the GUI never freezes.
        on_done(result) and on_error(exception) are both invoked on the main
        thread. Use this for anything that trains a model, hits a database,
        or otherwise takes more than a few milliseconds.

        IMPORTANT: work_fn must not touch any Tkinter widget or variable --
        only plain Python values captured by the caller beforehand. See the
        threading-model note at the top of this file.
        """
        if busy_message:
            self.log(busy_message)

        def target():
            try:
                result = work_fn()
            except Exception as exc:  # noqa: BLE001 - deliberately broad: surface any failure to the console
                tb = traceback.format_exc()
                error_msg = str(exc)
                # Capture exc/tb as default-arg values (evaluated now, not at call time) --
                # `except ... as exc` unbinds `exc` when the block ends, so a lambda that
                # merely *references* it would raise NameError once actually invoked later.
                if on_error:
                    self.enqueue(lambda err=exc: on_error(err))
                else:
                    self.enqueue(lambda msg=error_msg, trace=tb: self.log(f"ERROR: {msg}\n{trace}"))
                return
            if on_done:
                self.enqueue(lambda: on_done(result))

        threading.Thread(target=target, daemon=True).start()


if __name__ == "__main__":
    app = ResinProjectApp()
    app.mainloop()

