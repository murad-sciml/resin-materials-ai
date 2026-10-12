"""
Centralized visual styling for the Resin-Coated Material control panel.

Kept in one place so every tab looks consistent, and so the color scheme can
be tweaked without hunting through five files. Nothing in here talks to any
core-function module -- it's purely cosmetic.
"""
from __future__ import annotations

from tkinter import ttk

# ---- palette --------------------------------------------------------------
# A calm blue/teal palette: BG for the general background, PANEL for framed
# sections, ACCENT for primary buttons, and a dark console pair that reads
# like an actual terminal rather than a plain white text box.
BG = "#f4f6f8"
PANEL_BG = "#ffffff"
BORDER = "#d0d7de"
ACCENT = "#2563eb"        # primary buttons
ACCENT_DARK = "#1d4ed8"   # hover/active state
TEXT = "#1f2937"
MUTED_TEXT = "#5b6572"
HEADER_BG = "#0f2942"     # title bar behind the app name
HEADER_TEXT = "#ffffff"

CONSOLE_BG = "#111418"
CONSOLE_FG = "#c9f4d0"    # soft green, easy on the eyes for a log panel
CONSOLE_FONT = ("Consolas", 9)

TITLE_FONT = ("Segoe UI", 13, "bold")
SECTION_FONT = ("Segoe UI", 10, "bold")
BODY_FONT = ("Segoe UI", 9)


def apply_theme(root) -> None:
    """
    Configures ttk's global style. Call this once, right after creating the
    root Tk() window, before building any widgets.
    """
    root.configure(bg=BG)

    style = ttk.Style(root)
    # 'clam' is the most "themeable" built-in ttk theme -- unlike the default
    # theme on Windows/Mac, it actually respects the color options below.
    style.theme_use("clam")

    style.configure(".", background=BG, foreground=TEXT, font=BODY_FONT)
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=TEXT, font=BODY_FONT)
    style.configure("TLabelframe", background=BG, foreground=TEXT, font=SECTION_FONT, bordercolor=BORDER)
    style.configure("TLabelframe.Label", background=BG, foreground=ACCENT_DARK, font=SECTION_FONT)

    # Section titles used at the top of each tab (see helpers.section_title).
    style.configure("Title.TLabel", font=TITLE_FONT, foreground=HEADER_TEXT, background=HEADER_BG)
    style.configure("Subtitle.TLabel", font=BODY_FONT, foreground=MUTED_TEXT, background=BG)
    style.configure("Header.TFrame", background=HEADER_BG)

    # Buttons: a filled "primary" style for the main action in each section,
    # and a plain style (ttk default look) for secondary actions.
    style.configure(
        "Primary.TButton",
        background=ACCENT, foreground="white", font=("Segoe UI", 9, "bold"),
        padding=(10, 6), borderwidth=0,
    )
    style.map("Primary.TButton", background=[("active", ACCENT_DARK), ("disabled", BORDER)])

    style.configure("TButton", padding=(8, 5))

    # Notebook (tab strip) styling.
    style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(4, 4, 4, 0))
    style.configure(
        "TNotebook.Tab", padding=(14, 8), font=("Segoe UI", 9, "bold"),
        background=PANEL_BG, foreground=MUTED_TEXT,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", ACCENT)],
        foreground=[("selected", "white")],
    )

    # Treeview (tables): a slightly larger row height reads much better than
    # the cramped default, and a light selection color that isn't jarring.
    style.configure("Treeview", background=PANEL_BG, fieldbackground=PANEL_BG, rowheight=24, font=BODY_FONT)
    style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"), background="#eef2f6")
    style.map("Treeview", background=[("selected", ACCENT)], foreground=[("selected", "white")])


def section_title(parent, text: str, subtitle: str | None = None) -> ttk.Frame:
    """
    Standard header block used at the top of every tab: a colored bar with
    the section name, plus an optional muted subtitle line underneath.
    Returns the frame so the caller can .pack() it wherever needed.
    """
    wrapper = ttk.Frame(parent)
    header = ttk.Frame(wrapper, style="Header.TFrame", padding=(12, 8))
    header.pack(fill="x")
    ttk.Label(header, text=text, style="Title.TLabel").pack(anchor="w")
    if subtitle:
        ttk.Label(wrapper, text=subtitle, style="Subtitle.TLabel", wraplength=1050).pack(
            anchor="w", padx=2, pady=(6, 0)
        )
    return wrapper
