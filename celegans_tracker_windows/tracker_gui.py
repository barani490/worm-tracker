#!/usr/bin/env python3
"""
tracker_gui.py

A thin PyGame front-end for celegans_tracker.py.

This GUI does NOT reimplement any tracking logic. It just presents a
form of the same options you'd type on the command line, builds the
equivalent `python3 celegans_tracker.py ...` command from whatever you
filled in, and runs it as a subprocess -- exactly as if you'd typed it
yourself. Live output from the tracker streams into the console panel
at the bottom. When the run finishes, if you asked for a plot
(--output-plot), it's displayed inline; otherwise the console just
shows the tracker's own completion message and output paths.

Requires: pygame (pip install pygame)
Assumes celegans_tracker.py is in the same folder as this script.

Run with:
    python3 tracker_gui.py
"""

import os
import queue
import subprocess
import sys
import threading

import pygame

try:
    import tkinter as tk
    from tkinter import filedialog
    _HAS_TK = True
except Exception:
    _HAS_TK = False


# ---------------------------------------------------------------------------
# Config / layout constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKER_PATH = os.path.join(SCRIPT_DIR, "celegans_tracker.py")

WIDTH, HEIGHT = 980, 720
BG = (245, 246, 247)
PANEL = (255, 255, 255)
BORDER = (209, 213, 219)
TEXT = (27, 42, 58)
MUTED = (107, 114, 128)
ACCENT = (31, 138, 112)
ACCENT_DARK = (23, 105, 85)
FIELD_BG = (249, 250, 251)
FIELD_ACTIVE = (255, 255, 255)
CONSOLE_BG = (20, 24, 28)
CONSOLE_TEXT = (200, 230, 210)

FONT_NAME = None  # default pygame font
LABEL_SIZE = 15
INPUT_SIZE = 15
CONSOLE_SIZE = 13

MODES = ["analyze", "record", "record-analyze", "live", "still"]


# ---------------------------------------------------------------------------
# Simple widgets
# ---------------------------------------------------------------------------

class TextField:
    """A single-line text input box with optional file-browse button."""

    def __init__(self, rect, label, placeholder="", browse=None):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.placeholder = placeholder
        self.text = ""
        self.active = False
        self.browse = browse  # None, "open", or "save"
        if browse:
            self.browse_rect = pygame.Rect(
                self.rect.right - 70, self.rect.y, 70, self.rect.height
            )
            self.rect = pygame.Rect(
                self.rect.x, self.rect.y, self.rect.width - 78, self.rect.height
            )
        else:
            self.browse_rect = None

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
            if self.browse_rect and self.browse_rect.collidepoint(event.pos):
                self._open_dialog()
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_RETURN or event.key == pygame.K_TAB:
                self.active = False
            else:
                if event.unicode and event.unicode.isprintable():
                    self.text += event.unicode

    def _open_dialog(self):
        if not _HAS_TK:
            return
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            if self.browse == "open":
                path = filedialog.askopenfilename()
            else:
                path = filedialog.asksaveasfilename()
        finally:
            root.destroy()
        if path:
            self.text = path

    def draw(self, surf, font, label_font):
        label_surf = label_font.render(self.label, True, TEXT)
        surf.blit(label_surf, (self.rect.x, self.rect.y - 20))

        bg = FIELD_ACTIVE if self.active else FIELD_BG
        pygame.draw.rect(surf, bg, self.rect, border_radius=4)
        pygame.draw.rect(
            surf, ACCENT if self.active else BORDER, self.rect, 1, border_radius=4
        )

        display = self.text if self.text else self.placeholder
        color = TEXT if self.text else MUTED
        txt_surf = font.render(display, True, color)
        surf.blit(
            txt_surf,
            (self.rect.x + 8, self.rect.y + (self.rect.height - txt_surf.get_height()) // 2),
        )

        if self.browse_rect:
            pygame.draw.rect(surf, ACCENT, self.browse_rect, border_radius=4)
            bt = font.render("Browse", True, (255, 255, 255))
            surf.blit(
                bt,
                (
                    self.browse_rect.x + (self.browse_rect.width - bt.get_width()) // 2,
                    self.browse_rect.y + (self.browse_rect.height - bt.get_height()) // 2,
                ),
            )


class Checkbox:
    def __init__(self, pos, label):
        self.rect = pygame.Rect(pos[0], pos[1], 18, 18)
        self.label = label
        self.checked = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            hit = self.rect.inflate(120, 6)
            if hit.collidepoint(event.pos):
                self.checked = not self.checked

    def draw(self, surf, font):
        pygame.draw.rect(surf, FIELD_BG, self.rect, border_radius=3)
        pygame.draw.rect(surf, BORDER, self.rect, 1, border_radius=3)
        if self.checked:
            pygame.draw.rect(surf, ACCENT, self.rect.inflate(-6, -6), border_radius=2)
        label_surf = font.render(self.label, True, TEXT)
        surf.blit(label_surf, (self.rect.right + 8, self.rect.y - 1))


class Dropdown:
    def __init__(self, rect, options, label):
        self.rect = pygame.Rect(rect)
        self.options = options
        self.label = label
        self.selected = 0
        self.open = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.open = not self.open
            elif self.open:
                for i, _ in enumerate(self.options):
                    opt_rect = pygame.Rect(
                        self.rect.x, self.rect.bottom + i * self.rect.height,
                        self.rect.width, self.rect.height,
                    )
                    if opt_rect.collidepoint(event.pos):
                        self.selected = i
                self.open = False

    def value(self):
        return self.options[self.selected]

    def draw_closed(self, surf, font, label_font):
        label_surf = label_font.render(self.label, True, TEXT)
        surf.blit(label_surf, (self.rect.x, self.rect.y - 20))

        pygame.draw.rect(surf, FIELD_BG, self.rect, border_radius=4)
        pygame.draw.rect(surf, ACCENT if self.open else BORDER, self.rect, 1, border_radius=4)
        txt = font.render(self.options[self.selected], True, TEXT)
        surf.blit(txt, (self.rect.x + 8, self.rect.y + (self.rect.height - txt.get_height()) // 2))
        # little arrow
        cx, cy = self.rect.right - 14, self.rect.centery
        pygame.draw.polygon(surf, MUTED, [(cx - 5, cy - 3), (cx + 5, cy - 3), (cx, cy + 4)])

    def draw_open_overlay(self, surf, font):
        """Draws the expanded option list. Must be called AFTER every other
        widget on the screen has already been drawn, so this list renders on
        top of the form fields below it instead of being painted over by them."""
        if not self.open:
            return
        for i, opt in enumerate(self.options):
            opt_rect = pygame.Rect(
                self.rect.x, self.rect.bottom + i * self.rect.height,
                self.rect.width, self.rect.height,
            )
            pygame.draw.rect(surf, PANEL, opt_rect)
            pygame.draw.rect(surf, BORDER, opt_rect, 1)
            txt = font.render(opt, True, TEXT)
            surf.blit(txt, (opt_rect.x + 8, opt_rect.y + 4))


class Button:
    def __init__(self, rect, label, color=ACCENT, text_color=(255, 255, 255)):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.color = color
        self.text_color = text_color
        self.enabled = True

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.enabled:
            if self.rect.collidepoint(event.pos):
                return True
        return False

    def draw(self, surf, font):
        color = self.color if self.enabled else BORDER
        pygame.draw.rect(surf, color, self.rect, border_radius=6)
        txt = font.render(self.label, True, self.text_color)
        surf.blit(
            txt,
            (
                self.rect.x + (self.rect.width - txt.get_width()) // 2,
                self.rect.y + (self.rect.height - txt.get_height()) // 2,
            ),
        )


# ---------------------------------------------------------------------------
# Command building + subprocess execution
# ---------------------------------------------------------------------------

def build_command(fields, checkboxes, dropdown):
    """Turn the current form state into the actual CLI argument list.
    Only includes flags the user actually filled in / checked -- anything
    left blank is simply omitted, so celegans_tracker.py's own defaults apply.
    """
    cmd = [sys.executable, TRACKER_PATH, "--mode", dropdown.value()]

    flag_map = {
        "input": "--input",
        "output_csv": "--output-csv",
        "output_video": "--output-video",
        "output_plot": "--output-plot",
        "max_frames": "--max-frames",
        "frame_skip": "--frame-skip",
        "mm_per_px": "--mm-per-px",
    }
    for key, flag in flag_map.items():
        val = fields[key].text.strip()
        if val:
            cmd.extend([flag, val])

    if checkboxes["show"].checked:
        cmd.append("--show")

    extra = fields["extra_args"].text.strip()
    if extra:
        cmd.extend(extra.split())

    return cmd


def run_tracker(cmd, out_queue):
    """Runs in a background thread; streams stdout/stderr lines into out_queue."""
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            out_queue.put(("line", line.rstrip("\n")))
        proc.wait()
        out_queue.put(("done", proc.returncode))
    except Exception as e:
        out_queue.put(("line", f"[GUI] Failed to launch: {e}"))
        out_queue.put(("done", -1))


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("C. elegans Tracker - GUI")
    clock = pygame.time.Clock()

    label_font = pygame.font.SysFont(FONT_NAME, LABEL_SIZE, bold=True)
    input_font = pygame.font.SysFont(FONT_NAME, INPUT_SIZE)
    console_font = pygame.font.SysFont("consolas", CONSOLE_SIZE)
    title_font = pygame.font.SysFont(FONT_NAME, 22, bold=True)

    left_x = 30
    field_w = 560
    field_h = 30

    dropdown = Dropdown((left_x, 100, 220, field_h), MODES, "Mode")

    fields = {
        "input": TextField((left_x, 160, field_w, field_h), "Input video",
                            "path/to/video.mp4 or .avi", browse="open"),
        "output_csv": TextField((left_x, 220, field_w, field_h), "Output CSV",
                                 "leave blank for tracker's default", browse="save"),
        "output_video": TextField((left_x, 280, field_w, field_h), "Output video (optional)",
                                   "leave blank to skip", browse="save"),
        "output_plot": TextField((left_x, 340, field_w, field_h), "Output plot PNG (optional)",
                                  "leave blank to skip", browse="save"),
        "max_frames": TextField((left_x, 400, 180, field_h), "Max frames (optional)", "all"),
        "frame_skip": TextField((left_x + 210, 400, 180, field_h), "Frame skip (optional)", "1"),
        "mm_per_px": TextField((left_x + 420, 400, 170, field_h), "mm per px (optional)", "uncalibrated"),
        "extra_args": TextField((left_x, 460, field_w, field_h), "Extra flags (advanced, optional)",
                                 "e.g. --clahe --min-area 100"),
    }

    checkboxes = {
        "show": Checkbox((left_x, 520), "Show live preview window (--show)"),
    }

    run_button = Button((left_x, 560, 140, 40), "Run")
    clear_button = Button((left_x + 155, 560, 140, 40), "Clear console",
                           color=(107, 114, 128))

    console_rect = pygame.Rect(left_x, 618, field_w + 78, 90)
    console_lines = ["Ready. Fill in the fields above and click Run."]
    max_console_lines = 6

    out_queue = queue.Queue()
    worker_thread = None
    running_proc = False

    plot_surface = None  # loaded plot image, shown after a run with --output-plot

    all_fields = list(fields.values())

    def try_load_plot():
        nonlocal plot_surface
        path = fields["output_plot"].text.strip()
        if path and os.path.isfile(path):
            try:
                plot_surface = pygame.image.load(path)
            except Exception as e:
                console_lines.append(f"[GUI] Could not load plot image: {e}")

    clock_running = True
    while clock_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                clock_running = False

            dropdown.handle_event(event)
            for f in all_fields:
                f.handle_event(event)
            for cb in checkboxes.values():
                cb.handle_event(event)

            if run_button.handle_event(event) and not running_proc:
                cmd = build_command(fields, checkboxes, dropdown)
                console_lines = [f"[GUI] Running: {' '.join(cmd)}"]
                plot_surface = None
                running_proc = True
                run_button.enabled = False
                worker_thread = threading.Thread(
                    target=run_tracker, args=(cmd, out_queue), daemon=True
                )
                worker_thread.start()

            if clear_button.handle_event(event):
                console_lines = []

        # drain any output from the background thread
        try:
            while True:
                kind, payload = out_queue.get_nowait()
                if kind == "line":
                    console_lines.append(payload)
                elif kind == "done":
                    console_lines.append(f"[GUI] Finished (exit code {payload}).")
                    running_proc = False
                    run_button.enabled = True
                    try_load_plot()
        except queue.Empty:
            pass

        # ---- draw ----
        screen.fill(BG)

        title = title_font.render("C. elegans Tracker", True, TEXT)
        screen.blit(title, (left_x, 20))
        subtitle = input_font.render(
            "Fills in celegans_tracker.py's CLI flags and runs it for you.",
            True, MUTED,
        )
        screen.blit(subtitle, (left_x, 48))

        dropdown.draw_closed(screen, input_font, label_font)
        for f in all_fields:
            f.draw(screen, input_font, label_font)
        for cb in checkboxes.values():
            cb.draw(screen, input_font)

        run_button.label = "Running..." if running_proc else "Run"
        run_button.draw(screen, input_font)
        clear_button.draw(screen, input_font)

        # console panel
        pygame.draw.rect(screen, CONSOLE_BG, console_rect, border_radius=6)
        y = console_rect.y + 8
        for line in console_lines[-max_console_lines:]:
            line_surf = console_font.render(line[:110], True, CONSOLE_TEXT)
            screen.blit(line_surf, (console_rect.x + 10, y))
            y += 16

        # plot preview, if one was produced and loaded
        if plot_surface is not None:
            preview_x = console_rect.right + 20
            if preview_x + 220 < WIDTH:
                scaled = pygame.transform.smoothscale(plot_surface, (220, 160))
                screen.blit(scaled, (preview_x, console_rect.y))
                pygame.draw.rect(
                    screen, BORDER, (preview_x, console_rect.y, 220, 160), 1
                )

        # dropdown's open option list is drawn LAST so it renders on top of
        # every other field instead of being covered by them
        dropdown.draw_open_overlay(screen, input_font)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
