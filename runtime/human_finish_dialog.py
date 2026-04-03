from __future__ import annotations

import textwrap

from utils.colors import COLOR_RGB


VALID_CUBES = ("green", "red", "blue")


def _resolve_valid_cubes(valid_cubes: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if not valid_cubes:
        return VALID_CUBES
    normalized: list[str] = []
    for cube in valid_cubes:
        cube_name = str(cube).strip().lower()
        if cube_name and cube_name not in normalized:
            normalized.append(cube_name)
    return tuple(normalized) if normalized else VALID_CUBES


def prompt_human_cube_selection(
    *,
    ui_backend: str = "tkinter",
    valid_cubes: tuple[str, ...] | list[str] | None = None,
) -> dict[str, str] | None:
    resolved_cubes = _resolve_valid_cubes(valid_cubes)
    if ui_backend == "opencv":
        return _prompt_human_cube_selection_opencv(valid_cubes=resolved_cubes)
    if ui_backend == "tkinter":
        return _prompt_human_cube_selection_tkinter(valid_cubes=resolved_cubes)
    raise ValueError(f"Unsupported human finish ui_backend={ui_backend!r}.")


def _prompt_human_cube_selection_tkinter(*, valid_cubes: tuple[str, ...]) -> dict[str, str] | None:
    try:
        import tkinter as tk
        from tkinter import messagebox, scrolledtext
    except ImportError as exc:  # pragma: no cover - platform dependency
        raise RuntimeError(
            "tkinter is required for the human finish dialog."
        ) from exc

    result: dict[str, str] | None = None
    root = tk.Tk()
    root.title("Select a cube")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    selected_cube = tk.StringVar(value="")
    allowed_text = ", ".join(valid_cubes)

    def _submit(event=None):  # pragma: no cover - exercised via controller tests
        cube = selected_cube.get().strip().lower()
        reasoning = reasoning_input.get("1.0", "end").strip()
        if cube not in valid_cubes:
            messagebox.showerror("Select a cube", f"Choose one of: {allowed_text}.", parent=root)
            return "break"
        if not reasoning:
            messagebox.showerror("Add reasoning", "Enter your reasoning before submitting.", parent=root)
            return "break"

        nonlocal result
        result = {
            "selected_cube": cube,
            "reasoning": reasoning,
        }
        root.quit()
        return "break"

    def _cancel():  # pragma: no cover - exercised via controller tests
        root.quit()

    def _insert_newline(event):  # pragma: no cover - UI convenience
        reasoning_input.insert("insert", "\n")
        return "break"

    container = tk.Frame(root, padx=16, pady=16)
    container.grid(row=0, column=0, sticky="nsew")

    tk.Label(container, text="Select a cube", font=("Helvetica", 14, "bold")).grid(
        row=0,
        column=0,
        columnspan=3,
        sticky="w",
    )

    for column, cube in enumerate(valid_cubes):
        tk.Radiobutton(
            container,
            text=cube.capitalize(),
            value=cube,
            variable=selected_cube,
        ).grid(row=1, column=column, padx=(0, 12), pady=(8, 12), sticky="w")

    tk.Label(
        container,
        text="Reasoning",
        font=("Helvetica", 12, "bold"),
    ).grid(row=2, column=0, columnspan=3, sticky="w")

    reasoning_input = scrolledtext.ScrolledText(container, width=48, height=8, wrap="word")
    reasoning_input.grid(row=3, column=0, columnspan=3, pady=(8, 8), sticky="nsew")

    tk.Label(
        container,
        text="Press Enter to submit. Use Shift+Enter for a new line.",
    ).grid(row=4, column=0, columnspan=3, sticky="w")

    button_row = tk.Frame(container)
    button_row.grid(row=5, column=0, columnspan=3, pady=(12, 0), sticky="e")
    tk.Button(button_row, text="Cancel", command=_cancel).grid(row=0, column=0, padx=(0, 8))
    tk.Button(button_row, text="Submit", command=_submit).grid(row=0, column=1)

    root.protocol("WM_DELETE_WINDOW", _cancel)
    root.bind("<Return>", _submit)
    reasoning_input.bind("<Return>", _submit)
    reasoning_input.bind("<Shift-Return>", _insert_newline)
    root.after(0, reasoning_input.focus_set)
    root.mainloop()
    root.destroy()
    return result


def _prompt_human_cube_selection_opencv(*, valid_cubes: tuple[str, ...]) -> dict[str, str] | None:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - platform dependency
        raise RuntimeError(
            "OpenCV and numpy are required for the MuJoCo human finish dialog."
        ) from exc

    window_name = "Select a cube"
    selected_cube = valid_cubes[0]
    reasoning = ""
    shortcuts = "   ".join(f"{idx+1}={cube.capitalize()}" for idx, cube in enumerate(valid_cubes))
    status_message = "Pick a cube with number keys, type reasoning, then press Enter."
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            canvas = np.full((720, 960, 3), 248, dtype=np.uint8)
            cv2.putText(canvas, "Select a cube", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 2)
            cv2.putText(
                canvas,
                f"{shortcuts}   Enter=Submit   Esc=Cancel   Backspace=Delete",
                (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (60, 60, 60),
                1,
            )

            card_width = 860 // max(1, len(valid_cubes))
            for index, cube in enumerate(valid_cubes):
                x0 = 40 + index * card_width
                x1 = x0 + 250
                y0 = 140
                y1 = 260
                border = (20, 20, 20) if cube == selected_cube else (120, 120, 120)
                thickness = 4 if cube == selected_cube else 2
                cv2.rectangle(canvas, (x0, y0), (x1, y1), _cube_fill_bgr(cube), -1)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), border, thickness)
                cv2.putText(
                    canvas,
                    cube.capitalize(),
                    (x0 + 22, y0 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (20, 20, 20),
                    2,
                )
                if cube == selected_cube:
                    cv2.putText(
                        canvas,
                        "selected",
                        (x0 + 22, y0 + 104),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (20, 20, 20),
                        2,
                    )

            cv2.putText(canvas, "Reasoning", (40, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
            cv2.rectangle(canvas, (40, 340), (920, 660), (180, 180, 180), 2)

            visible_text = reasoning + ("|" if len(reasoning) % 2 == 0 else "")
            wrapped_lines = textwrap.wrap(visible_text, width=72) or ["|"]
            wrapped_lines = wrapped_lines[-11:]
            for line_index, line in enumerate(wrapped_lines):
                y = 380 + line_index * 24
                cv2.putText(
                    canvas,
                    line,
                    (56, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (40, 40, 40),
                    1,
                )

            status_color = (20, 120, 20) if selected_cube and reasoning.strip() else (20, 20, 160)
            cv2.putText(canvas, status_message, (40, 695), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1)
            cv2.imshow(window_name, canvas)

            key = cv2.waitKey(16)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key < 0:
                continue
            key = key & 0xFF

            selected_cube, reasoning, status_message, submission, cancelled = _handle_opencv_finish_key(
                key,
                selected_cube=selected_cube,
                reasoning=reasoning,
                valid_cubes=valid_cubes,
            )
            if cancelled:
                return None
            if submission is not None:
                return submission
    finally:
        cv2.destroyWindow(window_name)


def _handle_opencv_finish_key(
    key: int,
    *,
    selected_cube: str,
    reasoning: str,
    valid_cubes: tuple[str, ...] = VALID_CUBES,
) -> tuple[str, str, str, dict[str, str] | None, bool]:
    if key == 27:
        return selected_cube, reasoning, "Finish dialog cancelled.", None, True
    if ord("1") <= key <= ord("9"):
        index = key - ord("1")
        if 0 <= index < len(valid_cubes):
            cube = valid_cubes[index]
            return cube, reasoning, f"{cube.capitalize()} selected.", None, False
        return selected_cube, reasoning, "No cube bound to that shortcut.", None, False
    if key in (8, 127):
        return selected_cube, reasoning[:-1], "Editing reasoning.", None, False
    if key in (10, 13):
        if reasoning.strip():
            return (
                selected_cube,
                reasoning,
                "Submitting selection.",
                {
                    "selected_cube": selected_cube,
                    "reasoning": reasoning.strip(),
                },
                False,
            )
        return selected_cube, reasoning, "Reasoning is required before submitting.", None, False
    if 32 <= key <= 126:
        return selected_cube, reasoning + chr(key), "Editing reasoning.", None, False
    return selected_cube, reasoning, "Pick a cube with number keys, type reasoning, then press Enter.", None, False


def _cube_fill_bgr(cube_name: str) -> tuple[int, int, int]:
    rgb = COLOR_RGB.get(str(cube_name).strip().lower())
    if rgb is None:
        return (190, 190, 190)
    r = int(max(0.0, min(1.0, rgb[0])) * 255)
    g = int(max(0.0, min(1.0, rgb[1])) * 255)
    b = int(max(0.0, min(1.0, rgb[2])) * 255)
    return (b, g, r)
