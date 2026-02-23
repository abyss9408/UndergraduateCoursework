# app_seam_ui.py — DP seam carving with UI (loads your data/Broadway_tower_edit.jpg)
# Works without venv; uses .packages alongside this file.

"""
app_seam_ui.py — Interactive Seam Carving (DP + Greedy) with DearPyGui UI
===========================================================================
This project works without a virtual environment (uses .packages directory)
Simply run: run.bat to use the app.

This module provides a full graphical UI for content-aware image resizing
(seam carving). It loads an image, displays it, and allows the user to
remove seams using 
    Dynamic Programming (forward-energy)
    Greedy seam selection

Main features:
    - Live seam preview 
    - Step-by-step or N-seam carving
    - Dynamic texture updates inside DearPyGui
    - Optional preview window for seam visualization
    - Reset and save operations

Dependencies:
    - numpy
    - opencv-python (cv2)
    - dearpygui
    - seamcarvedp  (your DP implementation: min_vertical_seamdp, remove_vertical_seamdp)
    - seamcarvegreedy (for greedy_vertical_seam)

The program launches a DearPyGui window that allows interactive carving.
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".packages"))

import numpy as np
import cv2
from dearpygui import dearpygui as dpg
from seamcarvedp import min_vertical_seamdp, remove_vertical_seamdp
from seamcarvegreedy import greedy_vertical_seam



# ---------- helpers ----------

STATE = {
    "img": None,
    "orig": None,
    "path": "data/Broadway_tower_edit.jpg",
    "tex_tag": None,          # current texture tag
    "tex_idx": 0,             # increments to make tags unique
    "img_widget": "img_widget",
    "status": "Ready",
    "preview_on": False,
    "preview_win": "preview_win",
    "preview_tex": None,
    "preview_idx": 0,
    "preview_w": 0,
    "preview_h": 0,
    "preview_img_widget": "preview_img",  # fixed tag for the image in the preview window

}

## @brief Compute Sobel gradient energy map
## @param img Input image (BGR or grayscale)
## @return Grayscale energy map
def sobel_energy(img: np.ndarray) -> np.ndarray:
    """Energy = |∂x| + |∂y| on grayscale."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.abs(gx) + np.abs(gy)

## @brief Convert BGR image to flattened RGBA float array for DearPyGui
## @param img_bgr Input BGR image
## @return (flattened RGBA float list, width, height)
def img_to_rgba_float(img_bgr):
    rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
    h, w, _ = rgba.shape
    return rgba.ravel().tolist(), w, h  # DearPyGui is happiest with a list

## @brief Update the status text in the UI
## @param msg Status message
def set_status(msg):
    STATE["status"] = msg
    dpg.set_value("status_text", f"{msg}    ({STATE['img'].shape[1]}×{STATE['img'].shape[0]})")

## @brief Attach/update an image widget to point at a texture
## @param image_tag The DPG image widget tag
## @param tex_tag The texture tag
## @param parent Optional parent container for new image creation
def set_image_texture(image_tag: str, tex_tag: str, parent: str | None = None):
    """Point an image widget at a texture. If the widget doesn't exist yet,
    only create it when a valid parent is provided."""
    if dpg.does_item_exist(image_tag):
        try:
            dpg.configure_item(image_tag, texture_tag=tex_tag)   # DPG 1.11+
        except SystemError:
            p = dpg.get_item_parent(image_tag)
            dpg.delete_item(image_tag)
            dpg.add_image(tex_tag, tag=image_tag, parent=p)
    else:
        if parent is not None:
            dpg.add_image(tex_tag, tag=image_tag, parent=parent)
        # if parent is None, do nothing; caller will add the image later

## @brief Refresh the DPG texture from STATE["img"].
def refresh_texture():
    data, w, h = img_to_rgba_float(STATE["img"])

    old_tag = STATE.get("tex_tag")
    STATE["tex_idx"] = STATE.get("tex_idx", 0) + 1
    new_tag = f"tex_{STATE['tex_idx']}"

    with dpg.texture_registry(show=False):
        dpg.add_dynamic_texture(width=w, height=h, default_value=data, tag=new_tag)

    # point the image widget to the new texture (safe even if widget not created yet)
    set_image_texture(STATE["img_widget"], new_tag)

    if old_tag and dpg.does_item_exist(old_tag):
        dpg.delete_item(old_tag)

    STATE["tex_tag"] = new_tag

    if dpg.does_item_exist("status_text"):
        dpg.set_value("status_text", f"{STATE['status']}  |  {w}×{h}")

## @brief Load an image from the path textbox and update UI
def load_image_from_path(sender=None, app_data=None, user_data=None):
    path = dpg.get_value("path_input")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        set_status(f"Failed to load: {path}")
        return
    STATE["img"] = img
    STATE["orig"] = img.copy()
    STATE["path"] = path
    refresh_texture()
    set_status(f"Loaded: {os.path.basename(path)}")

## @brief Save current image to disk as out_seam.png 
def save_image():
    out = os.path.join(os.path.dirname(STATE["path"]), "out_seam.png")
    cv2.imwrite(out, STATE["img"])
    set_status(f"Saved: {out}")

## @brief Carve a single vertical seam using forward energy dynamic programming 
def carve_width_once():
    if STATE["img"].shape[1] <= 2:
        STATE["status"] = "Cannot carve: width too small"
        dpg.set_value("status_text", STATE["status"])
        return

    E = sobel_energy(STATE["img"])

    # Use forward energy - pass the image
    seam = min_vertical_seamdp(E, img=STATE["img"])

    # if Toggle preview is ON, show seam but still carve once
    if STATE["preview_on"]:
        vis = STATE["img"].copy()
        for i, j in enumerate(seam):
            vis[i, j] = (0, 0, 255)
        show_preview(vis)

    STATE["img"] = remove_vertical_seamdp(STATE["img"], seam)
    STATE["status"] = "Carved 1 column (pure forward energy)"
    refresh_texture()

## @brief Carve N vertical seams using forward energy DP    
def carve_width_n():
    n = max(0, int(dpg.get_value("n_cols")))
    n = min(n, max(0, STATE["img"].shape[1] - 2))

    for i in range(n):
        E = sobel_energy(STATE["img"])
        seam = min_vertical_seamdp(E, img=STATE["img"])
        STATE["img"] = remove_vertical_seamdp(STATE["img"], seam)

    STATE["status"] = f"Carved {n} columns (forward energy)"
    refresh_texture()

## @brief Carve a single horizontal seam via transpose trick
def carve_height_once():
    # transpose trick to reuse vertical code
    tr = np.transpose(STATE["img"], (1, 0, 2))
    E = sobel_energy(tr)
    seam = min_vertical_seamdp(E, img=tr)
    tr = remove_vertical_seamdp(tr, seam)
    STATE["img"] = np.transpose(tr, (1, 0, 2))
    STATE["status"] = "Carved 1 row (forward energy)"
    refresh_texture()

## @brief Carve N horizontal seams
def carve_height_n():
    n = max(0, int(dpg.get_value("n_rows")))
    for _ in range(n):
        carve_height_once()

## @brief Restore the working image to the original loaded image
def reset_to_original():
    if STATE["orig"] is not None:
        STATE["img"] = STATE["orig"].copy()
        STATE["status"] = "Reset"
        refresh_texture()

## @brief Toggle showing seam preview window
def toggle_preview():
    STATE["preview_on"] = not STATE["preview_on"]
    set_status(f"Seam preview: {STATE['preview_on']}")

## @brief Show preview image in a separate window.
## @param vis_img Image with seam visualization applied
def show_preview(vis_img):
    data, w, h = img_to_rgba_float(vis_img)
    win = STATE["preview_win"]

    # Ensure the preview window exists
    if not dpg.does_item_exist(win):
        with dpg.window(label="Seam preview", tag=win, width=w+32, height=h+64, pos=(40, 40)):
            pass
    else:
        dpg.configure_item(win, width=w+32, height=h+64)

    # If user closed it (hidden), show it again
    if not dpg.is_item_shown(win):
        dpg.show_item(win)

    # Texture: create once per size, otherwise just update pixels
    need_new_tex = (
        STATE["preview_tex"] is None
        or not dpg.does_item_exist(STATE["preview_tex"])
        or STATE["preview_w"] != w
        or STATE["preview_h"] != h
    )

    if need_new_tex:
        if STATE["preview_tex"] and dpg.does_item_exist(STATE["preview_tex"]):
            dpg.delete_item(STATE["preview_tex"])
        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(width=w, height=h, default_value=data, tag="tex_prev")
        STATE["preview_tex"] = "tex_prev"
        STATE["preview_w"], STATE["preview_h"] = w, h

        # Ensure there is an image widget pointing at the texture
        if dpg.does_item_exist(STATE["preview_img_widget"]):
            dpg.configure_item(STATE["preview_img_widget"], texture_tag=STATE["preview_tex"])
        else:
            dpg.add_image(STATE["preview_tex"], tag=STATE["preview_img_widget"], parent=win)
    else:
        dpg.set_value(STATE["preview_tex"], data)


#--------------------------------------------------------------------
## @brief Compute a vertical seam using a chosen algorithm 
## @param algo "dp" or "greedy" 
## @return Seam list of x-coordinates per row 
def _compute_vertical_seam(algo: str):
    E = sobel_energy(STATE["img"])
    if algo == "dp":
        return min_vertical_seamdp(E, img=STATE["img"])
    elif algo == "greedy":
        return greedy_vertical_seam(E)
    else:
        raise ValueError("algo must be 'dp' or 'greedy'")

## @brief Compute a horizontal seam using a chosen algorithm
## @param algo "dp" or "greedy"
## @return (seam, transposed_image)
def _compute_horizontal_seam(algo: str):
    tr = np.transpose(STATE["img"], (1, 0, 2))
    E  = sobel_energy(tr)
    if algo == "dp":
        seam = min_vertical_seamdp(E, img=tr)
    else:
        seam = greedy_vertical_seam(E)
    return seam, tr

## @brief Display a seam preview without modifying the image
## @param algo Algorithm name ("dp" or "greedy")
## @param vertical True = vertical seam, False = horizontal
def preview_seam(algo: str, vertical: bool):
    """Show next seam only (no modification)."""
    if vertical:
        seam = _compute_vertical_seam(algo)
        vis  = STATE["img"].copy()
        for i, j in enumerate(seam):
            vis[i, j] = (0, 0, 255)
        show_preview(vis)
        set_status(f"{algo.upper()} preview (vertical)")
    else:
        seam, tr = _compute_horizontal_seam(algo)
        vis_tr = tr.copy()
        for i, j in enumerate(seam):
            vis_tr[i, j] = (0, 0, 255)
        show_preview(np.transpose(vis_tr, (1, 0, 2)))
        set_status(f"{algo.upper()} preview (horizontal)")

## @brief Carve exactly one seam using the selected algorithm and direction
## @param algo Seam selection algorithm
## @param vertical True = vertical seam, False = horizontal
def carve_once_algo(algo: str, vertical: bool):
    """Carve 1 seam using the selected algo & orientation."""
    if vertical:
        if STATE["img"].shape[1] <= 2:
            set_status("Cannot carve: width too small")
            return
        seam = _compute_vertical_seam(algo)
        if STATE["preview_on"]:
            vis = STATE["img"].copy()
            for i, j in enumerate(seam): vis[i, j] = (0, 0, 255)
            show_preview(vis)
        STATE["img"] = remove_vertical_seamdp(STATE["img"], seam)
        set_status(f"{algo.upper()} carved 1 column")
    else:
        if STATE["img"].shape[0] <= 2:
            set_status("Cannot carve: height too small")
            return
        seam, tr = _compute_horizontal_seam(algo)
        if STATE["preview_on"]:
            vis_tr = tr.copy()
            for i, j in enumerate(seam): vis_tr[i, j] = (0, 0, 255)
            show_preview(np.transpose(vis_tr, (1, 0, 2)))
        tr = remove_vertical_seamdp(tr, seam)
        STATE["img"] = np.transpose(tr, (1, 0, 2))
        set_status(f"{algo.upper()} carved 1 row")
    refresh_texture()

## @brief Carve N seams using the selected algorithm and direction
## @param algo Algorithm ("dp" or "greedy")
## @param vertical Orientation flag 
## @param n_tag DPG widget tag containing seam count 
def carve_n_algo(algo: str, vertical: bool, n_tag: str):
    """Carve N seams using the selected algo & orientation. n_tag = DPG input tag."""
    import time
    n = max(0, int(dpg.get_value(n_tag)))
    if vertical:
        n = min(n, max(0, STATE["img"].shape[1] - 2))
        for i in range(n):
            seam = _compute_vertical_seam(algo)

            # Show seam in red on main image before carving
            vis = STATE["img"].copy()
            for row, col in enumerate(seam):
                vis[row, col] = (0, 0, 255)  # Red in BGR

            # Temporarily display the seam
            temp_img = STATE["img"]
            STATE["img"] = vis
            set_status(f"{algo.upper()} carving columns: {i+1}/{n} - showing seam")
            refresh_texture()
            time.sleep(0.1)  # Allow render cycle to show the seam

            # Now carve it
            STATE["img"] = remove_vertical_seamdp(temp_img, seam)

            # Show preview window if enabled
            if STATE["preview_on"]:
                show_preview(vis)

            # Update display after carve
            set_status(f"{algo.upper()} carving columns: {i+1}/{n}")
            refresh_texture()
            time.sleep(0.05)  # Allow render cycle to show the result
        set_status(f"{algo.upper()} carved {n} columns")
    else:
        n = min(n, max(0, STATE["img"].shape[0] - 2))
        for i in range(n):
            seam, tr = _compute_horizontal_seam(algo)

            # Show seam in red on main image before carving
            vis_tr = tr.copy()
            for row, col in enumerate(seam):
                vis_tr[row, col] = (0, 0, 255)  # Red in BGR
            vis = np.transpose(vis_tr, (1, 0, 2))

            # Temporarily display the seam
            temp_img = STATE["img"]
            STATE["img"] = vis
            set_status(f"{algo.upper()} carving rows: {i+1}/{n} - showing seam")
            refresh_texture()
            time.sleep(0.1)  # Allow render cycle to show the seam

            # Now carve it
            tr = remove_vertical_seamdp(tr, seam)
            STATE["img"] = np.transpose(tr, (1, 0, 2))

            # Show preview window if enabled
            if STATE["preview_on"]:
                show_preview(vis)

            # Update display after carve
            set_status(f"{algo.upper()} carving rows: {i+1}/{n}")
            refresh_texture()
            time.sleep(0.05)  # Allow render cycle to show the result
        set_status(f"{algo.upper()} carved {n} rows")
    refresh_texture()

# ---------- UI ----------
def toggle_preview():
    STATE["preview_on"] = not STATE["preview_on"]
    set_status(f"Seam preview: {STATE['preview_on']}")

## @brief Callback: preview a seam
## @param user_data (algo, vertical)
def preview_seam_cb(sender, app_data, user_data):
    algo, vertical = user_data            # e.g. ("dp", True)
    preview_seam(algo, vertical)

## @brief Callback: carve a single seam
## @param user_data (algo, vertical)
def carve_once_cb(sender, app_data, user_data):
    algo, vertical = user_data            # e.g. ("greedy", False)
    carve_once_algo(algo, vertical)

## @brief Callback: carve N seams
## @param user_data (algo, vertical, n_tag)
def carve_n_cb(sender, app_data, user_data):
    algo, vertical, n_tag = user_data     # e.g. ("dp", True, "n_cols_dp")
    carve_n_algo(algo, vertical, n_tag)

## @brief Callback: toggle preview window
def toggle_preview_cb(sender, app_data, user_data):
    toggle_preview()

## @brief Callback: load image from path
def load_cb(sender, app_data, user_data):
    load_image_from_path()

## @brief Callback: save image to disk
def save_cb(sender, app_data, user_data):
    save_image()

## @brief Callback: reset working image to original
def reset_cb(sender, app_data, user_data):
    (STATE.update(img=STATE['orig'].copy()) or set_status('Reset')) or refresh_texture()

## @brief Build and launch the complete DearPyGui UI
def build_ui():
    dpg.create_context()

    # load your image (or synth)
    img = cv2.imread(STATE["path"], cv2.IMREAD_COLOR)
    if img is None:
        X, Y = np.meshgrid(np.arange(480, dtype=np.uint8), np.arange(320, dtype=np.uint8))
        img = np.dstack((X%256, Y%256, 255 - X%256))
    STATE["img"] = img
    STATE["orig"] = img.copy()

    # create first texture/tag
    refresh_texture()

    with dpg.window(tag="main_win", label="Seam Carving with DP and Greedy"):
        dpg.add_text("", tag="status_text")
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="path_input", default_value=STATE["path"], width=450)
            dpg.add_button(label="Load",    callback=load_cb)
            dpg.add_button(label="Save PNG", callback=save_cb)
        dpg.add_spacer(height=6)

        # ---------- DP ----------
        dpg.add_text("Dynamic Programming (DP)")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Preview vertical", callback=preview_seam_cb, user_data=("dp", True))
            dpg.add_button(label="Carve 1 col",      callback=carve_once_cb,   user_data=("dp", True))
            dpg.add_input_int(tag="n_cols_dp", default_value=10, min_value=0, width=80)
            dpg.add_button(label="Carve N cols",     callback=carve_n_cb,      user_data=("dp", True,  "n_cols_dp"))
        with dpg.group(horizontal=True):
            dpg.add_button(label="Preview horizontal", callback=preview_seam_cb, user_data=("dp", False))
            dpg.add_button(label="Carve 1 row",        callback=carve_once_cb,   user_data=("dp", False))
            dpg.add_input_int(tag="n_rows_dp", default_value=5, min_value=0, width=80)
            dpg.add_button(label="Carve N rows",       callback=carve_n_cb,      user_data=("dp", False, "n_rows_dp"))

        dpg.add_spacer(height=8); dpg.add_separator(); dpg.add_spacer(height=6)

        # ---------- Greedy ----------
        dpg.add_text("Greedy")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Preview vertical", callback=preview_seam_cb, user_data=("greedy", True))
            dpg.add_button(label="Carve 1 col",      callback=carve_once_cb,   user_data=("greedy", True))
            dpg.add_input_int(tag="n_cols_greedy", default_value=10, min_value=0, width=80)
            dpg.add_button(label="Carve N cols",     callback=carve_n_cb,      user_data=("greedy", True,  "n_cols_greedy"))
        with dpg.group(horizontal=True):
            dpg.add_button(label="Preview horizontal", callback=preview_seam_cb, user_data=("greedy", False))
            dpg.add_button(label="Carve 1 row",        callback=carve_once_cb,   user_data=("greedy", False))
            dpg.add_input_int(tag="n_rows_greedy", default_value=5, min_value=0, width=80)
            dpg.add_button(label="Carve N rows",       callback=carve_n_cb,      user_data=("greedy", False, "n_rows_greedy"))

        dpg.add_spacer(height=6)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Toggle preview", callback=toggle_preview_cb)
            dpg.add_button(label="Reset",          callback=reset_cb)

        dpg.add_spacer(height=8)
        dpg.add_image(STATE["tex_tag"], tag=STATE["img_widget"])
        dpg.create_viewport(title="Seam Carving — DP + Greedy", width=1100, height=800)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_win", True)
        dpg.set_value("status_text", "Ready")
        dpg.start_dearpygui()
        dpg.destroy_context()


if __name__ == "__main__":
    build_ui()
