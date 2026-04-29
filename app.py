
import io
import math
import struct
import zipfile
from pathlib import Path

import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageStat, ImageOps
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    from rembg import remove as rembg_remove
except Exception:
    rembg_remove = None


DPI = 300

PHOTO_W_IN = 1.5
PHOTO_H_IN = 1.9
PHOTO_W_PX = int(PHOTO_W_IN * DPI)
PHOTO_H_PX = int(PHOTO_H_IN * DPI)

PAIR_W_IN = 1.9
PAIR_H_IN = 1.5
PAIR_W_PX = int(PAIR_W_IN * DPI)
PAIR_H_PX = int(PAIR_H_IN * DPI)

A4_W_IN = 8.27
A4_H_IN = 11.69
A4_W_PX = int(A4_W_IN * DPI)
A4_H_PX = int(A4_H_IN * DPI)

GAP_IN = 0.08
MARGIN_IN = 0.25


st.set_page_config(
    page_title="Passport Photo Proper v14",
    page_icon="🪪",
    layout="wide",
)


def hex_to_rgb(hex_color: str):
    hex_color = hex_color.strip().lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def gray_world_color_correction(img: Image.Image) -> Image.Image:
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    means = arr.reshape(-1, 3).mean(axis=0)
    gray_mean = means.mean()
    scale = gray_mean / np.maximum(means, 1.0)
    scale = np.clip(scale, 0.86, 1.18)
    corrected = arr * scale
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return Image.fromarray(corrected, "RGB")


def auto_enhance_for_print(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    img = gray_world_color_correction(img)
    img = ImageOps.autocontrast(img, cutoff=1)

    gray = img.convert("L")
    mean_luma = ImageStat.Stat(gray).mean[0]

    if mean_luma < 95:
        brightness = 1.24
    elif mean_luma < 120:
        brightness = 1.17
    elif mean_luma < 150:
        brightness = 1.11
    elif mean_luma > 210:
        brightness = 0.96
    else:
        brightness = 1.06

    img = ImageEnhance.Brightness(img).enhance(brightness)

    gray2 = img.convert("L")
    std_luma = ImageStat.Stat(gray2).stddev[0]

    if std_luma < 35:
        contrast = 1.18
    elif std_luma < 50:
        contrast = 1.12
    else:
        contrast = 1.07

    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(1.05)

    soft = img.filter(ImageFilter.GaussianBlur(radius=1.15))
    img = Image.blend(img, soft, 0.10)

    img = ImageEnhance.Sharpness(img).enhance(1.28)
    img = ImageEnhance.Brightness(img).enhance(1.04)
    img = ImageEnhance.Contrast(img).enhance(1.03)
    return img


def detect_face_box(img: Image.Image):
    if cv2 is None:
        return None

    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return tuple(int(v) for v in faces[0])


def crop_to_ratio(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    target_ratio = target_w / target_h
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        return img.crop((0, top, w, top + new_h))


def crop_to_passport_ratio(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    target_ratio = PHOTO_W_PX / PHOTO_H_PX
    face = detect_face_box(img)

    if face:
        x, y, fw, fh = face
        face_cx = x + fw / 2
        face_cy = y + fh / 2
        crop_h = int(fh * 3.2)
        crop_w = int(crop_h * target_ratio)

        if crop_w > w:
            crop_w = w
            crop_h = int(crop_w / target_ratio)
        if crop_h > h:
            crop_h = h
            crop_w = int(crop_h * target_ratio)

        left = int(face_cx - crop_w / 2)
        top = int(face_cy - crop_h * 0.38)
        left = max(0, min(left, w - crop_w))
        top = max(0, min(top, h - crop_h))
        return img.crop((left, top, left + crop_w, top + crop_h))

    return crop_to_ratio(img, PHOTO_W_PX, PHOTO_H_PX)


def replace_background(img: Image.Image, bg_rgb=(255, 255, 255), use_ai_bg=True) -> Image.Image:
    img = img.convert("RGB")
    if use_ai_bg and rembg_remove is not None:
        try:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            result = rembg_remove(buffer.getvalue())
            cutout = Image.open(io.BytesIO(result)).convert("RGBA")
            background = Image.new("RGBA", cutout.size, bg_rgb + (255,))
            background.alpha_composite(cutout)
            return background.convert("RGB")
        except Exception:
            return img
    return img


def prepare_passport_photo(uploaded_img, bg_rgb, use_ai_bg=True, auto_enhance=True):
    cropped = crop_to_passport_ratio(uploaded_img)
    resized = cropped.resize((PHOTO_W_PX, PHOTO_H_PX), Image.Resampling.LANCZOS)
    bg_changed = replace_background(resized, bg_rgb=bg_rgb, use_ai_bg=use_ai_bg)
    final_img = auto_enhance_for_print(bg_changed) if auto_enhance else bg_changed
    return final_img.resize((PHOTO_W_PX, PHOTO_H_PX), Image.Resampling.LANCZOS), bg_changed


def force_unified_pair_background(pair_img: Image.Image, bg_rgb) -> Image.Image:
    base = Image.new("RGB", pair_img.size, bg_rgb)
    base.paste(pair_img.convert("RGB"), (0, 0))
    return base


def make_pair_inside_one_landscape_photo(left_img: Image.Image, right_img: Image.Image, bg_rgb, auto_enhance=True) -> Image.Image:
    canvas = Image.new("RGB", (PAIR_W_PX, PAIR_H_PX), bg_rgb)
    half_w = PAIR_W_PX // 2

    left = crop_to_ratio(left_img, half_w, PAIR_H_PX).resize((half_w, PAIR_H_PX), Image.Resampling.LANCZOS)
    right = crop_to_ratio(right_img, PAIR_W_PX - half_w, PAIR_H_PX).resize((PAIR_W_PX - half_w, PAIR_H_PX), Image.Resampling.LANCZOS)

    canvas.paste(left, (0, 0))
    canvas.paste(right, (half_w, 0))
    canvas = force_unified_pair_background(canvas, bg_rgb)

    if auto_enhance:
        canvas = auto_enhance_for_print(canvas)

    return canvas.resize((PAIR_W_PX, PAIR_H_PX), Image.Resampling.LANCZOS)


def get_print_item(base_img: Image.Image, orientation: str) -> Image.Image:
    if orientation == "Landscape - rotate 90° clockwise":
        return base_img.rotate(-90, expand=True)
    return base_img


def fixed_single_max_per_row(orientation: str) -> int:
    return 4 if orientation == "Landscape - rotate 90° clockwise" else 5


def fixed_pair_max_per_row(orientation: str) -> int:
    return 4 if orientation == "Landscape - rotate 90° clockwise" else 5


def layout_info(base_img: Image.Image, orientation: str, items_per_row: int, max_cols: int):
    print_img = get_print_item(base_img, orientation)
    item_w, item_h = print_img.size
    gap = int(GAP_IN * DPI)
    margin = int(MARGIN_IN * DPI)

    cols = max(1, min(int(items_per_row), int(max_cols)))
    rows = max(1, int((A4_H_PX - 2 * margin + gap) // (item_h + gap)))
    per_page = cols * rows

    return {"cols": cols, "rows": rows, "per_page": per_page, "item_w": item_w, "item_h": item_h}


def create_a4_layout_images(base_img: Image.Image, copies: int, orientation: str, items_per_row: int, max_cols: int):
    print_img = get_print_item(base_img, orientation)
    expanded_items = [print_img] * copies
    return create_a4_layout_images_from_items(expanded_items, orientation, items_per_row, max_cols)


def create_a4_layout_images_from_items(print_items, orientation: str, items_per_row: int, max_cols: int):
    """
    Creates A4 portrait pages from a sequence of already-oriented print items.
    All items must have same size for clean grid layout.
    """
    if not print_items:
        return [Image.new("RGB", (A4_W_PX, A4_H_PX), "white")]

    item_w, item_h = print_items[0].size
    gap = int(GAP_IN * DPI)
    margin = int(MARGIN_IN * DPI)

    cols = max(1, min(int(items_per_row), int(max_cols)))
    rows = max(1, int((A4_H_PX - 2 * margin + gap) // (item_h + gap)))
    per_page = cols * rows

    content_w = cols * item_w + (cols - 1) * gap
    start_x = max(margin, (A4_W_PX - content_w) // 2)

    pages = []
    total_pages = math.ceil(len(print_items) / per_page)

    idx = 0
    for _ in range(total_pages):
        page = Image.new("RGB", (A4_W_PX, A4_H_PX), "white")
        draw = ImageDraw.Draw(page)

        for i in range(per_page):
            if idx >= len(print_items):
                break

            r = i // cols
            c = i % cols
            x = start_x + c * (item_w + gap)
            y = margin + r * (item_h + gap)

            page.paste(print_items[idx], (x, y))
            draw.rectangle([x, y, x + item_w - 1, y + item_h - 1], outline=(190, 190, 190), width=1)
            idx += 1

        pages.append(page)

    return pages


def create_mixed_a4_layout_images(print_items, items_per_row):
    """
    Places mixed output items into one A4 portrait canvas using the selected
    per-row count. If needed, horizontal gap/margin is auto-adjusted so 4
    landscape-size items can fit in one row without resizing the photos.
    """
    if not print_items:
        return [Image.new("RGB", (A4_W_PX, A4_H_PX), "white")]

    cols = max(1, int(items_per_row))
    items = [item.convert("RGB") for item in print_items]

    max_w = max(item.size[0] for item in items)
    max_h = max(item.size[1] for item in items)

    # Auto-fit horizontally. This fixes the issue where 4 selected items
    # wrapped to 3 because default margin/gap was slightly too large.
    side_margin = int(0.15 * DPI)
    available_gap_total = A4_W_PX - (2 * side_margin) - (cols * max_w)

    if cols > 1:
        h_gap = max(2, available_gap_total // (cols - 1))
    else:
        h_gap = 0

    # If still too wide, reduce side margin as a fallback.
    if available_gap_total < 0:
        side_margin = max(5, (A4_W_PX - cols * max_w) // 2)
        h_gap = 2 if cols > 1 else 0

    v_gap = int(GAP_IN * DPI)
    top_margin = int(MARGIN_IN * DPI)

    row_step = max_h + v_gap
    rows_per_page = max(1, (A4_H_PX - 2 * top_margin + v_gap) // row_step)
    per_page = cols * rows_per_page

    pages = []
    idx = 0
    total_pages = math.ceil(len(items) / per_page)

    for _ in range(total_pages):
        page = Image.new("RGB", (A4_W_PX, A4_H_PX), "white")
        draw = ImageDraw.Draw(page)

        for slot in range(per_page):
            if idx >= len(items):
                break

            r = slot // cols
            c = slot % cols

            item = items[idx]
            w, h = item.size

            cell_x = side_margin + c * (max_w + h_gap)
            x = cell_x + (max_w - w) // 2
            y = top_margin + r * row_step + (max_h - h) // 2

            page.paste(item, (x, y))
            draw.rectangle([x, y, x + w - 1, y + h - 1], outline=(190, 190, 190), width=1)

            idx += 1

        pages.append(page)

    return pages


def make_pdf_from_pages(pages):
    output = io.BytesIO()
    pages[0].save(output, format="PDF", resolution=DPI, save_all=True, append_images=pages[1:])
    return output.getvalue()


def make_a4_pdf(base_img, copies, orientation, items_per_row, max_cols):
    pages = create_a4_layout_images(base_img, copies, orientation, items_per_row, max_cols)
    return make_pdf_from_pages(pages)


def make_flat_psd_bytes(img: Image.Image) -> bytes:
    img = img.convert("RGB")
    width, height = img.size
    arr = np.asarray(img, dtype=np.uint8)

    out = io.BytesIO()
    out.write(b"8BPS")
    out.write(struct.pack(">H", 1))
    out.write(b"\x00" * 6)
    out.write(struct.pack(">H", 3))
    out.write(struct.pack(">I", height))
    out.write(struct.pack(">I", width))
    out.write(struct.pack(">H", 8))
    out.write(struct.pack(">H", 3))
    out.write(struct.pack(">I", 0))
    out.write(struct.pack(">I", 0))
    out.write(struct.pack(">I", 0))
    out.write(struct.pack(">H", 0))
    out.write(arr[:, :, 0].tobytes())
    out.write(arr[:, :, 1].tobytes())
    out.write(arr[:, :, 2].tobytes())
    return out.getvalue()


def make_a4_psd(base_img, copies, orientation, items_per_row, max_cols):
    pages = create_a4_layout_images(base_img, copies, orientation, items_per_row, max_cols)
    return make_flat_psd_bytes(pages[0])


def png_bytes(img: Image.Image) -> bytes:
    out = io.BytesIO()
    img.save(out, format="PNG", dpi=(DPI, DPI))
    return out.getvalue()


def safe_name(name: str) -> str:
    stem = Path(name).stem
    keep = "".join(c if c.isalnum() or c in "-_ " else "_" for c in stem).strip()
    return keep or "photo"


st.title("🪪 Passport Photo Proper v14")
st.caption("New: all uploaded photos can be placed into one shared A4 canvas.")

st.info(
    f"Single photo: {PHOTO_W_IN} × {PHOTO_H_IN} inch ({PHOTO_W_PX}×{PHOTO_H_PX}px). "
    f"Pair photo: {PAIR_W_IN} × {PAIR_H_IN} inch ({PAIR_W_PX}×{PAIR_H_PX}px). A4 page fixed Portrait."
)

with st.sidebar:
    st.header("⚙️ Global Settings")

    bg_option = st.selectbox("Background color", ["White", "Blue", "Light gray", "Custom"], index=0)
    if bg_option == "White":
        bg_hex = "#FFFFFF"
    elif bg_option == "Blue":
        bg_hex = "#7DB9E8"
    elif bg_option == "Light gray":
        bg_hex = "#F2F2F2"
    else:
        bg_hex = st.color_picker("Custom background", "#FFFFFF")

    bg_rgb = hex_to_rgb(bg_hex)

    use_ai_bg = st.checkbox("AI background remove/change", value=True)
    auto_enhance = st.checkbox("Auto photo enhancement ON/OFF", value=False)

    st.divider()
    st.header("🧾 Canvas Mode")
    combine_all_single = st.checkbox(
        "সব output একই A4 canvas-এ রাখুন",
        value=True,
        help="ON হলে single photo এবং pair/combined photo সব একই PDF/PSD canvas sequence-এ বসবে। OFF হলে প্রতিটি output আলাদা PDF/PSD হবে।"
    )

    st.divider()
    st.header("🖼️ Single photo layout")
    single_orientation = st.radio(
        "Single photo orientation",
        ["Portrait - normal", "Landscape - rotate 90° clockwise"],
        index=1,
    )
    single_max = fixed_single_max_per_row(single_orientation)
    single_default = 5 if single_orientation == "Portrait - normal" else 4
    single_per_row = st.number_input(
        "Single photo: ১ রোতে কয়টি?",
        min_value=1,
        max_value=single_max,
        value=single_default,
        step=1,
    )
    st.caption(f"Single photo max: {single_max} per row")

    st.divider()
    st.header("👥 Pair photo")
    enable_pair = st.checkbox("দুটি ছবি মিলে একটি 1.9×1.5 inch pair photo তৈরি করুন", value=False)

uploaded_files = st.file_uploader(
    "এক বা একাধিক ছবি upload করুন",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.warning("প্রথমে ছবি upload করুন।")
    st.stop()

st.subheader("🔢 প্রতিটি ছবির copy সংখ্যা")
copy_inputs = {}
cols = st.columns(2)
for idx, uploaded in enumerate(uploaded_files):
    with cols[idx % 2]:
        copy_inputs[uploaded.name] = st.number_input(
            f"{uploaded.name} - কত কপি?",
            min_value=0,
            max_value=300,
            value=4,
            step=1,
            key=f"copy_{idx}_{uploaded.name}",
        )

pair_settings = None
if enable_pair:
    st.subheader("👥 Pair photo settings")
    if len(uploaded_files) < 2:
        st.error("Pair photo করতে কমপক্ষে ২টি ছবি upload করতে হবে।")
    else:
        names = [u.name for u in uploaded_files]
        c1, c2, c3 = st.columns(3)
        with c1:
            left_name = st.selectbox("Left person", names, index=0)
        with c2:
            right_name = st.selectbox("Right person", names, index=1 if len(names) > 1 else 0)
        with c3:
            pair_copies = st.number_input("Pair photo কত কপি?", min_value=0, max_value=300, value=4, step=1)

        pair_orientation = st.radio(
            "Pair photo placement on A4",
            ["Portrait - normal", "Landscape - rotate 90° clockwise"],
            index=1,
            horizontal=True,
        )

        pair_max = fixed_pair_max_per_row(pair_orientation)
        pair_default = 5 if pair_orientation == "Portrait - normal" else 4
        pair_per_row = st.number_input(
            "Pair photo: ১ রোতে কয়টি?",
            min_value=1,
            max_value=pair_max,
            value=pair_default,
            step=1,
        )

        dummy_pair = Image.new("RGB", (PAIR_W_PX, PAIR_H_PX), bg_rgb)
        pair_info = layout_info(dummy_pair, pair_orientation, int(pair_per_row), pair_max)
        st.info(f"Pair layout: {pair_info['cols']} per row × {pair_info['rows']} rows = {pair_info['per_page']} pair photos per A4.")

        pair_settings = {
            "left": left_name,
            "right": right_name,
            "copies": int(pair_copies),
            "orientation": pair_orientation,
            "per_row": int(pair_per_row),
            "max": pair_max,
        }

processed = {}
progress = st.progress(0)

for idx, uploaded in enumerate(uploaded_files, start=1):
    original = Image.open(uploaded).convert("RGB")
    final, before_enhance = prepare_passport_photo(original, bg_rgb, use_ai_bg, auto_enhance)
    processed[uploaded.name] = {
        "original": original,
        "before": before_enhance,
        "final": final,
    }
    progress.progress(idx / len(uploaded_files))

st.success(f"{len(processed)} টি ছবি processed হয়েছে।")

outputs = []
all_canvas_print_items = []
single_print_items = []
single_names = []

for name, item in processed.items():
    copies = int(copy_inputs[name])
    if copies <= 0:
        continue

    print_img = get_print_item(item["final"], single_orientation)

    if combine_all_single:
        for _ in range(copies):
            all_canvas_print_items.append(print_img)
            single_print_items.append(print_img)
            single_names.append(name)
    else:
        info = layout_info(item["final"], single_orientation, int(single_per_row), single_max)
        outputs.append({
            "name": name,
            "label": name,
            "base": item["final"],
            "preview": print_img,
            "copies": copies,
            "layout": info,
            "pdf": make_a4_pdf(item["final"], copies, single_orientation, int(single_per_row), single_max),
            "psd": make_a4_psd(item["final"], copies, single_orientation, int(single_per_row), single_max),
            "png": png_bytes(item["final"]),
        })

if enable_pair and pair_settings and pair_settings["copies"] > 0:
    if pair_settings["left"] == pair_settings["right"]:
        st.warning("Pair photo-এর জন্য দুইটি আলাদা ছবি select করুন।")
    else:
        pair_img = make_pair_inside_one_landscape_photo(
            processed[pair_settings["left"]]["final"],
            processed[pair_settings["right"]]["final"],
            bg_rgb=bg_rgb,
            auto_enhance=auto_enhance,
        )
        pair_name = f"PAIR_{safe_name(pair_settings['left'])}_AND_{safe_name(pair_settings['right'])}"
        info = layout_info(pair_img, pair_settings["orientation"], pair_settings["per_row"], pair_settings["max"])
        if combine_all_single:
            pair_print_img = get_print_item(pair_img, pair_settings["orientation"])
            for _ in range(pair_settings["copies"]):
                all_canvas_print_items.append(pair_print_img)
        else:
            outputs.append({
                "name": pair_name,
                "label": f"Pair: {pair_settings['left']} + {pair_settings['right']}",
                "base": pair_img,
                "preview": get_print_item(pair_img, pair_settings["orientation"]),
                "copies": pair_settings["copies"],
                "layout": info,
                "pdf": make_a4_pdf(pair_img, pair_settings["copies"], pair_settings["orientation"], pair_settings["per_row"], pair_settings["max"]),
                "psd": make_a4_psd(pair_img, pair_settings["copies"], pair_settings["orientation"], pair_settings["per_row"], pair_settings["max"]),
                "png": png_bytes(pair_img),
            })


if combine_all_single and all_canvas_print_items:
    pages = create_mixed_a4_layout_images(all_canvas_print_items, int(single_per_row))
    outputs = [{
        "name": "ALL_OUTPUTS_COMBINED",
        "label": f"All outputs combined ({len(all_canvas_print_items)} total items)",
        "base": pages[0],
        "preview": pages[0],
        "copies": len(all_canvas_print_items),
        "layout": {"cols": int(single_per_row), "rows": "auto", "per_page": "mixed"},
        "pdf": make_pdf_from_pages(pages),
        "psd": make_flat_psd_bytes(pages[0]),
        "png": png_bytes(pages[0]),
    }]

tab1, tab2, tab3 = st.tabs(["Preview", "Downloads", "Batch ZIP"])

with tab1:
    st.subheader("Processed single photos")
    for name, item in processed.items():
        st.markdown(f"### {name}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("Original")
            st.image(item["original"], use_container_width=True)
        with c2:
            st.caption("Background/crop")
            st.image(item["before"], use_container_width=False)
        with c3:
            st.caption("Final")
            st.image(item["final"], use_container_width=False)

    st.subheader("Output items")
    for out in outputs:
        st.markdown(f"### {out['label']}")
        st.caption(f"Copies/items: {out['copies']} | Layout: {out['layout']['cols']} per row × {out['layout']['rows']} rows = {out['layout']['per_page']} per A4")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Base item / first canvas")
            st.image(out["base"], use_container_width=False)
        with c2:
            st.caption("Placement preview")
            st.image(out["preview"], use_container_width=False)

with tab2:
    if not outputs:
        st.warning("Download করার জন্য অন্তত একটি copy সংখ্যা ১ বা তার বেশি দিন।")
    for out in outputs:
        name = safe_name(out["name"])
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(f"⬇️ PDF: {name}", out["pdf"], f"{name}_A4_print.pdf", "application/pdf", key=f"pdf_{name}")
        with c2:
            st.download_button(f"⬇️ PSD: {name}", out["psd"], f"{name}_A4_print.psd", "image/vnd.adobe.photoshop", key=f"psd_{name}")
        with c3:
            st.download_button(f"⬇️ PNG: {name}", out["png"], f"{name}_item_or_first_canvas.png", "image/png", key=f"png_{name}")

with tab3:
    if outputs:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
            for out in outputs:
                name = safe_name(out["name"])
                z.writestr(f"{name}_A4_print.pdf", out["pdf"])
                z.writestr(f"{name}_A4_print.psd", out["psd"])
                z.writestr(f"{name}_item_or_first_canvas.png", out["png"])

        st.download_button(
            "⬇️ Download all PDFs + PSDs + PNGs as ZIP",
            data=zip_buffer.getvalue(),
            file_name="passport_photo_outputs_v14.zip",
            mime="application/zip",
        )
