# app/app.py
import os, pathlib

if os.name == "nt":  # only on Windows
    # Map any PosixPath encountered during unpickling to WindowsPath
    # happens due to training on kaggle
    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[attr-defined]

from pathlib import Path
import tempfile
import time

import streamlit as st
import pandas as pd
from PIL import Image
import cv2



# ---- LABEL SIZE OVERLAY ----
def draw_yolo_overlays(
        img_path: Path,
        label_paths: list[Path],
        class_names: list[str] | None,
        display_size: tuple[int, int],  # (H, W) you want to show in the app
        box_thickness: int = 2,
        font_scale: float = 0.6,
        font_thickness: int = 1,
):
    # load + resize for display
    im = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    orig_h, orig_w = im.shape[:2]
    disp_h, disp_w = display_size
    im_disp = cv2.resize(im, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

    overlay = im_disp.copy()

    # simple fixed palette
    palette = [
        (255, 99, 71),  # tomato
        (30, 144, 255),  # dodgerblue
        (60, 179, 113),  # mediumseagreen
        (238, 130, 238),  # violet
        (255, 215, 0),  # gold
    ]

    def denorm(xc, yc, bw, bh):
        # convert YOLO norm (0..1) → pixels on display image
        x1 = int((xc - bw / 2) * disp_w)
        y1 = int((yc - bh / 2) * disp_h)
        x2 = int((xc + bw / 2) * disp_w)
        y2 = int((yc + bh / 2) * disp_h)
        return max(0, x1), max(0, y1), min(disp_w - 1, x2), min(disp_h - 1, y2)

    for lp in label_paths:
        for line in lp.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:5])
            conf = None
            if len(parts) >= 6:
                try:
                    conf = float(parts[5])
                except:
                    conf = None

            x1, y1, x2, y2 = denorm(xc, yc, bw, bh)
            color = palette[cls % len(palette)]

            # draw border
            cv2.rectangle(im_disp, (x1, y1), (x2, y2), color, box_thickness)

            # label text
            label = class_names[cls] if class_names and 0 <= cls < len(class_names) else str(cls)
            if conf is not None:
                label = f"{label} {conf:.2f}"

            # text background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            tx1, ty1 = x1, max(0, y1 - th - 6)
            tx2, ty2 = x1 + tw + 6, y1
            cv2.rectangle(im_disp, (tx1, ty1), (tx2, ty2), color, -1)
            cv2.putText(im_disp, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    return im_disp


# ---- CONFIG ----
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "best.pt"
ASSETS = REPO_ROOT / "app" / "assets"

# importing YOLO from ultralytics

from ultralytics import YOLO

@st.cache_resource(show_spinner=True)
def load_model(weights: str):
    # Ultralytics auto-selects GPU if available; you can still pass device in predict()
    return YOLO(weights)

model = load_model(str(DEFAULT_WEIGHTS))

# ---- SIDEBAR ----
st.sidebar.title("⚙️ Settings")
weights_path = st.sidebar.text_input("Weights (.pt)", str(DEFAULT_WEIGHTS))
imgsz = st.sidebar.selectbox("Image size", [768, 640, 512, 416], index=0)
conf_thres = st.sidebar.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.01)  # third augment is default
iou_thres = st.sidebar.slider("NMS IoU", 0.10, 0.90, 0.50, 0.01)  # a tad higher
use_tta = st.sidebar.checkbox("Test-time augmentation (slower, more recall)", value=True)

st.title("♻️ Recyclables Detection — YOLOv11")
st.write("Upload an image; the model will draw boxes and list predictions with confidence.")

st.sidebar.markdown(f"**Using weights:** `{weights_path}`")
names_path = REPO_ROOT / "data" / "names.txt"
if names_path.exists():
    st.sidebar.markdown("**Classes:** " + ", ".join([n.strip() for n in names_path.read_text().splitlines()]))
else:
    st.sidebar.warning("names.txt not found; class table may be blank.")

# ---- LEFT: UPLOAD & RUN ----
col1, col2 = st.columns([2, 1], gap="large")

imgsz_pair = (int(imgsz), int(imgsz))

with col1:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        # Save to a temp file
        tmpdir = Path(tempfile.mkdtemp())
        src_path = tmpdir / uploaded.name
        src_path.write_bytes(uploaded.read())

        # Run YOLO detect.py
        out_root = REPO_ROOT / "app" / "tmp"
        out_root.mkdir(parents=True, exist_ok=True)
        run_name = f"infer_{int(time.time())}"

        # Use Ultralytics predict instead
        results = model.predict(
            source=str(src_path),
            imgsz=int(imgsz),  # single int is fine
            conf=conf_thres,  # confidence threshold
            iou=iou_thres,  # NMS IoU
            save=False,  # don't save rendered images (like nosave=True)
            save_txt=True,  # write YOLO .txt labels
            save_conf=True,  # include confidences in .txt
            project=str(out_root),  # same output root you already use
            name=run_name,  # keep your run folder layout
            exist_ok=True,
            device="cpu",  # or 0 to use GPU if available
            augment=use_tta,  # TTA-ish; Ultralytics uses 'augment' for predict
            agnostic_nms=False,
            max_det=300,
            line_thickness=2,  # only used if save=True (rendered imgs)
        )

        # Find the saved result image
        exp_dir = out_root / run_name

        txts = list(exp_dir.rglob("*.txt"))

        # build the display image using our renderer
        names = [n.strip() for n in names_path.read_text().splitlines()] if names_path.exists() else None

        # display at the chosen imgsz (square):
        H = W = int(imgsz)
        rendered = draw_yolo_overlays(
            img_path=src_path,
            label_paths=txts,
            class_names=names,
            display_size=(H, W),
            box_thickness=2,  # tweak in sidebar if you like
            font_scale=0.6,
            font_thickness=1,
        )

        st.image(Image.fromarray(rendered), caption="Detections", use_container_width=True)

        # detect.py writes an image with the same filename in exp_dir
        result_img_path = exp_dir / uploaded.name
        # Sometimes detect.py saves under 'exp' then subfolders; fallback:
        if not result_img_path.exists():
            # Grab first image in exp_dir
            imgs = list(exp_dir.rglob("*.jpg")) + list(exp_dir.rglob("*.png"))
            result_img_path = imgs[0] if imgs else None

        # Build a small table from labels txt if present
        txts = list(exp_dir.rglob("*.txt"))
        rows = []
        names_path = REPO_ROOT / "data" / "names.txt"
        names = [n.strip() for n in names_path.read_text().splitlines()] if names_path.exists() else None

        for t in txts:
            for line in t.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) >= 6:
                    cls = int(parts[0])
                    conf = float(parts[-1]) if parts[-1].replace('.', '', 1).isdigit() else None
                    label = names[cls] if names and 0 <= cls < len(names) else str(cls)
                    rows.append({"label": label, "conf": conf})
        if rows:
            df = pd.DataFrame(rows).sort_values("conf", ascending=False)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No boxes above threshold.")

# ---- RIGHT: EVAL VISUALS ----
with col2:
    st.subheader("Evaluation Visuals")
    for fname, caption in [
        ("BoxPR_curve.png", "Precision–Recall (per class, mAP@0.5)"),
        ("confusion_matrix.png", "Confusion Matrix (Predicted vs True + background)"),
        ("results.png", "Training Curves (loss, precision, recall, mAP)"),
    ]:
        p = ASSETS / fname
        if p.exists():
            st.image(str(p), caption=caption, use_container_width=True)
        else:
            st.caption(f"Missing: {fname} (place in app/assets/)")

st.markdown("---")