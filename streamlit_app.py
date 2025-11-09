import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
import torch

import streamlit as st
try:
    import cv2, ultralytics, torch, torchvision, numpy as np
    st.sidebar.write({"cv2": cv2.__version__,
                      "ultralytics": ultralytics.__version__,
                      "torch": torch.__version__,
                      "torchvision": torchvision.__version__,
                      "numpy": np.__version__})
except Exception as e:
    st.sidebar.error(f"import error: {e!r}")


st.set_page_config(page_title="Tree Detector - YOLOv8n & Faster R-CNN", layout="wide")

# ------------------- Rutas y config -------------------
YOLO_WEIGHTS = Path("models/yolov8n_trees.pt")
FRCNN_WEIGHTS = Path("models/fasterrcnn_trees.pt")  # .pt
LABELS_FILE = Path("labels.txt")

# limitar hilos cpu
try:
    torch.set_num_threads(1)
except Exception:
    pass

# ------------------- Utils -------------------
def normalize_class_names(names):
    """Asegura lista indexable 0..N-1 para nombres de clase."""
    if isinstance(names, dict):
        return [names[i] for i in range(len(names))]
    return list(names)

@st.cache_resource
def get_class_names():
    if LABELS_FILE.exists():
        lines = [ln.strip() for ln in LABELS_FILE.read_text(encoding="utf-8").splitlines()]
        return [ln for ln in lines if ln]
    return ["tree"]  # fallback

def draw_boxes_pil(img_pil, boxes, labels, scores, class_names, score_thr=0.25):
    img = img_pil.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    names = normalize_class_names(class_names)

    for (x1, y1, x2, y2), lab, sc in zip(boxes, labels, scores):
        if sc < score_thr:
            continue
        draw.rectangle([(float(x1), float(y1)), (float(x2), float(y2))],
                       outline=(0, 255, 0), width=3)
        name = names[int(lab)] if 0 <= int(lab) < len(names) else str(int(lab))
        caption = f"{name} {sc:.2f}"
        try:
            tw = draw.textlength(caption)
        except Exception:
            tw = 7 * len(caption)
        th = 14
        draw.rectangle([(x1, y1 - th - 4), (x1 + tw + 8, y1)], fill=(0, 255, 0))
        draw.text((x1 + 4, y1 - th - 2), caption, fill=(0, 0, 0))
    return img

def count_target(labels, class_names, target_name="tree"):
    names = normalize_class_names(class_names)
    names_lower = [n.lower() for n in names]
    if target_name in names_lower:
        idx = names_lower.index(target_name)
        return int(np.sum(np.array(labels) == idx))
    return int(len(labels))

def check_file(p: Path, label: str):
    if not p.exists():
        st.error(f"No se encontro {label}: {p}")
        st.stop()

# ------------------- YOLOv8n -------------------
@st.cache_resource
def load_yolo():
    from ultralytics import YOLO
    check_file(YOLO_WEIGHTS, "pesos YOLOv8n")
    model = YOLO(str(YOLO_WEIGHTS))
    return model

def yolo_infer(pil_img, conf=0.25):
    model = load_yolo()
    res = model.predict(pil_img, conf=conf, verbose=False)[0]
    if res.boxes is None:
        return np.empty((0, 4)), np.array([], dtype=int), np.array([], dtype=float), res.names or get_class_names()
    boxes = res.boxes.xyxy.cpu().numpy()
    labels = res.boxes.cls.cpu().numpy().astype(int)
    scores = res.boxes.conf.cpu().numpy()
    class_names = res.names if hasattr(res, "names") and res.names else get_class_names()
    return boxes, labels, scores, class_names

# ------------------- Faster R-CNN (.pt robusto) -------------------
@st.cache_resource
def load_frcnn():
    """
    Carga robusta para .pt:
      1) TorchScript (jit)
      2) Modelo completo (pickle)
      3) state_dict dict (o dict con 'model')
    """
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    check_file(FRCNN_WEIGHTS, "pesos Faster R-CNN (.pt)")
    device = torch.device("cpu")

    # 1) TorchScript
    try:
        model_ts = torch.jit.load(str(FRCNN_WEIGHTS), map_location=device)
        model_ts.eval()
        return ("scripted", model_ts)
    except Exception:
        pass

    # 2) torch.load: puede devolver modelo ya armado o state_dict
    obj = torch.load(str(FRCNN_WEIGHTS), map_location=device)

    # 2.a) si es un modelo listo (tiene .eval)
    if hasattr(obj, "eval") and callable(obj.eval):
        obj.eval()
        return ("full", obj)

    # 2.b) caso state_dict o dict con 'model'
    state_dict = obj["model"] if isinstance(obj, dict) and "model" in obj else obj

    # construir base sin pesos predescargados
    base = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None
    )
    in_feats = base.roi_heads.box_predictor.cls_score.in_features

    # +1 por background para torchvision
    num_classes = max(2, len(get_class_names()) + 1)
    base.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)

    # cargar state_dict (estricto=False para tolerar diferencias menores)
    base.load_state_dict(state_dict, strict=False)
    base.eval()
    return ("state", base)

def frcnn_infer(pil_img):
    import torchvision.transforms.functional as F
    kind, model = load_frcnn()
    img_t = F.to_tensor(pil_img).to("cpu")
    with torch.no_grad():
        out = model([img_t])[0]
    boxes = out.get("boxes", torch.empty((0, 4))).cpu().numpy()
    labels = out.get("labels", torch.empty((0,), dtype=torch.int64)).cpu().numpy().astype(int)
    scores = out.get("scores", torch.empty((0,), dtype=torch.float32)).cpu().numpy()
    # nombres: background + labels.txt
    class_names = ["__background__"] + get_class_names()
    return boxes, labels, scores, class_names

# ------------------- UI -------------------
st.title("Deteccion y conteo de arboles")
st.caption("Dos motores: YOLOv8n (Ultralytics) y Faster R-CNN (Torchvision). CPU only.")

tabs = st.tabs(["YOLOv8n", "Faster R-CNN"])

with tabs[0]:
    st.subheader("YOLOv8n")
    conf = st.slider("Confianza minima", 0.05, 0.95, 0.25, 0.05, key="yolo_conf")
    up = st.file_uploader("Sube una imagen (jpg/png)", type=["jpg", "jpeg", "png"], key="yolo_up")
    if up:
        img = Image.open(up).convert("RGB")
        t0 = time.time()
        boxes, labels, scores, names = yolo_infer(img, conf=conf)
        dt = time.time() - t0

        tree_count = count_target(labels, names, "tree")
        st.write(f"Detecciones totales: **{len(labels)}** | Arboles: **{tree_count}** | Tiempo: {dt:.2f}s")
        vis = draw_boxes_pil(img, boxes, labels, scores, names, score_thr=conf)
        st.image(vis, caption="Resultados YOLOv8n", use_column_width=True)

with tabs[1]:
    st.subheader("Faster R-CNN")
    thr = st.slider("Puntuacion minima", 0.05, 0.95, 0.25, 0.05, key="fr_thr")
    up2 = st.file_uploader("Sube una imagen (jpg/png)", type=["jpg", "jpeg", "png"], key="frcnn_up")
    if up2:
        img = Image.open(up2).convert("RGB")
        t0 = time.time()
        boxes, labels, scores, names = frcnn_infer(img)
        dt = time.time() - t0

        keep = scores >= thr
        kept_labels = np.array(labels)[keep]
        kept_boxes = np.array(boxes)[keep]
        kept_scores = np.array(scores)[keep]

        tree_count = count_target(kept_labels, names, "tree")
        st.write(f"Detecciones totales: **{int(keep.sum())}** | Arboles: **{tree_count}** | Tiempo: {dt:.2f}s")
        vis = draw_boxes_pil(img, kept_boxes, kept_labels, kept_scores, names, score_thr=thr)
        st.image(vis, caption="Resultados Faster R-CNN", use_column_width=True)

st.info("Tip: si tu clase no se llama 'tree', edita labels.txt (una clase por linea).")
