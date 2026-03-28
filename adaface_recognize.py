import os, sys, cv2, json, importlib.util
import numpy as np
import torch
from datetime import datetime
from collections import deque, Counter
from pathlib import Path

from ultralytics import YOLO
from insightface.app import FaceAnalysis
from insightface.utils import face_align

ENC_FILE = "students.npz"
MODEL_YOLO = "yolov8m-face.pt"
MODEL_DIR = "adaface_model"
VIDEO_PATH = r"C:\Users\TIH48\Desktop\face recognition\testing.mp4"
CTX_ID = 0
UNKNOWN_DIR = "unknown_faces_adaface"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SIM_FRONT = 0.38
SIM_BACK = 0.32
SIM_FLOOR = 0.28
TOP2_GAP_FRONT = 0.06
TOP2_GAP_BACK = 0.03
BESTGUESS_THR = 0.22

ZONE_SPLIT_RATIO = 0.55

MIN_FACE_PX_FRONT = 1600
MIN_FACE_PX_BACK = 300
MIN_EMB_NORM = 3.0
DET_SCORE_FRONT = 0.65
DET_SCORE_BACK = 0.40
MAX_YAW = 55
MAX_PITCH = 38

TILE_CONFIGS = [(1,1,0.00), (2,2,0.20), (3,3,0.20)]
TILE_IMGSZ = 640
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.50
NMS_IOU = 0.38
DETECT_EVERY = 1

PAD_RATIO = 0.35
MAX_UPSCALE = 4.0
USE_CLAHE = True

REIDENTIFY_EVERY = 12
SMOOTH_WINDOW = 7
CONFIRM_VOTES = 3
RECENT_FRAME_WIN = 30
UNK_SAVE_COOL = 60
DEBUG_PRINT = True

os.makedirs(UNKNOWN_DIR, exist_ok=True)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))


print("Loading embeddings ...")
if not Path(ENC_FILE).exists():
    print(f"ERROR: {ENC_FILE} not found.")
    print("Run: python adaface_build_embeddings.py --encode")
    sys.exit(1)

data = np.load(ENC_FILE, allow_pickle=True)
raw_emb = data["embeddings"].astype(np.float32)
raw_nm = data["names"].astype(str)

cent_embs, cent_names = [], []
for person in list(dict.fromkeys(raw_nm)):
    vecs = raw_emb[raw_nm == person]
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    centre = vecs.mean(axis=0)
    norm = np.linalg.norm(centre)
    if norm < 1e-6:
        continue
    cent_embs.append(centre / norm)
    cent_names.append(person)

known_embeddings = np.stack(cent_embs).astype(np.float32)
known_names = np.array(cent_names, dtype=str)
print(f"  {len(raw_nm)} raw embeddings -> {len(known_names)} people\n")


def _make_cfg(d):
    class Cfg:
        pass
    obj = Cfg()
    for k, v in d.items():
        setattr(obj, k, _make_cfg(v) if isinstance(v, dict) else v)
    obj.__dict__.update(d)
    return obj


def load_adaface():
    mp = Path(MODEL_DIR).absolute()

    required = [
        mp / "models" / "__init__.py",
        mp / "pretrained_model" / "model.pt",
        mp / "pretrained_model" / "model.yaml",
    ]
    for p in required:
        if not p.exists():
            print(f"ERROR: Missing {p}")
            print("Run: python adaface_build_embeddings.py --download")
            sys.exit(1)

    print(f"Loading AdaFace from {mp} ...")

    orig = os.getcwd()
    os.chdir(str(mp))
    sys.path.insert(0, str(mp))

    try:
        from omegaconf import OmegaConf
        from models import get_model

        model_conf = OmegaConf.load("pretrained_model/model.yaml")
        model = get_model(model_conf)

        ckpt = torch.load("pretrained_model/model.pt",
                          map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict):
            sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
        else:
            sd = ckpt

        for prefix in ("model.", "module.", "backbone."):
            sample_keys = list(sd.keys())[:5]
            if all(k.startswith(prefix) for k in sample_keys):
                sd = {k[len(prefix):]: v for k, v in sd.items()}

        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  missing keys ({len(missing)}): {missing[:2]}")
        if unexpected:
            print(f"  unexpected keys ({len(unexpected)}): {unexpected[:2]}")

    finally:
        os.chdir(orig)

    model = model.to(DEVICE).eval()
    print(f"  AdaFace IR-101 ready on {DEVICE}")
    return model

adaface = load_adaface()
yolo = YOLO(MODEL_YOLO)
detector = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
detector.prepare(ctx_id=CTX_ID, det_size=(640, 640))
print("  YOLO + InsightFace detector ready\n")


def enhance(bgr):
    if not USE_CLAHE:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


def get_aligned(bgr, target=112):
    for scale in [1.0, 1.5, 2.0, 3.0, 4.0]:
        img = cv2.resize(bgr, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_LANCZOS4) if scale > 1.01 else bgr.copy()
        img = enhance(img)
        faces = detector.get(img)
        if not faces:
            continue
        face = max(faces,
                   key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        if min(face.bbox[2]-face.bbox[0], face.bbox[3]-face.bbox[1]) < 20:
            continue
        aligned = face_align.norm_crop(img, landmark=face.kps, image_size=target)
        return aligned, getattr(face, "det_score", 1.0), getattr(face, "pose", None)
    return None, 0.0, None


@torch.no_grad()
def adaface_emb(aligned_bgr):
    t = torch.from_numpy(aligned_bgr.astype(np.float32))
    t = t.permute(2, 0, 1).unsqueeze(0)
    t = (t / 255.0 - 0.5) / 0.5
    t = t.to(DEVICE)
    out = adaface(t)
    if isinstance(out, (tuple, list)):
        emb, norm = out[0], out[1]
    else:
        emb = out
        norm = torch.norm(emb, dim=1)
    return (emb.squeeze().cpu().numpy().astype(np.float32),
            float(norm.squeeze().cpu().item()))


def get_tiles(frame, rows, cols, overlap):
    h, w = frame.shape[:2]
    sy, sx = h // rows, w // cols
    py, px = int(sy * overlap), int(sx * overlap)
    tiles = []
    for r in range(rows):
        for c in range(cols):
            y1 = max(0, r*sy - py);  y2 = min(h, (r+1)*sy + py)
            x1 = max(0, c*sx - px);  x2 = min(w, (c+1)*sx + px)
            tiles.append((frame[y1:y2, x1:x2], x1, y1))
    return tiles


def run_yolo_tile(tile, xo, yo):
    res = yolo(tile[..., ::-1], imgsz=TILE_IMGSZ,
               conf=DEFAULT_CONF, iou=DEFAULT_IOU, verbose=False)
    boxes = []
    for r in res:
        for box in r.boxes:
            c = float(box.conf.cpu().numpy())
            if c < DEFAULT_CONF:
                continue
            tx1, ty1, tx2, ty2 = map(int, box.xyxy[0].cpu().numpy())
            boxes.append((tx1+xo, ty1+yo, tx2+xo, ty2+yo, c))
    return boxes


def nms(boxes, thr):
    if not boxes:
        return []
    arr = np.array([[b[0],b[1],b[2],b[3]] for b in boxes], np.float32)
    sc = np.array([b[4] for b in boxes], np.float32)
    x1, y1, x2, y2 = arr[:,0], arr[:,1], arr[:,2], arr[:,3]
    areas = (x2-x1) * (y2-y1)
    order = sc.argsort()[::-1];  keep = []
    while order.size:
        i = order[0];  keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= thr]
    return [boxes[k] for k in keep]


def do_match(emb_unit, face_area, emb_norm, is_back):
    sims = known_embeddings.dot(emb_unit)
    idx = np.argsort(sims)[::-1]
    b_sim = float(sims[idx[0]])
    s_sim = float(sims[idx[1]]) if len(sims) > 1 else 0.0
    gap = b_sim - s_sim
    base = SIM_BACK if is_back else SIM_FRONT
    na = 600 if is_back else 3500
    sf = max(0.3, min(1.0, np.sqrt(face_area / na)))
    qf = max(0.8, min(1.2, emb_norm / 20.0))
    thr = max(SIM_FLOOR, (base + 0.10*(1-sf)) * qf)
    gthr = TOP2_GAP_BACK if is_back else TOP2_GAP_FRONT
    ok = b_sim >= thr and gap >= gthr
    return known_names[idx[0]], b_sim, gap, ok, thr


def smooth(hist, name, sim):
    hist.append((name, sim))
    real = [n for n, _ in hist if n != "Unknown"]
    if not real:
        return "Unknown", True
    top, cnt = Counter(real).most_common(1)[0]
    if cnt >= CONFIRM_VOTES:
        return top, True
    if len(hist) >= SMOOTH_WINDOW:
        return "Unknown", True
    return "Confirming...", False


def draw_label(img, text, x1, y1, color):
    f, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    (tw, lh), bl = cv2.getTextSize(text, f, fs, th)
    ty = max(lh+6, y1-4)
    cv2.rectangle(img, (x1, ty-lh-4), (x1+tw+4, ty+bl), (0,0,0), -1)
    cv2.putText(img, text, (x1+2, ty), f, fs, color, th, cv2.LINE_AA)


recognized_faces = {}
recent_known = {}
track_last = {}
track_hist = {}
unk_saved = {}


def process(frame, disp, x1, y1, x2, y2, track_id, frame_id, H, W):
    cy = (y1 + y2) / 2
    is_back = cy < H * ZONE_SPLIT_RATIO
    fa = (x2-x1) * (y2-y1)
    min_px = MIN_FACE_PX_BACK if is_back else MIN_FACE_PX_FRONT
    det_min = DET_SCORE_BACK if is_back else DET_SCORE_FRONT

    if fa < min_px * 0.4:
        cv2.rectangle(disp, (x1,y1), (x2,y2), (0,200,200), 1)
        draw_label(disp, "TooFar", x1, y1, (0,200,200))
        return

    pad = int(PAD_RATIO * (y2-y1))
    ax1 = max(0, x1-pad);   ay1 = max(0, y1-pad)
    ax2 = min(W-1, x2+pad); ay2 = min(H-1, y2+pad)
    crop = frame[ay1:ay2, ax1:ax2]
    if crop.size == 0:
        return

    cw, ch = crop.shape[1], crop.shape[0]
    sc = min(MAX_UPSCALE, 224 / max(min(cw, ch), 1))
    if sc > 1.05:
        crop = cv2.resize(crop, None, fx=sc, fy=sc,
                          interpolation=cv2.INTER_LANCZOS4)

    aligned, det_score, pose = get_aligned(crop)
    if aligned is None:
        return

    if det_score < det_min:
        if DEBUG_PRINT:
            print(f"Frame {frame_id:05d} | skip det_score={det_score:.2f}")
        return

    if pose is not None:
        yaw, pitch, _ = pose
        if abs(yaw) > MAX_YAW or abs(pitch) > MAX_PITCH:
            if DEBUG_PRINT:
                print(f"Frame {frame_id:05d} | skip pose yaw={yaw:.0f}")
            cv2.rectangle(disp, (ax1,ay1), (ax2,ay2), (128,0,128), 1)
            draw_label(disp, f"Angle({yaw:.0f})", ax1, ay1, (128,0,128))
            return

    emb, emb_norm = adaface_emb(aligned)
    if emb_norm < MIN_EMB_NORM:
        return

    emb_unit = emb / (emb_norm + 1e-8)
    best_name, best_sim, gap, accepted, thr = do_match(emb_unit, fa, emb_norm, is_back)

    zone = "BACK " if is_back else "FRONT"
    if DEBUG_PRINT:
        print(f"Frame {frame_id:05d} [{zone}] tid={track_id:5d} | "
              f"{best_name:<20s} sim={best_sim:.4f} thr={thr:.4f} "
              f"gap={gap:.4f} {'Y' if accepted else 'N'}")

    if track_id not in track_hist:
        track_hist[track_id] = deque(maxlen=SMOOTH_WINDOW)
    vote = best_name if accepted else "Unknown"
    final_name, conf = smooth(track_hist[track_id], vote, best_sim)
    track_last[track_id] = (final_name, frame_id, best_sim)

    if conf and final_name not in ("Unknown", "Confirming..."):
        if (final_name not in recent_known or
                frame_id - recent_known.get(final_name, 0) > RECENT_FRAME_WIN):
            recent_known[final_name] = frame_id
            recognized_faces.setdefault(final_name, frame_id)
        color = (0, 255, 0)
        label = f"{final_name} ({best_sim:.3f}>{thr:.3f})"

    elif final_name == "Confirming...":
        color = (0, 165, 255)
        label = f"? {best_name} ({best_sim:.3f})"

    else:
        if best_sim >= BESTGUESS_THR:
            color = (0, 220, 220)
            label = f"~{best_name} ({best_sim:.3f})"
        else:
            color = (0, 0, 255)
            label = f"Unknown ({best_sim:.3f})"

        last = unk_saved.get(track_id, -UNK_SAVE_COOL)
        if frame_id - last >= UNK_SAVE_COOL:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            fname = os.path.join(UNKNOWN_DIR,
                                 f"unk_{track_id}_{frame_id}_{ts}.jpg")
            cv2.imwrite(fname, aligned)
            unk_saved[track_id] = frame_id

    thick = 1 if is_back else 2
    cv2.rectangle(disp, (ax1, ay1), (ax2, ay2), color, thick)
    draw_label(disp, label, ax1, ay1, color)


cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Cannot open {VIDEO_PATH}")
    sys.exit(1)

frame_id = 0
print("Starting ... press 'q' to quit\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    disp = frame.copy()
    H, W = frame.shape[:2]
    zone_y = int(H * ZONE_SPLIT_RATIO)

    ts = datetime.now().strftime("%d-%m-%Y  %H:%M:%S")
    (tw, th), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(disp, (W-tw-30, 10), (W-10, th+20), (0,0,0), -1)
    cv2.putText(disp, ts, (W-tw-20, th+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    cv2.line(disp, (0, zone_y), (W, zone_y), (180, 180, 0), 1)
    cv2.putText(disp, "BACK",  (8, zone_y-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,0), 1)
    cv2.putText(disp, "FRONT", (8, zone_y+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,0), 1)

    if frame_id % DETECT_EVERY == 0:
        all_boxes = []
        for rows, cols, ovlp in TILE_CONFIGS:
            for tile, xo, yo in get_tiles(frame, rows, cols, ovlp):
                all_boxes.extend(run_yolo_tile(tile, xo, yo))
        all_boxes = nms(all_boxes, NMS_IOU)

        tr = yolo.track(
            frame[..., ::-1], imgsz=800,
            conf=DEFAULT_CONF, iou=DEFAULT_IOU,
            persist=True, tracker="bytetrack.yaml", verbose=False
        )
        tmap = {}
        for r in tr:
            for box in r.boxes:
                if box.id is None:
                    continue
                tid = int(box.id.cpu().numpy())
                bx1, by1, bx2, by2 = map(int, box.xyxy[0].cpu().numpy())
                tmap[((bx1+bx2)//2//10, (by1+by2)//2//10)] = tid

        for (x1, y1, x2, y2, conf_) in all_boxes:
            cx = (x1+x2)//2//10
            cy_ = (y1+y2)//2//10
            tid = tmap.get((cx, cy_), hash((cx//2, cy_//2)) % 100_000)

            cached = track_last.get(tid)
            if cached and frame_id - cached[1] < REIDENTIFY_EVERY:
                cn, _, cs = cached
                col = (0,255,0) if cn not in ("Unknown","Confirming...") \
                      else (0,0,255)
                pad = int(PAD_RATIO*(y2-y1))
                ax1_ = max(0, x1-pad);  ay1_ = max(0, y1-pad)
                ax2_ = min(W-1, x2+pad); ay2_ = min(H-1, y2+pad)
                cv2.rectangle(disp, (ax1_,ay1_), (ax2_,ay2_), col, 2)
                draw_label(disp, f"{cn} ({cs:.3f})", ax1_, ay1_, col)
                continue

            process(frame, disp, x1, y1, x2, y2, tid, frame_id, H, W)

    out = cv2.resize(disp, (1280, 720))
    cv2.imshow("AdaFace IR-101 Classroom", out)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\n=== RECOGNITION RESULTS ===")
if recognized_faces:
    for name, frm in sorted(recognized_faces.items(), key=lambda x: x[1]):
        print(f"  {name:<25s}  first confirmed at frame {frm}")
else:
    print("  No faces confirmed. Tips:")
    print("  1. Lower SIM_BACK / SIM_FRONT thresholds")
    print("  2. Re-encode embeddings: python adaface_build_embeddings.py --encode")
    print("  3. Add far-face crops to far_faces/ and re-encode")
print(f"\nUnknown crops saved to: {UNKNOWN_DIR}/")
