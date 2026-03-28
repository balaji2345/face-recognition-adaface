"""
adaface_build_embeddings.py
============================
Builds students.npz using AdaFace IR-101 WebFace12M.
"""

import os, sys, cv2, argparse, shutil
import numpy as np
import torch
from pathlib import Path


# ── Patch transformers ─────────────────────────
def _patch_transformers():
    try:
        import transformers.modeling_utils as _mu
        if getattr(_mu.PreTrainedModel, '_cvlface_patched', False):
            return
        _orig_getattr = _mu.PreTrainedModel.__getattr__

        def _safe_getattr(self, name):
            if name == "all_tied_weights_keys":
                return getattr(self, "_tied_weights_keys", [])
            return _orig_getattr(self, name)

        _mu.PreTrainedModel.__getattr__ = _safe_getattr
        _mu.PreTrainedModel._cvlface_patched = True

    except Exception as e:
        print(f"⚠️ transformers patch failed: {e}")


_patch_transformers()


# ═════════ CONFIG ═════════
TRAIN_DIR = r"C:\Users\TIH48\Desktop\face recognition\faces_split\train"
FAR_DIR = r"C:\Users\TIH48\Desktop\face recognition\far_faces"

OUTPUT_NPZ = "students.npz"


HF_REPO_ID = "minchul/cvlface_adaface_ir101_webface12m"
MODEL_DIR = "adaface_model"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INSIGHTFACE_CTX = 0

MIN_FACE_SIZE = 20
MIN_EMB_NORM = 3.0
USE_CLAHE = True

clahe_obj = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))


# ═════════ DOWNLOAD ═════════
def cmd_download():
    from huggingface_hub import hf_hub_download

    model_path = Path(MODEL_DIR)
    model_path.mkdir(exist_ok=True)

    if (model_path / "model.safetensors").exists() and \
            (model_path / "wrapper.py").exists():
        print("✅ Already downloaded")
        return

    print("Downloading model...")

    hf_hub_download(
        HF_REPO_ID,
        "files.txt",
        local_dir=str(model_path),
        local_dir_use_symlinks=False
    )

    extra = []

    ft = model_path / "files.txt"
    if ft.exists():
        extra = [l.strip() for l in ft.read_text().splitlines() if l.strip()]

    all_files = list(dict.fromkeys(
        extra + ["config.json", "wrapper.py", "model.safetensors"]
    ))

    for fname in all_files:

        dest = model_path / fname

        if dest.exists():
            print("✓", fname)
            continue

        print("↓", fname)

        dest.parent.mkdir(parents=True, exist_ok=True)

        hf_hub_download(
            HF_REPO_ID,
            fname,
            local_dir=str(model_path),
            local_dir_use_symlinks=False
        )

    print("✅ Download Complete")


# ═════════ LOAD MODEL ═════════
def load_adaface():
    from transformers import AutoModel

    model_path = Path(MODEL_DIR).absolute()

    if not (model_path / "model.safetensors").exists():
        print("Run --download first")
        sys.exit(1)

    print("Loading AdaFace...")

    orig = os.getcwd()

    os.chdir(str(model_path))

    sys.path.insert(0, str(model_path))

    model = AutoModel.from_pretrained(
        str(model_path),
        trust_remote_code=True
    )

    os.chdir(orig)

    model = model.to(DEVICE).eval()

    print("✅ AdaFace Ready")

    return model


def load_detector():
    from insightface.app import FaceAnalysis

    print("Loading detector...")

    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection"]
    )

    app.prepare(
        ctx_id=INSIGHTFACE_CTX,
        det_size=(640, 640)
    )

    print("✅ Detector Ready")

    return app


# ═════════ PREPROCESS ═════════
def enhance(bgr):

    if not USE_CLAHE:
        return bgr

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    l = clahe_obj.apply(l)

    return cv2.cvtColor(
        cv2.merge([l, a, b]),
        cv2.COLOR_LAB2BGR
    )


def align_face(detector, bgr, target=112):

    from insightface.utils import face_align

    for scale in [1.0, 1.5, 2.0, 3.0, 4.0]:

        img = cv2.resize(
            bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LANCZOS4
        ) if scale > 1.01 else bgr.copy()

        img = enhance(img)

        faces = detector.get(img)

        if not faces:
            continue

        face = max(
            faces,
            key=lambda f:
            (f.bbox[2] - f.bbox[0]) *
            (f.bbox[3] - f.bbox[1])
        )

        bw = face.bbox[2] - face.bbox[0]
        bh = face.bbox[3] - face.bbox[1]

        if min(bw, bh) < MIN_FACE_SIZE:
            continue

        return face_align.norm_crop(
            img,
            landmark=face.kps,
            image_size=target
        )

    return None


def make_tensor(img):

    t = torch.from_numpy(
        img.astype(np.float32)
    )

    t = t.permute(2, 0, 1).unsqueeze(0)

    t = (t / 255.0 - 0.5) / 0.5

    return t.to(DEVICE)


@torch.no_grad()
def get_embedding(model, detector, bgr):

    aligned = align_face(detector, bgr)

    if aligned is None:
        return None, 0.0

    out = model(make_tensor(aligned))

    if isinstance(out, (tuple, list)):
        emb, norm = out[0], out[1]
    else:
        emb = out
        norm = torch.norm(emb, dim=1)

    emb_np = emb.squeeze().cpu().numpy().astype(np.float32)

    norm_val = float(norm.squeeze().cpu().item())

    if norm_val < MIN_EMB_NORM:
        return None, norm_val

    return emb_np, norm_val


# ═════════ ENCODE ═════════
def encode_folder(model, detector, folder, label, embeddings, names, tag=""):

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    images = sorted(
        f for f in Path(folder).iterdir()
        if f.suffix.lower() in exts
    )

    ok = fail = 0

    for p in images:

        bgr = cv2.imread(str(p))

        if bgr is None:
            fail += 1
            continue

        emb, _ = get_embedding(
            model,
            detector,
            bgr
        )

        if emb is None:
            fail += 1
            continue

        embeddings.append(emb)

        names.append(label)

        ok += 1

    print(f"{label} ✓{ok} ✗{fail}")

    return ok


def cmd_encode():

    train_path = Path(TRAIN_DIR)

    model = load_adaface()

    detector = load_detector()

    embeddings = []

    names = []

    print("Encoding Training Images")

    # ✅ CHANGE DONE HERE — NO SKIPPING ANY FOLDER
    for d in sorted(p for p in train_path.iterdir() if p.is_dir()):

        # MEN / WOMEN / ANY folder will now encode
        encode_folder(
            model,
            detector,
            d,
            d.name,
            embeddings,
            names,
            "CLOSE"
        )

    if not embeddings:
        print("No embeddings extracted")
        return

    emb_arr = np.stack(embeddings).astype(np.float32)

    name_arr = np.array(names, dtype=str)

    out = Path(OUTPUT_NPZ)

    if out.exists():

        shutil.copy(
            str(out),
            BACKUP_NPZ
        )

    np.savez(
        str(out),
        embeddings=emb_arr,
        names=name_arr
    )

    print("✅ students.npz saved")


# ═════════ VERIFY ═════════
def cmd_verify():

    data = np.load(
        OUTPUT_NPZ,
        allow_pickle=True
    )

    print("Embeddings :", len(data["names"]))


# ═════════ TEST ═════════
def cmd_test_image(img_path):

    model = load_adaface()

    detector = load_detector()

    bgr = cv2.imread(img_path)

    emb, norm = get_embedding(
        model,
        detector,
        bgr
    )

    print("Norm :", norm)


# ═════════ MAIN ═════════
if __name__ == "__main__":

    p = argparse.ArgumentParser()

    p.add_argument("--download", action="store_true")

    p.add_argument("--encode", action="store_true")

    p.add_argument("--verify", action="store_true")

    p.add_argument("--test-image")

    args = p.parse_args()

    if args.download:
        cmd_download()

    elif args.encode:
        cmd_encode()

    elif args.verify:
        cmd_verify()

    elif args.test_image:
        cmd_test_image(args.test_image)

    else:
        p.print_help()