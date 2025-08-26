# train_yolov8_seg_fixed_gray.py
import os
import sys
import random
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

# 尝试安装缺失依赖（可选）
def ensure_packages():
    reqs = ["ultralytics", "opencv-python", "pycocotools", "tqdm", "matplotlib", "numpy"]
    need_install = []
    for pkg in reqs:
        try:
            __import__(pkg if pkg != "opencv-python" else "cv2")
        except Exception:
            need_install.append(pkg)
    if need_install:
        print("检测到缺失的 Python 包，将尝试自动安装:", need_install)
        subprocess.check_call([sys.executable, "-m", "pip", "install", *need_install])

ensure_packages()

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ================= 配置 =================
TRAIN_ROOT = r"G://ai//new_try//all_dataset"
VAL_ROOT   = r"G://ai//new_try//val"
OUTPUT_WORKDIR = Path("yolov8_data_fixed_gray")
IMG_EXTS = (".jpg", ".jpeg", ".png")
TARGET_SIZE = (640,300)
BATCH = 8
EPOCHS = 30
DEVICE = "cuda" if (os.getenv("CUDA_VISIBLE_DEVICES","")!="-1" and shutil.which("nvidia-smi")) else "cpu"
MODEL_ARCH = "yolov8n-seg"
CLASS_MAP = {"inner":0, "outer":1,"background":2}
CLASS_NAMES = ["inner_edge","outer_edge","background"]

# ================= 辅助函数 =================
def find_mask_for_image(img_name: str, masks_dir: str) -> Tuple[str, str]:
    base = os.path.splitext(img_name)[0]
    base_no_suffix = base.replace("_ib", "").replace("_ob", "")
    for ext in IMG_EXTS:
        candidate = os.path.join(masks_dir, base_no_suffix + "_mask" + ext)
        if os.path.exists(candidate):
            return candidate, ext
    for ext in IMG_EXTS:
        candidate = os.path.join(masks_dir, base + ext)
        if os.path.exists(candidate):
            return candidate, ext
    return None, None

def mask_to_polygons_rel(mask: np.ndarray, class_selector: str) -> List[List[float]]:
    """按灰度值提取 polygons"""
    if class_selector == "inner":
        bin_mask = (mask == 128).astype(np.uint8)*255
    elif class_selector == "outer":
        bin_mask = (mask == 255).astype(np.uint8)*255
    elif class_selector == "background":
        bin_mask = (mask ==0).astype(np.uint8)*255
    else:
        raise ValueError("Unknown class_selector")

    h, w = bin_mask.shape[:2]
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1:  # 最小面积阈值
            continue
        eps = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if approx.shape[0] < 3:
            continue
        coords = approx.reshape(-1,2).tolist()
        flat = []
        for (x,y) in coords:
            flat.append(float(x/w))
            flat.append(float(y/h))
        polys.append(flat)
    return polys

def prepare_yolo_seg_dataset(root_dir: str, out_root: Path, split: str = "train"):
    images_src = os.path.join(root_dir, "images")
    masks_src  = os.path.join(root_dir, "masks")
    out_images_dir = out_root / "images" / split
    out_labels_dir = out_root / "labels" / split
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in os.listdir(images_src) if f.lower().endswith(IMG_EXTS)])
    processed, skipped = 0, 0
    for fname in tqdm(files, desc=f"prepare {split}"):
        img_path = os.path.join(images_src, fname)
        mask_path, _ = find_mask_for_image(fname, masks_src)
        if mask_path is None:
            print(f"跳过: {fname}, 未找到掩码")
            skipped += 1
            continue
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"跳过: {fname}, 无法读取掩码")
            skipped += 1
            continue

        print(f"{fname} mask unique values: {np.unique(mask)}")

        dst_img = out_images_dir / fname
        shutil.copy2(img_path, dst_img)

        base = os.path.splitext(fname)[0]
        label_file = out_labels_dir / (base + ".txt")
        lines = []

        inner_polys = mask_to_polygons_rel(mask, "inner")
        for poly in inner_polys:
            if len(poly)>=6:
                coords = " ".join([f"{x:.6f} {y:.6f}" for x,y in zip(poly[0::2], poly[1::2])])
                lines.append(f"{CLASS_MAP['inner']} {coords}")

        outer_polys = mask_to_polygons_rel(mask, "outer")
        for poly in outer_polys:
            if len(poly)>=6:
                coords = " ".join([f"{x:.6f} {y:.6f}" for x,y in zip(poly[0::2], poly[1::2])])
                lines.append(f"{CLASS_MAP['outer']} {coords}")

        if len(lines)==0:
            print(f"⚠️ 警告: {fname} 没有生成 polygons")
            skipped += 1
            continue

        with open(label_file, "w", encoding="utf-8") as f:
            for L in lines:
                f.write(L+"\n")
        processed += 1

    print(f"{split} done: processed={processed}, skipped={skipped}")
    return processed, skipped

def create_data_yaml(out_root: Path, yaml_path: Path):
    # 使用相对于data.yaml的相对路径（推荐）
    train_path = os.path.relpath(out_root / 'images' / 'train', start=yaml_path.parent)
    val_path = os.path.relpath(out_root / 'images' / 'val', start=yaml_path.parent)
    
    content = f"train: {train_path}\n"
    content += f"val:   {val_path}\n"
    content += f"nc: {len(CLASS_NAMES)}\n"
    content += "names:\n"
    for n in CLASS_NAMES:
        content += f"  - {n}\n"
    yaml_path.write_text(content, encoding="utf-8")
    print(f"data.yaml -> {yaml_path}")

def debug_visual_check(out_root: Path, n=5):
    imgs = list((out_root / "images" / "val").glob("*"))
    if len(imgs)==0:
        imgs = list((out_root / "images" / "train").glob("*"))
    if len(imgs)==0:
        print("无图片可视化")
        return
    samples = random.sample(imgs, min(n, len(imgs)))
    for img_path in samples:
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        base = img_path.stem
        label_path = out_root / "labels" / "val" / (base+".txt")
        if not label_path.exists():
            label_path = out_root / "labels" / "train" / (base+".txt")
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        ax.imshow(img); ax.axis("off"); ax.set_title(base)
        if label_path.exists():
            with open(label_path,"r",encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts)<3: continue
                    cls=int(parts[0])
                    coords = list(map(float, parts[1:]))
                    xs = coords[0::2]
                    ys = coords[1::2]
                    h,w = img.shape[:2]
                    pts = np.array([[x*w,y*h] for x,y in zip(xs,ys)],np.int32)
                    ax.plot(np.append(pts[:,0],pts[0,0]), np.append(pts[:,1],pts[0,1]), '-', linewidth=2, label=f"class{cls}")
        plt.legend()
        plt.show()

# ================= 主流程 =================
def main():
    random.seed(42)
    out_root = OUTPUT_WORKDIR
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("准备训练集...")
    p_train, s_train = prepare_yolo_seg_dataset(TRAIN_ROOT, out_root, split="train")
    print("准备验证集...")
    p_val, s_val = prepare_yolo_seg_dataset(VAL_ROOT, out_root, split="val")

    if p_train + p_val == 0:
        print("错误: 未生成任何 label！请检查 mask。")
        return

    data_yaml = out_root / "data.yaml"
    create_data_yaml(out_root, data_yaml)

    debug_visual_check(out_root, n=3)

    print("训练 YOLOv8-seg ...")
    model = YOLO(MODEL_ARCH)
    model.model.model[-1].nc = len(CLASS_NAMES)
    model.model.names = CLASS_NAMES
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=TARGET_SIZE,
        batch=BATCH,
        device=DEVICE,
        workers=4,
        name="solar_panel_seg_fixed_gray",
        project=str(Path("G:/ai/runs/segment")),
        exist_ok=True,
        verbose=True
    )

if __name__=="__main__":
    main()
