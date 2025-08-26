import os
import json
import cv2
from ultralytics import YOLO

def yolo_infer_to_labelme(model, img_path, out_dir, class_map):
    results = model.predict(img_path)

    for result in results:
        if result.masks is None:   # ✅ 没有检测到分割目标
            print(f"⚠️ No detections for {img_path}, skipping...")
            continue  

        shapes = []
        for mask, cls_id in zip(result.masks.xy, result.boxes.cls):
            points = mask.tolist()
            label = class_map[int(cls_id)]
            shape = {
                "label": label,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
            shapes.append(shape)

        data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": os.path.basename(img_path),
            "imageData": None,
            "imageHeight": result.orig_img.shape[0],
            "imageWidth": result.orig_img.shape[1]
        }

        out_file = os.path.join(out_dir, os.path.splitext(os.path.basename(img_path))[0] + ".json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {out_file}")


if __name__ == "__main__":
    # 1. 加载你训练好的模型
    model = YOLO(r"G:\ai\runs\segment\solar_panel_seg_fixed_gray\weights\best.pt")

    # 2. 设置类别映射（0=内边, 1=外边）
    class_map = {0: "inner", 1: "outer"}

    # 3. 输入图片和输出目录
    img_dir = r"G://ai//new_try//dataset//images"
    out_dir = r"G://ai//new_labelme_json"

    for img_file in os.listdir(img_dir):
        if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(img_dir, img_file)
            yolo_infer_to_labelme(model, img_path, out_dir, class_map)
