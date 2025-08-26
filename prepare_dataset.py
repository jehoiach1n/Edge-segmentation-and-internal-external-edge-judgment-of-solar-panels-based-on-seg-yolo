import os
import json
import cv2
import numpy as np
from tqdm import tqdm

def labelme_to_mask(json_path, img_shape):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for shape in data.get('shapes', []):
        points = np.array(shape['points'], dtype=np.int32)
        if shape['label'] == 'ib':
            cv2.fillPoly(mask, [points], 2)  # 内边pip uninstall -y opencv-python
        elif shape['label'] == 'ob':
            cv2.fillPoly(mask, [points], 1)  # 外边
    return mask

def main(images_dir, labels_dir, out_dir):
    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'masks'), exist_ok=True)

    for img_file in tqdm(os.listdir(images_dir)):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(images_dir, img_file)
        json_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.json')
        if not os.path.exists(json_path):
            continue

        img = cv2.imread(img_path)
        mask = labelme_to_mask(json_path, img.shape)

        cv2.imwrite(os.path.join(out_dir, 'images', img_file), img)
        cv2.imwrite(os.path.join(out_dir, 'masks', os.path.splitext(img_file)[0] + '.png'), mask)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    main(args.images, args.labels, args.out)
