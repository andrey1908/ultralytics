import numpy as np
import cv2
from ultralytics.yolo.utils.ops import preprocess_results


def draw_results(image, results, min_score=0.7, palette=((0, 0, 255),)):
    assert len(results) == 1
    height, width = image.shape[:2]
    scores, classes_ids, boxes, masks = preprocess_results(results, (height, width))

    overlay = image.copy()
    for i, (score, class_id, mask) in enumerate(zip(scores, classes_ids, masks)):
        if score < min_score:
            continue
        polygons, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        color = palette[i % len(palette)]
        cv2.polylines(image, polygons, True, color, thickness=2)
        overlay[mask != 0] = np.array(color, dtype=np.uint8)
    cv2.addWeighted(image, 0.7, overlay, 0.3, 0, dst=image)
