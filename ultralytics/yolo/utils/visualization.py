import numpy as np
import cv2


def draw_detections(image, scores, classes_ids, boxes, masks,
        min_score=0.7, palette=((0, 0, 255),)):
    overlay = image.copy()
    for i, (score, class_id, mask) in enumerate(zip(scores, classes_ids, masks)):
        if score < min_score:
            continue
        polygons, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        color = palette[i % len(palette)]
        cv2.polylines(image, polygons, True, color, thickness=2)
        overlay[mask != 0] = np.array(color, dtype=np.uint8)
    cv2.addWeighted(image, 0.7, overlay, 0.3, 0, dst=image)
