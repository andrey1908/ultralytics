import numpy as np
import cv2


def draw_detections(image, scores, classes_ids, boxes, masks,
        min_score=0.7, draw_boxes=True, draw_masks=True, palette=((0, 0, 255),)):
    overlay = image.copy()
    for i, (score, class_id, box, mask) in enumerate(zip(scores, classes_ids, boxes, masks)):
        if score < min_score:
            continue
        if draw_boxes:
            if draw_masks:
                color = (255, 255, 255)
            else:
                color = palette[i % len(palette)]
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        if draw_masks:
            color = palette[i % len(palette)]
            overlay[mask != 0] = np.array(color, dtype=np.uint8)
            if not draw_boxes:
                polygons, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.polylines(image, polygons, True, color, thickness=2)
    cv2.addWeighted(image, 0.7, overlay, 0.3, 0, dst=image)
