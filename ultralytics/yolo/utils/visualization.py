import numpy as np
import cv2
from ultralytics.yolo.utils.ops import scale_image


def draw_results(image, results, min_score=0.7):
    assert len(results) == 1
    result = results[0]

    if result.masks is None:
        assert result.boxes.boxes.numel() == 0
        return

    boxes = result.boxes.boxes.cpu().numpy()
    masks = result.masks.masks.cpu().numpy()
    assert len(masks) == len(boxes)

    scores = boxes[:, 4]
    classes_ids = boxes[:, 5].astype(int)
    masks = masks.astype(np.uint8)

    height, width = image.shape[:2]
    mask_height, mask_width = masks.shape[1:]
    masks = masks.transpose(1, 2, 0)
    scaled_masks = scale_image((mask_height, mask_width), masks, (height, width))
    scaled_masks = scaled_masks.transpose(2, 0, 1)

    overlay = image.copy()
    for score, class_id, scaled_mask in zip(scores, classes_ids, scaled_masks):
        if score < min_score:
            continue
        polygons, _ = cv2.findContours(scaled_mask,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.polylines(image, polygons, True, (0, 0, 255), thickness=2)
        overlay[scaled_mask != 0] = np.array([0, 0, 255], dtype=np.uint8)
    cv2.addWeighted(image, 0.7, overlay, 0.3, 0, dst=image)
