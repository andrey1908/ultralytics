Imports:
```
import torch
from ultralytics import YOLO
```

Create model:
```
model = YOLO('ultralytics/models/v8/seg/yolov8n-seg.yaml')
```

Load weights:
```
# YOLO('yolov8n-seg.pt')  # to download pretrained model
weights = torch.load('yolov8n-seg.pt')['model']
model.model.load(weights)
```

Train (results in "runs" folder):
```
# default parameters in "ultralytics/yolo/cfg/default.yaml"
model.train(data='coco128-seg.yaml', epochs=20, save_period=1, close_mosaic=20, lr0=0.01, lrf=1, warmup_epochs=0)
```

Inference (results in "runs" folder):
```
results = model('img.jpg', save=True, show=False)

# or

from ultralytics.yolo.utils.ops import preprocess_results
from ultralytics.yolo.utils.visualization import draw_detections

image = cv2.imread("path/to/image.jpg")
results = model(image, save=False, show=False, verbose=False)
# assert len(results) == 1
height, width = image.shape[:2]
scores, classes_ids, boxes, masks = preprocess_results(results, (height, width))
draw_detections(image, scores, classes_ids, boxes, masks, min_score=0.7)
```
