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

Train (trained weights and figures will be in "runs" folder):
```
# Notes:
#   default training parameters are in "ultralytics/yolo/cfg/default.yaml"
#   to disable mosaic augmentation set 'close_mosaic' equal to epochs
#   'nbs' parameter sets accumulated batch size
#   to enable advanced augmentation install albumentations for python. Probabilities for albumentations augmentations can be adjusted in class named 'Albumentations'.
#   you might want to freeze backbone of trained network. To do so for YOLOv8n, set requires_grad = False for all parameters with names starting not with 'model.22.'
#   if you run training from terminal, it will spam a lot of warnings. To suppress them you can run training in notebook (tested in vscode)
model.train(data='coco128-seg.yaml', epochs=20, save_period=1, close_mosaic=20, lr0=0.01, lrf=1, warmup_epochs=0, batch=16, nbs=16)
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
