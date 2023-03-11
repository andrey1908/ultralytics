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

Train (results in runs):
```
model.train(data='coco128-seg.yaml', epochs=20, save_period=1, close_mosaic=20, lr0=0.01, lrf=1, warmup_epochs=0)
```
