import os

os.environ['WANDB_DISABLED'] = 'true'
from ultralytics import YOLO

# Load a model  COCO基准是209，voc基准是55
model =YOLO("snn_yolov8s.yaml")

print(model)

#train
model.train(data="coco.yaml",device=[0],epochs=100)  # train the modelaa

#TEST
# model = YOLO('runs/detect/train1/weights/last.pt')  # load a pretrained model (recommended for training)

