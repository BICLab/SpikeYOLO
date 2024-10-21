import os

os.environ['WANDB_DISABLED'] = 'true'
from ultralytics import YOLO

model =YOLO("snn_yolov8s.yaml")

print(model)

#train
# model.train(data="coco.yaml",device=[0],epochs=100)  # train the model
model.train(data="coco.yaml",device=[0,1],epochs=300)  # train the model

#TEST
# model = YOLO('runs/detect/train1/weights/last.pt')  # load a pretrained model (recommended for training)

