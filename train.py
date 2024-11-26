import os

os.environ['WANDB_DISABLED'] = 'true'
from ultralytics import YOLO

model =YOLO("snn_yolov8l.yaml").load('69M_best.pt')

print(model)

#train
# model.train(data="coco.yaml",device=[7],epochs=100)  # train the model
model.train(data="coco.yaml",device=[4],epochs=100)  # train the model

#TEST
# model = YOLO('runs/detect/train1/weights/last.pt')  # load a pretrained model (recommended for training)

