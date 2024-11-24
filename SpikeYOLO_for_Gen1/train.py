import os

from ultralytics import YOLO

os.environ['WANDB_DISABLED'] = 'true'
fr_dict = {}

model = YOLO("snn_yolov8s.yaml")

model.train(data="gen1.yaml",device=[4,5,6,7],epochs=100)





#测试模型

