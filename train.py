import os

os.environ['WANDB_DISABLED'] = 'true'
from ultralytics import YOLO

# Load a model


print(model)

#train
# model.train(data="coco.yaml",device=[0],epochs=100)  # train the model
model.train(data="coco.yaml",device=[0,1],epochs=100)  # train the model

#TEST
# model = YOLO('runs/detect/train1/weights/last.pt')  # load a pretrained model (recommended for training)

