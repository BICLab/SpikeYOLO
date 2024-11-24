import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'  # YOLO会默认占用一部分第一张能看到的卡的显存

from ultralytics import YOLO


model = YOLO('/lxh/yolo/yolov8/gen1_result/gen1_result/262_gen1_50ep_bs10_lr001_232m_t1_d1_quant_soft_use/weights/best.pt')  # load a pretrained model (recommended for training)
# model = YOLO('/lxh/yolo/yolov8/gen1_result/gen1_result/252_gen1_50ep_bs10_lr001_232m_t4_d1_lif_soft_use/weights/best.pt')
# model = YOLO('/lxh/yolo/yolov8/gen1_result/gen1_result/249_gen1_50ep_bs10_lr001_232m_t4_d2_quant_soft_true/weights/best.pt')


model.val(data="gen1.yaml",device=[7])