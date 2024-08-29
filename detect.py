# -*- coding: utf-8 -*-
import os
from PIL import Image
from ultralytics import YOLO

model = YOLO("/path/to/checkpoint/.pt")
results = model(['/path/to/data/.jpg'])
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    im_rgb.show()
    im_rgb.save('/result.jpg')  # save to disk

