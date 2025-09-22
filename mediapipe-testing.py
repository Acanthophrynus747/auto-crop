import os
from PIL import Image, ImageDraw
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector/index#models

#no clue what model to use

cwd = os.getcwd()

model_path = cwd + '/efficientdet_lite2.tflite'

image_path = cwd + '/test-img-crow.jpg'

BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ObjectDetectorOptions(
    base_options = BaseOptions(model_asset_path = model_path),
    max_results = 5,
    running_mode = VisionRunningMode.IMAGE
    )

with ObjectDetector.create_from_options(options) as detector:
    #detector initialized, use here

    mp_image = mp.Image.create_from_file(image_path)

    detection_result = detector.detect(mp_image)

    # print(detection_result)
    # print(type(detection_result))
    # print(dir(detection_result))
    # print(len(detection_result.detections))

    detection = detection_result.detections[0]
    box = detection.bounding_box
    
    print(box)

    box_x = box.origin_x
    box_y = box.origin_y
    box_width = box.width
    box_height = box.height

    left = box_x - box_width / 2
    bottom = box_y - box_height / 2
    right = left + box_width
    top = bottom + box_height

    with Image.open(image_path) as img:

        draw = ImageDraw.Draw(img)

        draw.rectangle([(left, bottom), (right, top)], fill = None, outline = (255, 255, 255), width = 5)

        # for some reason the box is off. the second is right on only the beak, the third is just off in the background (robin image)
        #tested with another image. they are just fucky in general. will investigate
        
        img.show()
