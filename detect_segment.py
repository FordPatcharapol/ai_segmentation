import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import numpy as np
import shapely
import shapely.geometry
from shapely.geometry import Polygon
from itertools import groupby
import os, json, cv2, random
# from google.colab.patches import cv2_imshow
from skimage import measure
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import _create_text_labels
import yaml
from turbojpeg import TurboJPEG
import urllib.request

# ***** read config yaml *****
with open("config.yaml", "r") as file:
    try:
        # print(yaml.safe_load(file))
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# print(config['model'])

model = config['model']
device = model['device']
weights = model['weights']
threshold = model['threshold']
threshold_intersection = config['boundingbox']['threshold']

cfg = get_cfg()
if device == 'cpu':
    cfg.MODEL.DEVICE = 'cpu'
cfg.merge_from_file(model_zoo.get_config_file(weights))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weights)

# ***** define value *****
predictor = DefaultPredictor(cfg)
im = np.zeros((2, 3, 4))
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
model_classes = v.metadata.get("thing_classes", None)
jpeg = TurboJPEG()

# ***** convert dimention array *****
oneD2twoD = lambda x: [(x[2*i], x[2*i+1]) for i in range(len(x)//2)]

def binary_mask_to_polygon(binary_mask, tolerance=0):
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def crop_image_polygon(image):
    crop_img = image

    return crop_img

def load_image(img_path, image_type):

    if image_type == 'url':
        url_response = urllib.request.urlopen(img_path)
        image = jpeg.decode(bytearray(url_response.read()))
    elif image_type == 'base64':
        image = cv2.imread(img_path)
    elif image_type == 'file':
        in_file = open(img_path, 'rb')
        # // time decode only
        image = jpeg.decode(in_file.read())
        in_file.close()
    elif image_type == '':
        in_file = open(img_path, 'rb')
        image = jpeg.decode(in_file.read())
        in_file.close()

    else:
        return None

    return image

def output_instances(outputs):

    # output model paddleSeg
    pred_masks = outputs["instances"].pred_masks
    pred_classes = outputs["instances"].pred_classes
    scores = outputs["instances"].scores
    boxes = outputs["instances"].pred_boxes.tensor.numpy().tolist()
    classes = _create_text_labels(pred_classes, scores, model_classes)

    return pred_masks, classes, boxes

def rectangle_polygon(segments):
    polygon_coords = []
    polygon_coords.append([])

    for face in segments:
        multipoint = shapely.geometry.MultiPoint(face)
        label = [0,*np.array(multipoint.minimum_rotated_rectangle.exterior.coords[:-1]).ravel().tolist()]
        label_pixel = np.copy(label)
        polygon_coords[-1].append(np.vstack((label_pixel[1:].reshape(-1, 2), label_pixel[1:3])))

    return polygon_coords

def find_key_area(areas, segments, threshold):
    scores = []
    for area in areas:
        zone_area = convert_demention(areas[area])
        result = area_intersect(zone_area, segments)
        if (result / 100) < threshold:
            result = 0
        scores.append(result)

    result = dict(zip(areas, scores))
    max_value = max(result, key=result.get)

    if all(value == 0 for value in result.values()):
        return None

    return max_value

def convert_demention(polygon):
    for segment in polygon:
        segments = []
        segments.extend(oneD2twoD(segment) + segments[:2])
        segments = [segments]
    
    return segments

def show_image_window(image):
    while(1):
        cv2.imshow('image', image)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

def draw_polygonline(im, polygon):
    pts = np.array(polygon,np.int32)
    pts = pts.reshape((-1, 1, 2))
    isClosed = True
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    color = (b, g, r)
    thickness = 5

    drawn_image = cv2.polylines(im, [pts], isClosed, color, thickness)

    return drawn_image

def sub_message_from_kafka():

    message = {
        "images" : [
            {
                "image": {
                    "type": "",
                    "url": './data/parking_hik.jpg'
                },
                "boxes": {
                    "zone_1": [[0, 0, 0, 957, 1891, 957, 1891, 0]],
                    # "zone_2": [[0, 0, 100, 0, 0, 100, 0, 0]]
                }
            }
        ]
    }

    return message

def reply_message_to_kafka(message):
    print('reply')

def write_to_json(messages):
    with open('./polygon_message.json', "w") as outfile:
        json.dump(messages, outfile)

def area_intersect(polygon_1, polygon_2):
    polygon_1 = Polygon(polygon_1[0])
    polygon_2 = Polygon(polygon_2[0])

    intersect = polygon_2.intersection(polygon_1).area / polygon_2.area
    
    return round(intersect * 100, 2)

def segmentation_finding(im, areas):
    # message loop
    outputs = predictor(im)
    boxes = {}

    for area in areas:
        boxes[area] = []

    index = 0
    objects, classes, bbox = output_instances(outputs)

    for object_in_frame in objects:
        polygon = binary_mask_to_polygon(object_in_frame)
        object_polygon = {
            "label": '',
            "polygon": [],
            "box": [],
            "score": 0
        }
        segments = convert_demention(polygon)
        # polygon_box = rectangle_polygon(segments)

        object_polygon["label"] = classes[index][:-4]
        object_polygon["polygon"] = polygon
        object_polygon["box"] = bbox[index]
        object_polygon["score"] = int(classes[index][-3:-1])

        # print("in segmentation")
        area_intersect = find_key_area(areas, segments, threshold_intersection)
        # print("in segmentation {}".format(area_intersect))
        # print(area_intersect)
        if area_intersect == None:
            continue
        boxes[area_intersect].append(object_polygon)
        index = index + 1
        # image = draw_polygonline(im, polygon)
    # show_image_window(image)
    # write_to_json(boxes)
    return boxes

def main():
    # loop 
    messages = sub_message_from_kafka()

    for message in messages:
        images = messages[message]
        # print(messages[message])
        for image in images:
            # print(image)
            path = image['image']['url']
            image_type = image['image']['type']
            boxes = image['boxes']

            kafka_message = {
                "images": []
            }

            result_messages = {
                "boxes": {}
            }

            im = load_image(path, image_type)
            if im is None:
                continue

            result_messages['boxes'] = segmentation_finding(im, boxes)
            kafka_message['images'].append(result_messages)
            write_to_json(kafka_message)
            reply_message_to_kafka(kafka_message)
        # print(result_messages)
        # print('************************')

    # convert base to np image (condition)
    # type of file and read file from type (turboJPEG)
    # base64 url file = (jpg)
    # jpg (im, w, h) => nparray (function)
    # color level (bgr) up to model

if __name__ == "__main__":
    main()