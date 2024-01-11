#Creating a step-by-step tutorial for object detection using Python

#------------------------------------------------------------------------------
#             1. Introduction to Object Detection                      
#------------------------------------------------------------------------------

"""Object detection is a computer vision technique used to identify and locate
objects within images or videos. It involves two primary tasks: localizing 
objects within an image (drawing bounding boxes around them) and classifying 
the type of objects present."""




"""Applications of object detection are diverse and continue to expand:

1.Autonomous Vehicles: Object detection is crucial for vehicles to recognize 
pedestrians, other vehicles, traffic signs, and obstacles on the road for safe 
navigation.

2.Surveillance and Security: Identifying people, intruders, or suspicious 
activities in surveillance footage helps enhance security.

3.Retail: Retailers use object detection for inventory management, tracking 
products on shelves, and analyzing customer behaviour in stores.

4.Medical Imaging: Object detection assists in analyzing medical images, 
locating anomalies, tumors, or other medical conditions.

5.Augmented Reality: Recognizing and tracking objects in real-time facilitates 
AR applications, enhancing user experiences.

6.Industrial Automation: Object detection is used in quality control, robotics, 
and monitoring production lines."""




"""Object detection continues to evolve, with ongoing research focusing on 
improving accuracy, speed, and applicability across various domains."""






#------------------------------------------------------------------------------
#  2.List of prerequisites (Python basics, libraries like OpenCV or TensorFlow)
#------------------------------------------------------------------------------

"""We are using OpenCV library"""

# Install OpenCV 

pip install opencv-python numpy








#-----------------------------------------------------------------------------
#              3.Setting up the programming environment
#------------------------------------------------------------------------------

"""We have installed & imported libraries like OpenCV, numpy. Using Spyder note 
book as an Integrated Development Environment (IDE) to write our code in Python 
programming language. In next step we will download YOLO Weights and Config 
File."""






#------------------------------------------------------------------------------
#            4.Download YOLO Weights and Config File
#------------------------------------------------------------------------------

"""Download the YOLOv3 weights and configuration file from the official 
YOLO website: https://pjreddie.com/darknet/yolo/"""






#------------------------------------------------------------------------------
#                 5.Code for Object Detection
#------------------------------------------------------------------------------

import cv2                                               #import OpenCV library

import numpy as np                                       #import Numpy liabrary

import matplotlib.pyplot as plt


# Load pretrained YOLO Model

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
    
 
"""This section gets the output layers of the YOLO network. YOLO has some output 
layers that give predictions at different scales. This code retrieves those 
output layers.""" 
  
layer_names = net.getUnconnectedOutLayersNames()




# Load image

image = cv2.imread("pexels-pixabay-52500.jpg")  


"""This part reads an image ("sample_image.jpg") using OpenCV's imread() function 
and extracts its height, width, and number of channels (RGB channels)."""

height, width, channels = image.shape




# Preprocess Image

blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(layer_names)




# Get class IDs, confidences, and bounding boxes

class_ids = []
confidences = []
boxes = []
    
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Confidence threshold
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
 
"""The above section processes the detections obtained from the YOLO network. 
It iterates through the detections and filters out low-confidence detections 
(confidence threshold set at 0.5). For each detected object, it extracts the 
class ID, confidence, and bounding box coordinates."""




# Non-maximum suppression to remove overlapping bounding boxes

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

"""Non-Maximum Suppression (cv2.dnn.NMSBoxes()) is applied to eliminate 
overlapping bounding boxes, keeping only the most confident ones."""




# Choose font style

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))

"""Here, a font style is chosen for the labels, and random colours are generated 
for each class."""




# Draw bounding boxes

if len(indexes) > 0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = colors[class_ids[i]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x, y + 30), font, 1, color, 2)

"""Finally, this block draws rectangles around the detected objects, labels 
them with their class names and confidence scores, and displays the image using 
OpenCV's """



# Display the result

cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


plt.imshow(image)





    
    





















