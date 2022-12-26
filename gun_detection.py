import cv2
import numpy as np
import matplotlib.pyplot as plt
import winsound

net = cv2.dnn.readNetFromDarknet("yolov3_custom.cfg", r"yolov3_custom_4000.weights")

classes = ['notGun', 'gun']

# cap = cv2.VideoCapture('gun1.mp4')
img = cv2.imread(r"img20.jpg")

# total_frames = cap.get(7)
# cap.set(1, 400)

# if cap.isOpened()== False:
#     print("Error opening video  file")

# while (cap.isOpened()):
# _, img = cap.read()
img = cv2.resize(img, (1280, 720))
hight, width, _ = img.shape
blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

net.setInput(blob)

output_layers_name = net.getUnconnectedOutLayersNames()

layerOutputs = net.forward(output_layers_name)

boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
    for detection in output:
        score = detection[5:]
        class_id = np.argmax(score)
        confidence = score[class_id]
        if confidence > 0.2:  # 0.7
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * hight)
            w = int(detection[2] * width)
            h = int(detection[3] * hight)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            print(x,y,w,h)
            confidences.append((float(confidence*100)))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)

boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
    for detection in output:
        score = detection[5:]
        class_id = np.argmax(score)
        confidence = score[class_id]
        if confidence > 0.2: #
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * hight)
            w = int(detection[2] * width)
            h = int(detection[3] * hight)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append((float(confidence*100)))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, .8, .4)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(boxes), 3))



if len(indexes) > 0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label + " " + confidence, (x, y + 400), font, 2, color, 2)
        winsound.Beep(1000, 100)
i=1
for (x, y, w, h) in boxes:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi_color = img[y:y + h, x:x + w]
    print("[INFO] Object found. Saving locally.")
    cv2.imwrite('Gun_detected/' + str(i) + '_Gun.jpg', roi_color)
    i = i + 1

cv2.imshow('img', img)
# if cv2.waitKey(1) == ord('q'):
#     break

cv2.waitKey(0)
# cap.release()
cv2.destroyAllWindows()

