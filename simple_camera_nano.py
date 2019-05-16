from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import torch
import cv2
import sys

def gstreamer_pipeline (capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=0) :   
    return ('nvarguscamerasrc ! ' 
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

class_names = ['background' , # always index 0
				'bb_extinguisher','bb_drill','bb_backpack']
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
state_dict = torch.load("/home/nvidia/model.pth")
net = create_mobilenetv1_ssd(len(class_names), is_test=True)
net.to(DEVICE)
predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200, device = DEVICE)
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
timer = Timer()
while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        continue
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    interval = timer.end()
    print('FPS: {:.2f}, Detect Objects: {:d}.'.format(1./interval, labels.size(0)))
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        check = True
        for j in box:
            if not 0 <= j < 640:
                check = False
        if check:
            if (box[0], box[1]) != (box[2], box[3]):
                label = "{}: {:.2f}".format(class_names[labels[i]],probs[i])
                cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

                cv2.putText(orig_image, label,
                    (box[0]+20, box[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
        cv2.imshow('annotated', orig_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
