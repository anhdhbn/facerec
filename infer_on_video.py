
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_face_bank_cus, draw_box_name
import cv2
from pathlib import Path
import os
from PIL import Image
import numpy as np
conf = get_config(False)


mtcnn = MTCNN()
print('mtcnn loaded')

learner = face_learner(conf, True)

learner = face_learner(conf, True)
learner.threshold = 1.54
if conf.device.type == 'cpu':
    learner.load_state(conf, 'cpu_final.pth', True, True)
else:
    learner.load_state(conf, 'final.pth', True, True)
learner.model.eval()
print('learner loaded')

data = load_face_bank_cus(conf)
targets = np.array([embed for embed, _ in data])
# print(targets.shape)
names = [label for _, label in data]
# print(len(targets), len(names))
print(names)
print('facebank loaded')

video_input_path = "test.mp4"
filename = os.path.basename(video_input_path).split(".")[0]
video_ouput_path = Path(video_input_path).parent/'{}_output.mp4'.format(filename)
cap = cv2.VideoCapture(video_input_path)

cap.set(cv2.CAP_PROP_POS_MSEC, 0)

width, height = 112, 112
fps = 10
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')

if cap.isOpened():
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    #print(cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT) # 3, 4

    print('[INFO] Width, Height input and output video:', width, height)
else:
    print("Error opening video stream or file")

video_writer = cv2.VideoWriter(str(video_ouput_path), fourcc,  int(fps), (int(width),int(height)))

# video_writer = cv2.VideoWriter(video_ouput_path, -1, 20.0, frameSize=(1280,720))
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# video_writer =  cv2.VideoWriter('output.avi', fourcc, 29.0, (640, 480))

print(video_ouput_path)

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        image = Image.fromarray(frame)
        bboxes, faces = mtcnn.align_multi(image,None, 16)
        if len(bboxes) == 0:
            print('no face')
            continue
        else:
            bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1,-1,1,1] # personal choice
            results, score = learner.infer_cus(conf, faces, targets, True)
            # print(f"[INFO] Results idx name: {results}")
            # print(len(bboxes))
            # print(len(results))
            for idx,bbox in enumerate(bboxes):
                if True:
                    frame = draw_box_name(bbox, names[results[idx]] + '_{:.2f}'.format(score[idx]), frame)
                else:
                    frame = draw_box_name(bbox, names[results[idx] + 1], frame)
        video_writer.write(frame)
        # Display the resulting frame
        cv2.imshow('Frame',frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

video_writer.release()

# Closes all the frames
cv2.destroyAllWindows()
