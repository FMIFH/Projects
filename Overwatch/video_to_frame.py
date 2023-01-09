import cv2
import os
VIDEO_PATH = 'Videos'
FRAMES_PATH = 'Frames'
os.makedirs(FRAMES_PATH, exist_ok=True)
video_ids = [os.path.join(VIDEO_PATH, i) for i in next(os.walk(VIDEO_PATH))[2]]

for v in video_ids:
    id = v.split('\\')[1].split('.')[0]
    print(id)
    cam = cv2.VideoCapture(v)
    currentframe = 0
    while(True):
        ret,frame = cam.read()
        if ret:
            if (currentframe%(30*10))==0:
                name = '{id}_frame_{f}.jpg'.format(id=id, f=currentframe)
                path = os.path.join(FRAMES_PATH, name)
                cv2.imwrite(path, frame)
            currentframe += 1
        else:
            break
    cam.release()
    cv2.destroyAllWindows()
