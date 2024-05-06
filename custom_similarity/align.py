from facenet_pytorch import MTCNN
import cv2
import torch
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mtcnn = MTCNN(keep_all=True, device=device)

'''
Extract face from image
in_fly - if True - retrun image. If False - save image to /data/aligned
'''
def extract_face(path, size=[112, 112], in_fly=False):
    img = cv2.imread(path)
    boxes, probs = mtcnn.detect(img)
    if boxes is None:
        return None
    #get only face
    boxes = boxes[0]
    x1, y1, x2, y2 = boxes
    #make it square by adding padding equally on both sides
    width = x2 - x1
    height = y2 - y1
    if width > height:
        diff = width - height
        y1 -= diff/2
        y2 += diff/2
    else:
        diff = height - width
        x1 -= diff/2
        x2 += diff/2
    face = img[int(y1):int(y2), int(x1):int(x2)]
    #resize
    face = cv2.resize(face, size)
    if in_fly:
        return face
    else:
        imgname = os.path.basename(path)
        cv2.imwrite(f'data/aligned/{imgname}', face)
        return None