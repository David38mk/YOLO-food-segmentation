import cv2
import matplotlib.pyplot as plt
import random
import os

img_dir = "../yolo/images/train"
lbl_dir = "../yolo/labels/train"

samples = random.sample(os.listdir(img_dir), 5)

for img_name in samples:
    img = cv2.imread(os.path.join(img_dir,img_name))
    h,w,_ = img.shape

    lbl = img_name.replace(".jpg",".txt")
    with open(os.path.join(lbl_dir,lbl)) as f:
        for row in f:
            _,x,y,ww,hh = map(float,row.split())
            x1 = int((x-ww/2)*w)
            y1 = int((y-hh/2)*h)
            x2 = int((x+ww/2)*w)
            y2 = int((y+hh/2)*h)
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.show()
