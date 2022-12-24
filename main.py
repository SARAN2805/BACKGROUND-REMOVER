import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
cap = cv2.VideoCapture(0)
cap.set(3, 648)
cap.set(4, 488)
cap.set(cv2.CAP_PROP_FPS, 60)
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()
listImg = os.listdir("images")
imgList = []
for imgPath in listImg:
    img = cv2.imread(f'images/{imgPath}')
    imgList.append(img)
indexImg = 0
while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img, imgList[indexImg], threshold=0.95)
    imgStacked = cvzone.stackImages([img, imgOut], 2, 1)
    _, imgStacked = fpsReader.update(imgStacked)
    print(indexImg)
    cv2.imshow("image", imgStacked)
    key = cv2.waitKey(1)
    # move to previous background images 'a'
    if key == ord('a'):
        if indexImg > 0:
            indexImg -= 1
    # move to next background images press 'd'
    elif key == ord('d'):
        if indexImg < len(imgList)-1:
            indexImg += 1
    # to quit press 'q'
    elif key == ord('q'):
        break
