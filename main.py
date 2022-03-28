import cv2 as cv
import numpy as np
#firs edges(gray blur canny)
#contour (contour , peri pol , x,y,w,h, rectangel)

def getCountor(img):
    contour,steps=cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    for con in contour:
        area=cv.contourArea(con)
        #print(area)
        if area>500:
            cv.drawContours(imgCopy,con,-1,(255,0,0),3)
            peri=cv.arcLength(con,True)
           # print(peri)
            pols=cv.approxPolyDP(con,0.02*peri,True)
            #print(pols) # point of pol
            #print(len(pols)) # the number of pol
            x,y,w,h=cv.boundingRect(pols)
            objpol=len(pols)
            if objpol==3: obType="triangel"
            elif objpol==4: obType="rectangel"
            else: obType="circle"
            cv.rectangle(imgCopy,(x,y),(x+w,y+h),(255,0,0),4)
            cv.putText(imgCopy,obType,(x+(w//2),y+(h//2)),cv.FONT_ITALIC,0.5,(255,100,100),2)


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
        return ver


path='sources/shapes.png'
img=cv.imread(path)
imgCopy=img.copy()

imgGray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
imgBlur=cv.GaussianBlur(imgGray,(7,7),1)
imgCanny=cv.Canny(imgGray,50,50)
imgblank=np.zeros_like(img)
getCountor(imgCanny)

imgFINAL=stackImages(0.5,([img,imgGray,imgBlur],[imgCanny,imgCopy,imgblank]))
cv.imshow("final",imgFINAL)
cv.waitKey(0)

