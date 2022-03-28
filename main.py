import cv2 as cv
import numpy as np
frameWidth = 540
frameHeight = 640
cap = cv.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)

kernel=np.ones((5,5))
#filter image gray>blur>canny>dilate>erode
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
def GrayScale(img):
    imgGray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    return imgGray
def BlurScale(imgGray):
    imgBlur=cv.GaussianBlur(imgGray,(5,5),1)
    return imgBlur
def CannyScale(imgBlur):
    imgCanny=cv.Canny(imgBlur,200,200)
    return imgCanny
def DilateScale(imgCanny):
    imgDilate=cv.dilate(imgCanny,kernel,iterations=2) #iteration means thikness
    return imgDilate
def ErodeScale(imgDgit):
    imgErode=cv.erode(imgDgit,kernel,iterations=1)
    return imgErode

#find contour
def FindContour(imgErode):
    contour, steps = cv.findContours(imgErode,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    maxArea=0
    biggestRectangel = np.array([])
    for cnt in contour:
        areaCnt=cv.contourArea(cnt)
        if areaCnt>2000:
            cv.drawContours(imgcopy,cnt,-1,(255,100,100),thickness=3)
            peri=cv.arcLength(cnt,True)
            approx=cv.approxPolyDP(cnt,0.02*peri,True)
            objPoly=len(approx)

            if objPoly==4 and areaCnt>maxArea:
                biggestRectangel=approx
                maxArea=areaCnt
                x,y,w,h=cv.boundingRect(approx)
                #cv.rectangle(imgcopy,(x,y),(x+w,y+h),(0,255,0),thickness=5)
    cv.drawContours(imgcopy, biggestRectangel, -1, (255, 0, 0), 20)
    return biggestRectangel
#get warp
def GetWarp(imgWarp,biggestRect):
    biggestRect = reorder(biggestRect)
    pts1 = np.float32(biggestRect)
    pts2 = np.float32([[0, 0], [frameWidth, 0], [0, frameHeight], [frameWidth, frameHeight]])
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv.warpPerspective(imgWarp, matrix, (frameWidth, frameHeight))

    imgCropped = imgOutput[20:imgOutput.shape[0]-20,20:imgOutput.shape[1]-20]
    imgCropped = cv.resize(imgCropped,(frameWidth,frameHeight))
    return imgCropped
#reorder
def reorder (myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    #print("add", add)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1]= myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    #print("NewPoints",myPointsNew)
    return myPointsNew




OUT=np.array([])
while True:
    sucess, img = cap.read()
    img=cv.resize(img,(frameWidth,frameHeight))
    imgcopy=img.copy()
    imgWrap=img.copy()
    imgOut=np.zeros_like(img)
    imgGray=GrayScale(img)
    imgBlur=BlurScale(imgGray)
    imgCanny=CannyScale(imgBlur)
    imgDilate=DilateScale(imgCanny)
    imgErode=ErodeScale(imgDilate)
    biggestRectangel=FindContour(imgErode)
    if len(biggestRectangel) !=0:
        imgOut=GetWarp(imgWrap,biggestRectangel)
        OUT=imgOut.copy()

    if len(OUT)!=0:
        imgOut=OUT.copy()


    imgFinal=stackImages(0.5,([imgOut,imgGray,imgBlur,imgcopy],[imgCanny,imgDilate,imgErode,imgWrap]))
    FindContour(imgErode)

    cv.imshow("final", imgFinal)



    if cv.waitKey(1) & 0xFF==ord('q'):
        break