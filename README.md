# shape_detection
in this project we will apply some filter ( Gray , Blur , canny ) to find all edges of image then we will be able to find contours in the image and finally we will calculate the number of polygon

first we need to add these two libraries :
                                            import cv2 as cv
                                            import numpy as np
                                            
                                            

then we will apply gray method to our image to gat GRAYSCALE : imgGray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
then we will have 1 channel and a gray image so we need to apply some BLUR : imgBlur=cv.GaussianBlur(imgGray,(7,7),1)
finally we need to set CANNY filter to finding edge : imgCanny=cv.Canny(imgGray,50,50)
so that we are able to find the contours in our image : contour,steps=cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
after that we need to calculate the Area and number of polygon in each contours 

 

