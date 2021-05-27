#COLOR DETECTION

import numpy as np
import cv2

a= cv2.VideoCapture(0)

while(True):
    ret,frame=a.read()
    frame = cv2.flip(frame, 1)
    
    #cv2.cvtColor() 
    #method is used to convert an image from one color space to another
    hsvFrame= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    red_lower= np.array([136, 87, 111], np.uint8)
    red_upper= np.array([180, 255, 255], np.uint8)
    red_mask= cv2.inRange(hsvFrame, red_lower, red_upper)
    
    green_lower= np.array([53, 74, 160], np.uint8)
    green_upper= np.array([94, 255, 255], np.uint8)
    green_mask= cv2.inRange(hsvFrame, green_lower, green_upper)
    
    blue_lower= np.array([100, 80, 150], np.uint8)
    blue_upper= np.array([120, 255, 255], np.uint8)
    blue_mask= cv2.inRange(hsvFrame, blue_lower, blue_upper)
    
    kernel=np.ones((5,5),"uint8")
    
    #Setting up the masks for each color
    
    #For red color
    red_mask= cv2.dilate(red_mask,kernel)
    res_red= cv2.bitwise_and(frame, frame, mask=red_mask)
    
    #For green color
    green_mask= cv2.dilate(green_mask,kernel)
    res_green= cv2.bitwise_and(frame, frame, mask=green_mask)
    
    #For blue color
    blue_mask= cv2.dilate(blue_mask,kernel)
    res_blue= cv2.bitwise_and(frame, frame, mask=blue_mask)
    
    
    #Creating contour to track red color
    contours, hierarchy= cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for pic, contour in enumerate(contours):
        area=cv2.contourArea(contour)
        if (area>300):
            x,y,w,h = cv2.boundingRect(contour)
            frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame,"RED",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255))

    #Creating contour to track green color
    contours, hierarchy= cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for pic, contour in enumerate(contours):
        area=cv2.contourArea(contour)
        if (area>300):
            x,y,w,h = cv2.boundingRect(contour)
            frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,"GREEN",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0))

    #Creating contour to track red color
    contours, hierarchy= cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for pic, contour in enumerate(contours):
        area=cv2.contourArea(contour)
        if (area>300):
            x,y,w,h = cv2.boundingRect(contour)
            frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(frame,"BLUE",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,0,0))
        
            
    cv2.imshow('BGR Detector in Real Time',frame)
    
    if(cv2.waitKey(1)& 0xFF==ord('q')):
        break

a.release()
cv2.destroyAllWindows()