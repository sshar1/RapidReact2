import cv2 as cv
import numpy as np
import json

f = open("processing/data.json") 
data = json.load(f)

#accessing camera
source = 1
camera = cv.VideoCapture(source, cv.CAP_DSHOW)
camera.set(cv.CAP_PROP_EXPOSURE, -7) # change to -1 for internal camera, -7 for FISHEYE, -4 for Microsoft hd3000

def getDistance(focal_length, real_width, width_in_frame):
    distance = (real_width * focal_length) / width_in_frame
   
    return distance

def isCircle(cnt):
    approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
    
    _, radius = cv.minEnclosingCircle(cnt)
    contour_area = cv.contourArea(cnt)

    return len(approx) > 7 and 1.0 >= contour_area / (radius**2 * 3.14) >= .8 and contour_area > 300

def drawRect(mask, img, color):
    
    #Get contours on the mask
    colors = {'red' : (0, 0, 255), 'blue' : (255, 0, 0), }

    contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) == 2:
        contours = contours[0]
        
    else:
        contours = contours[1]
    
    for contour in contours:
        (coord_x, coord_y), _ = cv.minEnclosingCircle(contour)
        center = (int(coord_x), int(coord_y))

        if isCircle(contour):

            x, y, w, h = cv.boundingRect(contour)
            aspect_ratio = w / h

            if .8 <= aspect_ratio <= 1.2:
                distance = getDistance(630, 24.13, int(w))
                distance = format((int(distance) * 1.1) / 100, '.2f')

                cv.circle(img, center, int(w/2), (0,255,0), 5)
                cv.putText(img, color.upper() + " BALL " + str(distance) + " METERS", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255) , 2)    
          

def editImage(img):

    #blur, erode, and dilate frame
    img = cv.GaussianBlur(img, (3, 3), None)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, (7, 7))
    img = cv.morphologyEx(img, cv.MORPH_OPEN, (3, 3))
    
    return img

def createBlueMask(img):
    global mask_blue

    #blue values 
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower1 = np.array(data['cargo']["hsv_blue"]["lower"])
    upper1 = np.array(data['cargo']["hsv_blue"]["upper"])
    
    mask_blue = cv.inRange(hsv, lower1, upper1)
    
    return mask_blue

def createRedMask(img):
    global mask_red

    #red values
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    lower1 = np.array(data['cargo']["hsv_red1"]["lower"])
    upper1 = np.array(data['cargo']["hsv_red1"]["upper"])
    lower2 = np.array(data['cargo']["hsv_red2"]["lower"])
    upper2 = np.array(data['cargo']["hsv_red2"]["upper"])
    
    if(source == 0):
        mask_red2 = cv.inRange(hsv, lower1, upper1)

        mask_red = cv.bitwise_not(mask_red2)
        mask_red = cv.morphologyEx(mask_red, cv.MORPH_CLOSE, (27, 27))

    elif(source == 1): #fisheye
        mask_red = cv.inRange(hsv, lower1, upper1)
        mask_red2 = cv.inRange(hsv, lower2, upper2)
        mask_red = cv.bitwise_or(mask_red, mask_red2)

    return mask_red

def detectCargo(frame, message):
    
    #call on functions
    if message == "red":
        drawRect(createRedMask(editImage(frame)), frame, "red")
    elif message == "blue":
        drawRect(createBlueMask(editImage(frame)), frame, "blue")
    elif message == "both":
        drawRect(createRedMask(editImage(frame)), frame, "red")
        drawRect(createBlueMask(editImage(frame)), frame, "blue")

    return frame
    
camera.release()
cv.destroyAllWindows()