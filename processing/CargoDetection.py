import cv2 as cv
import numpy as np
import json

f = open("processing/data.json") 
data = json.load(f)

#accessing camera
source = 0
camera = cv.VideoCapture(source, cv.CAP_DSHOW)
camera.set(cv.CAP_PROP_EXPOSURE, -7) # change to -1 for internal camera, -7 for FISHEYE, -4 for Microsoft hd3000

blue_cargo = {}
red_cargo = {}

def getClosestBlue():

    if len(blue_cargo) > 0:
        return blue_cargo[max(blue_cargo.keys())]

    return 0, 0, 0, 0

def getClosestRed():

    if len(red_cargo) > 0:
        return red_cargo[max(red_cargo.keys())]

    return 0, 0, 0, 0

def getDistance(focal_length, real_width, width_in_frame):
    distance = (real_width * focal_length) / width_in_frame
   
    return distance

def getAngle(x, w):
    return (x + (w / 2) - 320) / 10

def getDistanceFromCenter(centerX, centerY):
    xDist = centerX - (data["dimensions"]["LENGTH"])
    yDist = centerY - (data["dimensions"]["WIDTH"])

    return xDist, yDist

def isCircle(cnt):
    approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
    
    _, radius = cv.minEnclosingCircle(cnt)
    contour_area = cv.contourArea(cnt)

    return len(approx) > 7 and 1.0 >= contour_area / (radius**2 * 3.14) >= .8 and contour_area > 60

def drawRect(mask, img, color, cargoDict):
    
    #Get contours on the mask
    contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cargoDict.clear()
    cargoDict.clear()

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

            if .5 <= aspect_ratio <= 2:
                
                cargoDict[cv.contourArea(contour)] = (x, y, w, h)

                distance = getDistance(630, 24.13, int(w))
                distance = format((int(distance) * 1.1) / 100, '.2f')

                cv.circle(img, center, int(w / 2), (0,255,0), 5)
                cv.putText(img, color.upper() + " BALL " + str(distance) + " METERS", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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
    if message == "Red":
        drawRect(createRedMask(editImage(frame)), frame, "red", red_cargo)
    elif message == "Blue":
        drawRect(createBlueMask(editImage(frame)), frame, "blue", blue_cargo)
    elif message == "Both":
        drawRect(createRedMask(editImage(frame)), frame, "red", red_cargo)
        drawRect(createBlueMask(editImage(frame)), frame, "blue", blue_cargo)

    return frame

def sendData(frame, message):

    drawRect(createRedMask(editImage(frame)), frame, "red", red_cargo)
    drawRect(createBlueMask(editImage(frame)), frame, "blue", blue_cargo)

    closestRed = getClosestRed()
    closestBlue = getClosestBlue()

    redX, redY, redW, redH = closestRed[0], closestRed[1], closestRed[2], closestRed[3]
    blueX, blueY, blueW, blueH = closestBlue[0], closestBlue[1], closestBlue[2], closestBlue[3]

    redData = {'x' : 0, 'y' : 0, 'centerX' : 0, 'centerY' : 0, 'dist' : 0, 'angle' : 0}
    blueData = {'x' : 0, 'y' : 0, 'centerX' : 0, 'centerY' : 0, 'dist' : 0, 'angle' : 0}

    # Store 1. x cords, 2. y cords, 3. distance from center, 4. disance from camera, and 5. angle
    if redW > 0 and (message == "Red" or message == 'Both'):
        distance = getDistance(data['dimensions']['FOCAL_LENGTH'], 24.3, redW)
        distance = format((int(distance) * 1.1) / 100, '.2f')

        centerDistX, centerDistY = getDistanceFromCenter(redX, redY)

        redData = {'x' : redX, 'y' : redY, 'centerX' : centerDistX, 'centerY' : centerDistY, 'dist' : distance, 'angle' : getAngle(redX, redW)}

    if blueW > 0 and (message == 'Blue' or message == 'Both'):
        distance = getDistance(data['dimensions']['FOCAL_LENGTH'], 24.3, blueW)
        distance = format((int(distance) * 1.1) / 100, '.2f')

        centerDistX, centerDistY = getDistanceFromCenter(blueX, blueY)

        blueData = {'x' : blueX, 'y' : blueY, 'centerX' : centerDistX, 'centerY' : centerDistY, 'dist' : distance, 'angle' : getAngle(blueX, blueW)}

    return {'red' : redData, 'blue' : blueData}
    
camera.release()
cv.destroyAllWindows()