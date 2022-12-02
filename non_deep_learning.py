from local_utils import interval_mapping
# import pytesseract
import numpy as np
import imutils
import cv2
import easyocr

def preprocess_img(file):
    img = cv2.imread(file)
    img = cv2.resize(img, (620,480))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey scale
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    edged = cv2.Canny(gray, 15, 200) #Perform Edge detection

    return img, edged, gray

def find_contours(edged):
    contours = cv2.findContours(edged.copy(),cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours,key=cv2.contourArea, reverse = True)[:10]
    screenCnt = 0
    detected = 0

    for c in contours:     
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            detected = 1
            break

    return screenCnt, detected

def create_mask_image(img, gray, screenCnt):
    mask = np.zeros(gray.shape,np.uint8)
    cv2.drawContours(mask, [screenCnt], 0, 255, -1)
    cv2.bitwise_and(img,img,mask=mask)

    return mask

def read_plate_number(mask, img):
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = img[topx:bottomx+1, topy:bottomy+1]

    cropped_text = interval_mapping(Cropped, 0.0, 1.0, 0, 255)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(np.uint8(cropped_text))

    return result, Cropped

def plate_recognition(file):
    img, edged, gray = preprocess_img(file)

    screenCnt, detected = find_contours(edged)

    if (detected == 1):

        mask = create_mask_image(img, gray, screenCnt)

        result, Cropped = read_plate_number(mask, img)
        finalResult = ""

        for i in range (len(result)):
            finalResult += result[i][1]
            if i == 2:
                break

        img = imutils.resize(img, width=250)

        return Cropped, finalResult
    
    notDetected = cv2.imread("./dataset/not_detected.jpg")
    return notDetected, "Plate Number not Detected!"
            



