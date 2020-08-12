import cv2
import numpy as np
import os
import copy

image = cv2.imread('D:/waterGate/image/8.jpg')
clone = image.copy()
#cv2.imshow("clone", clone)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 1)
#blurred = cv2.GaussianBlur(gray, (11, 11), 0)
binaryIMG = cv2.Canny(blurred, 20, 160)

ret, binary = cv2.threshold(binaryIMG.copy(), 127, 255, cv2.THRESH_BINARY)  
(_, cnts, _) = cv2.findContours(binaryIMG.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(clone, cnts, -1, (0, 255, 0), 2)

cnt = cnts[5]
print(len(cnts))
imgnew = cv2.drawContours(clone, cnt, -1, (0,255,0), 3)

M = cv2.moments(cnt)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
area = cv2.contourArea(cnt)
perimeter = cv2.arcLength(cnt,True)
epsilon = 0.02*perimeter
approx = cv2.approxPolyDP(cnt,epsilon,True)
imgnew1 = cv2.drawContours(clone, approx, -1, (0,0,255), 3)
imgnew1 = cv2.circle(clone, (cx, cy), 5, (0, 0, 255), -1)

print(cx, cy, area)

cv2.imshow("new", imgnew)
cv2.imshow("new1", imgnew1)
cv2.imshow("bin", binaryIMG)
cv2.imshow("blu", blurred)
cv2.imshow("gray", gray)
cv2.imshow("clone", clone)
cv2.waitKey(0)