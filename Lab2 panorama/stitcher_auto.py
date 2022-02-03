import numpy as np
import cv2 
import glob 
import imutils

image_path = "Mosaique_2014/Data/"
numero = ["1.jpg","2.jpg","3.jpg"]
images = []

for num in numero:
    img = cv2.imread(image_path+num)
    images.append(img)
    cv2.imshow("Image",img)
    cv2.waitKey(0)

imageStitcher = cv2.Stitcher_create()
error, stitched_img = imageStitcher.stitch(images)

if not error:
    
    cv2.imwrite("stitchedOutput.png", stitched_img)
    cv2.imshow("Stitched image", stitched_img)
    cv2.waitKey(0)
