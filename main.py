import cv2
import numpy as np

def main():
    
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)

    while True : 
        sucess, img = cap.read()
        cv2.imshow('Result', img)
        cv2.waitKey(1)
        


if __name__ == "__main__":
    main()