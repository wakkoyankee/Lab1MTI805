import cv2
sift = cv2.SIFT_create()

image_path = "Mosaique_2014/Data/"
img1 = cv2.imread(image_path + "1.jpg")
img2 = cv2.imread(image_path + "2.jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

def siftEntre2images(img1,img2):

    bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
    keypoints_1 , descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2 , descriptors_2 = sift.detectAndCompute(img2, None)
    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x:x.distance)
    img3 = cv2.drawMatches(img1,keypoints_1, img2, keypoints_2, matches,img2, flags=2)
    return img3

img3 = siftEntre2images(img1, img2)
cv2.imwrite("SIFT 1 et 2 tsst.jpg",img3)