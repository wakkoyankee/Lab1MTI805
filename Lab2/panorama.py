import cv2
import numpy as np
import matplotlib.pyplot as plt

root = "Images/"
image1 = cv2.imread(root + "foto1A.jpg")
image2 = cv2.imread(root + "foto1B.jpg")

cv2.imshow("img1",image1)
cv2.imshow("img2",image2)
cv2.waitKey(0)  
cv2.destroyAllWindows() 

image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

#method = cv2.xfeatures2d.SIFT_create()
#method = cv2.BRISK_create()
method = cv2.ORB_create()

################  SIFT, SURF, BRISK, ORB  #################

(keypoints1, features1) = method.detectAndCompute(image1_gray,None)
(keypoints2, features2) = method.detectAndCompute(image2_gray,None)

fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,8), constrained_layout=False)
ax1.imshow(cv2.drawKeypoints(image1_gray,keypoints1,None,color=(0,255,0)))
ax1.set_xlabel("(a)", fontsize=14)
ax2.imshow(cv2.drawKeypoints(image2_gray,keypoints2,None,color=(0,255,0)))
ax2.set_xlabel("(b)", fontsize=14)

plt.show()

########## KEY POINTS MATCH, BFMATCHER, KNN #################

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # BRISK & ORB
#bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) # SIFT

best_matches = bf.match(features1,features2)
rawMatches = sorted(best_matches, key = lambda x:x.distance)

fig = plt.figure(figsize=(20,8))
img3 = cv2.drawMatches(image1,keypoints1,image2,keypoints2,rawMatches[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3)
plt.show()

################ HOMOGRAPHY ###################""

keypoints1 = np.float32([kp.pt for kp in keypoints1])
keypoints2 = np.float32([kp.pt for kp in keypoints2])
reprojThresh = 4

if len(rawMatches) > 4:

    # construct the two sets of points
    ptsA = np.float32([keypoints1[m.queryIdx] for m in rawMatches])
    ptsB = np.float32([keypoints2[m.trainIdx] for m in rawMatches])
    
    # estimate the homography between the sets of points
    (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)



###### WARP PERSPECTIVE ###############
width = image1.shape[1] + image2.shape[1]
height = image1.shape[0] + image2.shape[0]

result = cv2.warpPerspective(image1, H, (width, height))
result[0:image2.shape[0], 0:image2.shape[1]] = image2

plt.figure(figsize=(20,10))
plt.imshow(result)

plt.axis('off')
plt.show()