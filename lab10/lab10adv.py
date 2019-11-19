import cv2
import numpy as np
from matplotlib import pyplot as plt
from drawMatches import drawMatches

img1 = cv2.imread('GMIT1.jpg')
img2 = cv2.imread('GMIT2.jpg')

# Initiate SIFT detector
orb = cv2.ORB()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
#img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)
img3 = drawMatches(img1,kp1,img2,kp2,matches[:20])

plt.imshow(img3),plt.show()
