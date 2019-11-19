import cv2
import numpy as np
from matplotlib import pyplot as plt

nrows = 3
ncols = 2
KernelSizeHeight = 13
KernelSizeWidth = 13

image = cv2.imread('GMIT1.jpg')

original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
imageHarris = original.copy()
corners = cv2.goodFeaturesToTrack(gray,200,0.01,10)
imgShiTomasi = original.copy()
imgSift = original.copy()

threshold = 0.01; #number between 0 and 1
for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold*dst.max()):
            cv2.circle(imageHarris,(j,i),3,(255, 0, 0),-1)

for i in corners:
    x,y = i.ravel()
    cv2.circle(imgShiTomasi,(x,y),3,(255, 0, 0),-1)

#Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create(50)
(kps, descs) = sift.detectAndCompute(gray, None)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
#Draw keypoints
imgSift = cv2.drawKeypoints(imgSift,kps,outImage=None,color=(255, 0, 0),flags=4)

plt.subplot(nrows, ncols,1),plt.imshow(original, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(gray, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,3),plt.imshow(dst, cmap = 'gray')
plt.title('DST'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,4),plt.imshow(imageHarris, cmap = 'gray')
plt.title('Harris'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,5),plt.imshow(imgShiTomasi, cmap = 'gray')
plt.title('Tomasi'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,6),plt.imshow(imgSift, cmap = 'gray')
plt.title('Sift'), plt.xticks([]), plt.yticks([])
plt.show()
  
cv2.waitKey(0)
cv2.destroyAllWindows()