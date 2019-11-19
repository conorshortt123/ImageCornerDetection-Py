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
blurthree = cv2.GaussianBlur(gray,(3, 3),0)
blurthirteen = cv2.GaussianBlur(gray,(13, 13),0)
blurfive = cv2.GaussianBlur(gray,(5, 5),0)

sobelHorizontal = cv2.Sobel(blurfive,cv2.CV_64F,1,0,ksize=5) # x dir
sobelVertical = cv2.Sobel(blurfive,cv2.CV_64F,0,1,ksize=5) # y dir

sobelSum = sobelHorizontal + sobelVertical

canny = cv2.Canny(gray,100,200)

plt.subplot(nrows, ncols,1),plt.imshow(original, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(gray, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,3),plt.imshow(blurthree, cmap = 'gray')
plt.title('3x3 blur'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,4),plt.imshow(blurthirteen, cmap = 'gray')
plt.title('13x13 blur'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,5),plt.imshow(sobelSum, cmap = 'gray')
plt.title('sobel'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,6),plt.imshow(canny, cmap = 'gray')
plt.title('canny'), plt.xticks([]), plt.yticks([])
plt.show()
  
cv2.waitKey(0)
cv2.destroyAllWindows()