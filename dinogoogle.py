import sys

#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import webbrowser
import cv2
import pyautogui
import numpy as np
import time
import sys

webbrowser.get(sys.argv[1]).open('http://google.com')  # Go to example.com

time.sleep(2)
im0 = pyautogui.screenshot('my_screenshot.png')
img0 =cv2.imread('my_screenshot.png')
pyautogui.press('space')
#img0 = cv2.imread('scr.png')
 
img1 = img0[230:255, 570:620, :] 
#cv2.imshow('image',img1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
jl=0
while(1):
	jl+=1
	print(jl)
	im0 = pyautogui.screenshot('my_screenshot'+str(jl)+'.png')
	img0 =cv2.imread('my_screenshot'+str(jl)+'.png')
	 
	img1 = img0[230:257, 580:610, :]	
	
	count = 0
	for i in range(27):
		for j in range(30):
			for k in range(3):
				if img1[i,j,k]!=255:
					count+=1         
	
	if count!=0:
		# cv2.imshow('image',img1)
		# cv2.destroyAllWindows()
		pyautogui.press('space')

