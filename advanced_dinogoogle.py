import sys

#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import webbrowser
import cv2
import pyautogui
import numpy as np
import time
import sys

def match(contours, prev_contours, img):
	for cnt in contours:	
		(x,y),radius = cv2.minEnclosingCircle(cnt)
		center = (int(x),int(y))
		radius = int(radius)

		matchx = 10000
		matchcentre = []
		matchradius = 0
		match = False

		for pcnt in prev_contours:
			(px,py),pradius = cv2.minEnclosingCircle(pcnt)
			pcenter = (int(px),int(py))
			pradius = int(pradius)

			if px > x and px < matchx:
				matchx = px
				matchcentre = pcenter
				matchradius = pradius
				match = True


		if match:
			img = cv2.circle(img,center,radius,(0,255,0),2)
			speed = min(matchx-x,120)
			print("frame speed = "+str(speed)+"/s")
			realx = x+540
			newx = realx-speed
			print("predicted: "+str(newx))
			barrier = 560+40*(speed/80)
			print(barrier)
			if newx<barrier and y>40:
				return True
	return False

			

webbrowser.open('http://google.com')  # Go to example.com

time.sleep(2)
im0 = pyautogui.screenshot('my_screenshot.png')
img0 =cv2.imread('my_screenshot.png')
pyautogui.press('space')

flag=0
contours = []
prev_contours = []

while(1):
	flag+=1
	im0 = pyautogui.screenshot('my_screenshot.png')
	img0 =cv2.imread('my_screenshot.png')
	 
	img1 = img0[230:257, 580:610, :]
	img = img0[170:280, 540:1010, :]

	imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	imgray = cv2.blur(imgray,(10, 10))

	edged = cv2.Canny(imgray, 30, 200)
	kernel = np.ones((6,6),np.uint8)
	edged = cv2.dilate(edged, kernel,iterations = 4)

	ret, thresh = cv2.threshold(edged, 127, 255, 0)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	should_turn = match(contours, prev_contours, img)
	prev_contours = contours
	if should_turn:
		pyautogui.press('space')
	previmg = img
	if (previmg==img).all() and len(contours)>=3:
		print("GAME OVER !!")
		flag=0
		contours = []
		prev_contours = []
		pyautogui.press('space')

