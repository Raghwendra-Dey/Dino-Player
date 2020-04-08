import sys
import cv2
import webbrowser
import cv2
import pyautogui
import numpy as np
import time
import sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from collections import deque
from skimage import transform
from tqdm import tqdm

class env_dino:
	def __init__(self):
		self.is_game_over = False
		self.is_new_episode = False
		self.prev_state = None
		self.curr_state = None

		self.state_size = [84, 84, 4];
		self.action_size = 2;
		
		self.chromeOptions = Options()
		self.chromeOptions.add_argument("--window-size=500,960")
		self.driver = webdriver.Chrome("/home/debjoy/Downloads/chromedriver (2)", chrome_options=self.chromeOptions)

		self.stack_size = 4
		self.stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(self.stack_size)], maxlen=4)

	def game_init(self):
		self.driver.get('http://google.com')
		time.sleep(2)

	def pre_process(self, img):
		imcrop = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)[210:330, 120:530, :]
		return imcrop

	def pre_process_2(self, img):
		frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		cell_col = frame[0, 0]
		if(cell_col<128):
			frame = cv2.bitwise_not(frame)

		normalized_frame = frame/255.0
		preprocessed_frame = transform.resize(normalized_frame, [84,84])
		return preprocessed_frame

	def stack(self, img):
		frame = self.pre_process_2(img)
		if self.is_new_episode:
			self.is_new_episode = False
			# Clear our stacked_frames
			self.stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(self.stack_size)], maxlen=4)
			
			# Because we're in a new episode, copy the same frame 4x
			self.stacked_frames.append(frame)
			self.stacked_frames.append(frame)
			self.stacked_frames.append(frame)
			self.stacked_frames.append(frame)
			
			# Stack the frames
			stacked_state = np.stack(self.stacked_frames, axis=2)
			
		else:
			# Append frame to deque, automatically removes the oldest frame
			self.stacked_frames.append(frame)

			# Build the stacked state (first dimension specifies different frames)
			stacked_state = np.stack(self.stacked_frames, axis=2) 
		
		return stacked_state

	def game_get_state(self):
		im = np.array(pyautogui.screenshot())
		im_pro = self.pre_process(im)
		self.prev_state = self.curr_state
		self.curr_state = im_pro
		return im_pro

	def reset(self):
		self.is_game_over = False
		# self.game_init()
		pyautogui.press('space')
		# time.sleep(1)
		state = self.game_get_state()
		f_state = self.stack(state)
		return f_state;

	def get_reward(self, action, new_state):
		r = 0
		if self.is_game_over:
			r = -500
		# elif action == 1:
		# 	b = np.min(new_state[80:90, 10:90])
		# 	print("b"+str(b))
		# 	if b<128:
		# 		r = 10
		else:
			r = 1
		cv2.imwrite("pics/"+str(r)+str(np.random.rand(1))+"_.png", new_state)
		print("Reward: "+str(r))
		return r

	def step(self, action):
		if action == 0:
			pyautogui.press('up')
		# if action == 1:
		# 	pyautogui.press('down')

		# method = cv2.TM_SQDIFF_NORMED
		# small_image = cv2.imread('end3.png')
		# large_image = self.curr_state
		# cv2.imwrite("abc.png", large_image)
		# result = cv2.matchTemplate(large_image, small_image, method)
		# print(result)
		# if result[result.shape[0]-1][0] > 0.065:
		# 	self.is_game_over = True
		# 	self.is_new_episode = True

		if np.array_equal(self.prev_state,self.curr_state):
			self.is_game_over = True
			self.is_new_episode = True

		new_state = self.game_get_state()
		reward = self.get_reward(action, new_state)
		done = self.is_game_over
		info = None

		f_new_state = self.stack(new_state)
		# return new_state, reward, done, info
		return f_new_state, reward, done, info
		

# env = env_dino()
# done = False
# env.game_init()
# i=0
# while True:
# 	im = env.reset()

# 	while not done:
# 		i+=1
# 		new_state, reward, done, info = env.step(1)
# 		cv2.imwrite('abc'+str(i)+'.png', new_state)
# 		print("Done "+str(done))
# 	done = False

# def game_new_episode():
# 	global is_game_over

# 	is_game_over = False
# 	game_init()
# 	pyautogui.press('space')
# 	time.sleep(1)

# def game_start():
# 	global is_game_over

# 	is_game_over = False
# 	game_init()
# 	pyautogui.press('space')
# 	time.sleep(1)

# def game_get_state():
# 	global prev_state
# 	global curr_state
# 	im = np.array(pyautogui.screenshot())
# 	imcrop = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)[190:330, 120:550, :]
# 	prev_state = curr_state
# 	curr_state = imcrop
# 	return imcrop

# def create_environment():	
# 	game_init()
# 	# Here our possible actions
# 	jump = [1, 0, 0]
# 	duck = [0, 1, 0]
# 	none = [0, 0, 1]
# 	possible_actions = [jump, duck, none]
	
# 	return possible_actions

# def game_make_action(action):
# 	print("Action: "+str(action))
# 	if action[0] == 1:
# 		pyautogui.press('up')
# 	elif action[1] == 1:
# 		pyautogui.press('down')

# 	state = game_get_state()
# 	reward = get_reward()
# 	return(reward)

# def get_reward():
# 	global is_game_over

# 	if is_game_over:
# 		return -1000
# 	else:
# 		return 1

# def find_is_episode_finished():
# 	global is_game_over
# 	global prev_state 
# 	global curr_state

# 	if np.array_equal(prev_state,curr_state):
# 		print("Game Finished")
# 		is_game_over = True

# def is_episode_finished():
# 	global is_game_over
# 	return is_game_over

# def takepic():
# 	im = np.array(pyautogui.screenshot())
# 	imcrop = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)[190:330, 120:550, :]
# 	cv2.namedWindow("win", 0)
# 	cv2.imshow("win", imcrop)
# 	cv2.waitKey(0)

# def play_example():
# 	global is_game_over
# 	for _ in range(5):
# 		game_start()
# 		state = None
# 		while not is_game_over:
# 			state = game_get_state()
# 			find_is_episode_finished()
# 			print("is_game_over: "+str(is_game_over))

# 		print("Finished")

