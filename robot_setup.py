import gym
from pybullet_utils import *
#from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
import math
import gym
import numpy as np
import pybullet as p
import pybullet_data
import pickle
from kukaCamGymEnv import KukaCamGymEnv
from kuka_diverse_object_gym_env import KukaDiverseObjectEnv


objects = ['block.urdf', 'cube_small.urdf', 'duck_vhacd.urdf']

#env = KukaGymEnv(renders=True, isDiscrete=True, objectName='block.urdf')
#env = KukaCamGymEnv(renders=True, isDiscrete=True)
env = KukaDiverseObjectEnv(renders=True, isDiscrete=True)

W=720
H=960
depth_training_set = np.zeros((W,H,0))
rgb_training_set = np.zeros((W,H,3,0))
target_set = np.zeros((4,0))

collected_ex = 0
num_ex = 10

while collected_ex < num_ex:
	env.reset()
	blockPos,blockOrn=p.getBasePositionAndOrientation(env.blockUid)
	blockAngle = np.mod(p.getEulerFromQuaternion(blockOrn)[2], 2*np.pi)
	targetAngle = -blockAngle + np.pi/2
	env._kuka.endEffectorAngle = targetAngle
	changeAngle(env, targetAngle)
	moveBot(env, blockPos[0], blockPos[1], blockPos[2]+.15, .3, steps=600)
	perform_grasp(env, 300)
	block_pos = p.getBasePositionAndOrientation(env.blockUid)[0]
	if block_pos[2] > .05:
		target = np.r_[blockPos, targetAngle][:,None]
		collected_ex += 1
		print(collected_ex)
		rgb_array = env.get_rgb(dist=.25,pitch=-90)[:,:,:3]
		depth_array = env.get_depth(dist=.25,pitch=-90)
		rgb_training_set = np.concatenate((rgb_training_set, rgb_array[:,:,:,None]), axis=3)
		depth_training_set = np.concatenate((depth_training_set, depth_array[:,:,None]), axis=2)
		target_set = np.concatenate((target_set, target), axis=1)


#rgb_array = env.get_rgb()[:,:600,:3]
rgb_array = env.get_rgb(dist=.25,pitch=-90,
						yaw=env._cam_yaw,
						W = 480, H=360)[:,:,:3]

pickle.dump(rgb_array, open('rgb_test.p', 'wb'))

depth_array = env.get_depth(dist=.25,pitch=-90,yaw=env._cam_yaw,
							W=28, H=28)
pickle.dump(depth_array, open('depth_test.p', 'wb'))

plt.ion()
img = pickle.load(open('rgb_test.p', 'rb'))
image = plt.imshow(img,interpolation='none',animated=True,label="blah")
ax = plt.gca()


plt.ion()
img = pickle.load(open('depth_test.p', 'rb'))
image = plt.imshow(img,interpolation='none',animated=True,label="blah")
ax = plt.gca()







