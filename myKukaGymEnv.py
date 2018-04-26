from pybullet_utils import moveBot, performGrasp, clean_px
from kukaGymEnv import KukaGymEnv
from gym import spaces
import pybullet_data
import pybullet as p
import gym
import numpy as np
import os
import random
import kuka
import math
import scipy.ndimage as ndimage


class MyKukaGymEnv(KukaGymEnv):
	def __init__(self,
				 urdfRoot=pybullet_data.getDataPath(),
				 actionRepeat=1,
				 isEnableSelfCollision=True,
				 renders=False,
				 isDiscrete=False,
				 maxSteps = 1000, objectName='block.urdf',
				 gridCoarsity=5,
				 objUrdfRoot=None):
		#print("KukaGymEnv __init__")
		if objUrdfRoot is not None:
			self.objUrdfRoot = objUrdfRoot
		else:
			self.objUrdfRoot = urdfRoot
		self._objectName = objectName

		self.angle_scale = 1
		self.x0, self.y0 = .7, 0
		self.xscale, self.yscale = .12, .2
		super().__init__(urdfRoot=urdfRoot,
							 actionRepeat=actionRepeat,
							 isEnableSelfCollision=isEnableSelfCollision,
							 renders=renders,
							 isDiscrete=isDiscrete,
							 maxSteps = maxSteps, objectName=objectName)

		self.gridCoarsity = gridCoarsity
		xgrid = np.linspace(self.x0, self.x0+self.xscale, gridCoarsity)
		ygrid = np.linspace(self.y0, self.y0+self.yscale, gridCoarsity)
		self.grid  = np.meshgrid(xgrid, ygrid)
		self.grid  = np.concatenate((self.grid[0][:,:,None], self.grid[1][:,:,None]), axis=2)

		
		if (self._isDiscrete):
			self.action_space = spaces.Discrete(gridCoarsity**2)
		else:
			 action_dim = 2
			 self._action_bound = 1
			 action_high = np.array([self.xscale, self.yscale])
			 self.action_space = spaces.Box(-action_high, action_high)


	def reset(self, num_steps=100):
		#print("KukaGymEnv _reset")
		self.terminated = 0
		p.resetSimulation()
		p.setPhysicsEngineParameter(numSolverIterations=150)
		p.setTimeStep(self._timeStep)
		p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"),[0,0,-1])

		p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), 0.5000000,0.00000,-.820000,0.000000,0.000000,0.0,1.0)

		xpos = self.x0 + self.xscale*(random.random()-.5)
		ypos = self.y0 + self.yscale*(random.random()-.5)
		ang = 3.14*0.5+3.1415925438*random.random() * self.angle_scale
		orn = (0,0,.9,.8)
		#print(os.path.join(self.objUrdfRoot,self._objectName))
		self.blockUid = p.loadURDF(os.path.join(self.objUrdfRoot,self._objectName), xpos,ypos,-0.15,orn[0],orn[1],orn[2],orn[3])
		
		p.setGravity(0,0,-10)
		self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
		self._envStepCounter = 0
		for _ in range(num_steps):
			p.stepSimulation()
		self._observation = self.getExtendedObservation()
		return np.array(self._observation)

	def step(self, action, graspSteps=300, liftSteps=300, downSteps=400):
		# returns
		# observation, reward, done, info
		if self._isDiscrete:
			action_onehot = np.zeros(num_act**2)
			action_onehot[action] = 1
			x, y = self.action_to_xy(action_onehot)
		else:
			x, y = action
			#x, y = x + self.x0, y + self.y0
		fingerAngle = 0.3
		for _ in range(graspSteps):
			graspAction = [0,0,0.0001,0,fingerAngle]
			self._kuka.applyAction(graspAction)
			p.stepSimulation()

		moveBot(self, x, y, .08, .3, maxSteps=downSteps)
		blockPos, _ = p.getBasePositionAndOrientation(self.blockUid)
		gripperState = p.getLinkState(self._kuka.kukaUid,self._kuka.kukaGripperIndex)
		gripperPos = gripperState[0]
		reward = 0
		#reward -= np.sqrt((x - blockPos[0])**2 + (y - blockPos[1])**2)
		performGrasp(self, graspSteps=graspSteps, liftSteps=liftSteps)
		blockPos, _ = p.getBasePositionAndOrientation(self.blockUid)
		if blockPos[2] > .05: reward += 1
		done = True
		observation = self.getExtendedObservation()
		info = None
		return(observation, reward, done, info)

	def getBlockPosition(self):
		blockPos, _ = p.getBasePositionAndOrientation(self.blockUid)
		return(blockPos)

	def actionToSim(self, a):
		return(a[0] + self.x0, a[1] + self.y0)

	def getExtendedObservation(self):
		 self._observation = self._kuka.getObservation()
		 gripperState  = p.getLinkState(self._kuka.kukaUid,self._kuka.kukaGripperIndex)
		 gripperPos = gripperState[0]
		 gripperOrn = gripperState[1]
		 blockPos,blockOrn = p.getBasePositionAndOrientation(self.blockUid)

		 invGripperPos,invGripperOrn = p.invertTransform(gripperPos,gripperOrn)
		 gripperMat = p.getMatrixFromQuaternion(gripperOrn)
		 dir0 = [gripperMat[0],gripperMat[3],gripperMat[6]]
		 dir1 = [gripperMat[1],gripperMat[4],gripperMat[7]]
		 dir2 = [gripperMat[2],gripperMat[5],gripperMat[8]]

		 gripperEul =  p.getEulerFromQuaternion(gripperOrn)
		 #print("gripperEul")
		 #print(gripperEul)
		 blockPosInGripper,blockOrnInGripper = p.multiplyTransforms(invGripperPos,invGripperOrn,blockPos,blockOrn)
		 projectedBlockPos2D =[blockPosInGripper[0],blockPosInGripper[1]]
		 blockEulerInGripper = p.getEulerFromQuaternion(blockOrnInGripper)

		 #we return the relative x,y position and euler angle of block in gripper space
		 blockInGripperPosXYEulZ =[blockPosInGripper[0],blockPosInGripper[1],blockEulerInGripper[2]]

		 self._observation.extend(list(blockInGripperPosXYEulZ))
		 self._observation.extend(list(blockPos))
		 return self._observation

	def action_to_xy(self, action):
		num_x = self.grid.shape[0]
		action_mat = action.reshape(num_x, num_y, 1)
		xy = (action_mat * self.grid).sum(axis=1).sum(axis=0)
		moveBot(self, xy[0], xy[1], .08, .3, steps=600)

		return(xy[0], xy[1])


	def render(self, mode="rgb_array", width=960, height=720, lose=False):
		if mode != "rgb_array":
			return np.array([])

		base_pos,orn = self._p.getBasePositionAndOrientation(self._kuka.kukaUid)
		view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
				cameraTargetPosition=base_pos,
				distance=self._cam_dist,
				yaw=self._cam_yaw,
				pitch=self._cam_pitch,
				roll=0,
				upAxisIndex=2)
		proj_matrix = self._p.computeProjectionMatrixFOV(
				fov=60, aspect=float(width)/height,
				nearVal=0.1, farVal=100.0)
		# this is what eats most of the time
		# the time seems to be about linear with width*height
		(_, _, px, _, _) = self._p.getCameraImage(
				width=width, height=height, viewMatrix=view_matrix,
				projectionMatrix=proj_matrix, renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
				#renderer=self._p.ER_TINY_RENDERER)


		rgb_array = np.array(px, dtype=np.uint8)
		rgb_array = np.reshape(rgb_array, (width, height, 4))

		rgb_array = rgb_array[:, :, :3]
		return rgb_array

	def clean_render(self, width=960, height=720):
		h1, h2, h3 = np.linspace(0,height, 4)[1:].astype(int)
		w1, w2 = int(width/2), int(5./6 * width)
		px = self.render(height=height,width=width)

		px1 = px[w1:w2,:h1,:]
		px2 = px[w1:w2,h1:h2,:]
		px3 = px[w1:w2,h2:h3,:]

		px1 = clean_px(px1)
		px2 = clean_px(px2)
		px3 = clean_px(px3)
		px = (px1+px2+px3)/3
		obj = np.where(np.sum(px, axis=2)<80)
		px[obj[0], obj[1]] = 0
		img = ndimage.gaussian_filter(px, sigma=(1, 1, 0), order=0)
		img[img[:,:,1]<120] = 0
		img[img[:,:,1]>120] = (0,125,0)

		return img







