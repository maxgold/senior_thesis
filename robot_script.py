import gym
from pybullet_utils import *
#from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
import math
import gym
import numpy as np
import pybullet as p
import pybullet_data
#from kukaGymEnv import KukaGymEnv


def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts


if __name__ == '__main__':
	from sys import argv
	myargs = getopts(argv)
	if '-t' in myargs:
		trials = int(myargs['-t'])
	else:
		trials = 100
	if '-o' in myargs:
		objectName=myargs['-o']
	else:
		objectName= 'block.urdf'
	if '-r' in myargs:
		render= myargs['-r'] == 'True'
	else:
		render=True

	print(render)
	env = KukaGymEnv(renders=render, isDiscrete=True, objectName=objectName)
	success = 0
	for i in range(trials):
		
		env.reset()
		blockPos,blockOrn=p.getBasePositionAndOrientation(env.blockUid)
		blockAngle = np.mod(p.getEulerFromQuaternion(blockOrn)[2], 2*np.pi)
		targetAngle = -blockAngle + np.pi/2
		env._kuka.endEffectorAngle = targetAngle
		changeAngle(env, targetAngle)
		moveBot(env, blockPos[0], blockPos[1], blockPos[2]+.23, .3, steps=600)
		perform_grasp(env, 300)
		block_pos = p.getBasePositionAndOrientation(env.blockUid)[0]
		if block_pos[2] > .05: success += 1
		works = block_pos[2] > .05
		print(i, works)

	print(success, trials, float(success)/trials)





