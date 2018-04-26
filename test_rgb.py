import matplotlib.pyplot as plt 
import pybullet
import time
import gym
from pybullet_utils import *
#from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
import math
import gym
import numpy as np
import pybullet as p
import pybullet_data
import pickle

plt.ion()

img = pickle.load(open('rgb_test.p', 'rb'))
#img = [[1,2,3]*50]*100#np.random.rand(200, 320)
#img = [tandard_normal((50,100))
image = plt.imshow(img,interpolation='none',animated=True,label="blah")
ax = plt.gca()


plt.ion()

img = pickle.load(open('depth_test.p', 'rb'))
#img = [[1,2,3]*50]*100#np.random.rand(200, 320)
#img = [tandard_normal((50,100))
image = plt.imshow(img,interpolation='none',animated=True,label="blah")
ax = plt.gca()



while(1): 
  for yaw in range (0,360,10) :
    rgb = env.get_rgb()[:,:,:3]

    image.set_data(rgb)
    ax.plot([0])
    #plt.draw()
    #plt.show()
    plt.pause(0.01)

pybullet.resetSimulation()










