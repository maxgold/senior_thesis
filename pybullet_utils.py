import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print ("current_dir=" + currentdir)
os.sys.path.insert(0,currentdir)

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import kuka
import random
import pybullet_data
from pkg_resources import parse_version

largeValObservation = 100



def performGrasp(env, graspSteps=100,liftSteps=300):
  fingerAngle = 0.3
  for _ in range(graspSteps):
    graspAction = [0,0,0.0001,0,fingerAngle]
    env._kuka.applyAction(graspAction)
    p.stepSimulation()
    fingerAngle = fingerAngle-(0.3/100.)
    if (fingerAngle<0):
      fingerAngle=0
  for _ in range(liftSteps):
    graspAction = [0,0,0.001,0,fingerAngle]
    env._kuka.applyAction(graspAction)
    p.stepSimulation()
    blockPos,blockOrn=p.getBasePositionAndOrientation(env.blockUid)

def changeAngle(env, targetAngle):
  for _ in range(100):
    p.setJointMotorControl2(env._kuka.kukaUid,7,p.POSITION_CONTROL,
      targetPosition=targetAngle,force=env._kuka.maxForce)
    p.stepSimulation()

def moveBot(env, x, y, z, fingerAngle, maxSteps=300):
  done  = False
  count = 0
  while not done:
    count += 1
    prev_pos = p.getLinkState(env._kuka.kukaUid,env._kuka.kukaEndEffectorIndex)[0]

    env._kuka.endEffectorPos[0] = x
    if (env._kuka.endEffectorPos[0]>0.65):
      env._kuka.endEffectorPos[0]=0.65

    if (env._kuka.endEffectorPos[0]<0.50):
      env._kuka.endEffectorPos[0]=0.50

    env._kuka.endEffectorPos[1] = y
    if (env._kuka.endEffectorPos[1]<-0.17):
      env._kuka.endEffectorPos[1]=-0.17
    if (env._kuka.endEffectorPos[1]>0.22):
      env._kuka.endEffectorPos[1]=0.22

    env._kuka.endEffectorPos[2] = z

    pos = env._kuka.endEffectorPos
    orn = p.getQuaternionFromEuler([0, -math.pi, 0]) # -math.pi,yaw])
    
    jointPoses = p.calculateInverseKinematics(env._kuka.kukaUid,env._kuka.kukaEndEffectorIndex,pos,orn,jointDamping=env._kuka.jd)


    for i in range (env._kuka.kukaEndEffectorIndex+1):
      p.setJointMotorControl2(bodyUniqueId=env._kuka.kukaUid,jointIndex=i,controlMode=p.POSITION_CONTROL,targetPosition=jointPoses[i],targetVelocity=0,force=env._kuka.maxForce,maxVelocity=env._kuka.maxVelocity, positionGain=0.3,velocityGain=1)
    p.setJointMotorControl2(env._kuka.kukaUid,7,p.POSITION_CONTROL,targetPosition=env._kuka.endEffectorAngle,force=env._kuka.maxForce)
    p.setJointMotorControl2(env._kuka.kukaUid,8,p.POSITION_CONTROL,targetPosition=-fingerAngle,force=env._kuka.fingerAForce)
    p.setJointMotorControl2(env._kuka.kukaUid,11,p.POSITION_CONTROL,targetPosition=fingerAngle,force=env._kuka.fingerBForce)

    p.setJointMotorControl2(env._kuka.kukaUid,10,p.POSITION_CONTROL,targetPosition=0,force=env._kuka.fingerTipForce)
    p.setJointMotorControl2(env._kuka.kukaUid,13,p.POSITION_CONTROL,targetPosition=0,force=env._kuka.fingerTipForce)
    #print(ind)
    p.stepSimulation()
    next_pos = p.getLinkState(env._kuka.kukaUid,env._kuka.kukaEndEffectorIndex)[0]
    diff = np.array([next_pos[i] - prev_pos[i] for i in range(len(prev_pos))])
    if np.all(np.abs(diff) < 1e-10):
      done = True
    if count >= maxSteps:
      done = True

def clean_px(px):
  inds = np.where(np.sum(px, axis=2)==0)
  px[:,:,:] = (0,125,0)
  px[inds[0], inds[1], :] = (0,0,0)
  return px.astype(int)




