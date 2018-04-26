import numpy as np
from myKukaGymEnv import MyKukaGymEnv
import pybullet_data
import pybullet as p
import pandas as pd
import pickle

if __name__ == '__main__':
	urdf_root = '/Users/maxgold/Documents/Princeton/Senior/thesis/code/mycode/urdfs/dexnet'
	object_name = 'ChickenSoup_800_tex.urdf'
	env = MyKukaGymEnv(renders=False, isDiscrete=False, 
					  objUrdfRoot=urdf_root, objectName=object_name)


	t = pd.read_csv('obj_list.csv')
	object_files = list(t.Object)
	res = pd.DataFrame(columns=['Object', 'x_perturb', 'y_perturb', 'successes', 'trials'])

	env = env.unwrapped
	env.seed(1)
	env.reset()

	max_p = 2
	xgrid = np.linspace(-.01*max_p, .01*max_p, 2*max_p+1)
	ygrid = np.linspace(-.01*max_p, .01*max_p, 2*max_p+1)
	perturb  = np.meshgrid(xgrid, ygrid)

	ind = 0
	trials = 50
	for object_name in object_files:
		print(object_name)
		env._objectName = object_name
		for (dx, dy) in zip(perturb[0].flatten(), perturb[1].flatten()):
			successes = 0
			for _ in range(trials):
				env.reset(500)
 				action = env.getBlockPosition()[:2]
				action = (action[0]+dx, action[1]+dy)
				_, s, _, _ = env.step(action, graspSteps=500, liftSteps=500, downSteps=500)
				successes += s
			res.loc[ind] = [object_name, round(dx,2), round(dy,2), successes, trials]
			print(object_name, '  ',round(dx,2), '  ', round(dy,2), '  ', successes, successes/trials)

			ind += 1
		with open('object_trials.p', 'wb') as f:
			pickle.dump(res, f)

