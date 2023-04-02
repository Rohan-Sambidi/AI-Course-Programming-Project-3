import importlib
import os
import json
import time
import subprocess
import time
import os



ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
								 os.path.pardir))

class JSONUtils(object):
	def __init__(self, module_name):
		self.module = module_name

	def custom_dict_hook(self, dct):
		result_dict = {}
		for objname in dct.keys():
			obj = self.import_from_name(objname)
			result_dict[obj] = dct[objname]
		return result_dict

	def import_from_name(self, object_name):
		module = importlib.import_module(self.module)
		return getattr(module, object_name, object_name)

def env_json_setup(ftypes, fcnt, env):
	GAZEBO_PARENT_DIR = os.getenv('HOME')
	food_dict = {
		"obstacles": {
			"Pizza": {"count": fcnt,
				"file":os.path.join(GAZEBO_PARENT_DIR, ".gazebo/models/pizza/meshes/model.dae")
			},
			"Hamburger": {"count": fcnt,
				"file":os.path.join(GAZEBO_PARENT_DIR, ".gazebo/models/hamburger/meshes/model.dae")
			}

		},
		"bounding": {
			"DAEBounding": {
				"count": 1,
				"file":os.path.join(GAZEBO_PARENT_DIR, ".gazebo/models/world/meshes/world_final.dae"),
				"scale":[0.4,0.4,0.4]
			}
		},
		"goal":{
			"Table":{
				"count": 2*ftypes,
				"file":os.path.join(GAZEBO_PARENT_DIR, ".gazebo/models/cafe_table/meshes/cafe_table.dae")
			}
		}
	}

	book_dict = {
		"obstacles": {
			"Book": {"count": fcnt,
			"subjects": ftypes
			}
		},
		"bounding": {
			"Wall": {
				"count":1
			}
		},
		"goal":{
			"Trolley":{
				"count": 2*ftypes
			}
		}
	}

	if env == "bookWorld":
		default_dict = book_dict
	
	elif env == "cafeWorld":
		default_dict = food_dict
		obstacles = {}
		for i in list(default_dict['obstacles'].keys())[:ftypes]:
			obstacles[i] = default_dict['obstacles'][i]

		default_dict['obstacles'] = obstacles

	else:
		raise ValueError("Env can be either bookWorld or cafeWorld")

	with open(os.path.join(ROOT_DIR, 'config/maze.json'),'w') as of:
		json.dump(default_dict, of)

def initialize_ros():

	fileHandle=open("/dev/null", "w")

	# Cleanup any previous instances of roscore.
	subprocess.call("pkill roscore", shell=True,
		stdout=fileHandle, stderr=fileHandle)
	
	# Start a new instance.
	proc = subprocess.Popen("roscore", shell=True,
			stdout=fileHandle, stderr=fileHandle)
	
	# Wait a few seconds for ROS Core to come up.
	time.sleep(5)
	return proc

def cleanup_ros(*pids):

	fileHandle=open("/dev/null", "w")
	for pid in pids:
		subprocess.call("kill -9 {}".format(pid), shell=True,
		stdout=fileHandle, stderr=fileHandle)
	# Cleanup any previous instances of roscore.
	subprocess.call("pkill roscore", shell=True,
		stdout=fileHandle, stderr=fileHandle)

if __name__ == "__main__":
	env_json_setup(3,3)