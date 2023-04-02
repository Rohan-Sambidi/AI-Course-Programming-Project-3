#!/usr/bin/env python3
# encoding: utf-8

__copyright__ = "Copyright 2021, AAIR Lab, ASU"
__authors__ = ["Naman Shah"]
__credits__ = ["Siddharth Srivastava"]
__license__ = "MIT"
__version__ = "1.0"
__maintainers__ = ["Naman Shah"]
__contact__ = "aair.lab@asu.edu"
__docformat__ = 'reStructuredText'

from hw3.srv import *
import rospy
from action_server import RobotActionsServer
import subprocess
import os
import rospy
from gen_maze import *
from maze_objects import *
from utils import JSONUtils, env_json_setup
import json
import pickle
import time
import copy
from std_msgs.msg import String

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))


class Server(object):

    def __init__(self, maze, blocked_edges, book_info):
        if maze is not None:
            self.maze = maze
            self.grid_size, self.cell_size, self.blocked_edges = self.maze.getMazeState()
            self.init_x, self.init_y, _ = self.maze.initState()
            self.goalstates = self.maze.getGoalState()
        self.cell_size = 0.5
        self.blocked_edges = blocked_edges
        self.book_info = book_info
        self.actionserver = None
        self.maze_default = None
        self.cache_variables = {}
    
    def start(self):

        rospy.init_node('get_successor_server', log_level=rospy.ERROR)
        rospy.Service('remove_blocked_edge', RemoveBlockedEdgeMsg, self.remove_blocked_edge)
        rospy.Service('reset_world',ResetWorldMsg, self.handle_reset_world)
        rospy.Service('check_is_edge', CheckEdge, self.check_is_edge)
        rospy.Service("generate_maze", GenerateMaze, self.ros_generate_maze)
        print("Ready!")
        rospy.spin()

    def check_is_edge(self, req):
        
        # Out of bounds check
        if not float(self.init_x) <= req.x2 <= self.grid_size[0] \
            or not float(self.init_y) <= req.y2 <= self.grid_size[1]:
                return 0
        # Blocked edges check
        elif (req.x1, req.y1, req.x2, req.y2) in self.blocked_edges \
            or (req.x2, req.y2, req.x1, req.y1) in self.blocked_edges:
            # edge found in blocked edges
            return 0
        else:
            # edge NOT found in blocked edges
            return 1

    def handle_reset_world(self, req):
        
        _, _, beold = self.maze.getMazeState()

        grid_dimensions = copy.deepcopy(self.cache_variables["grid_dimensions"])
        maze_objects = copy.deepcopy(self.cache_variables["maze_objects"])
        cell_size = copy.deepcopy(self.cache_variables["cell_size"])
        seed = copy.deepcopy(self.cache_variables["seed"])
        blocked_edges = copy.deepcopy(self.cache_variables["blocked_edges"])
        book_info = copy.deepcopy(self.cache_variables["book_info"])
        env = copy.deepcopy(self.cache_variables["env"])

        self.maze = MazeGenerator(grid_dimensions,
                                    maze_objects,
                                    cell_size,
                                    seed,
                                    blocked_edges,
                                    book_info,
                                    env=env)


        _, _, benew = self.maze.getMazeState()
        self.actionserver.current_state = self.actionserver.generate_init_state()
        self.actionserver.status_publisher.publish(String(data="+++++++++++++++++++++Reset world++++++++++++++++++++++"))
        return 1
    
    def ros_generate_maze(self,req):

        blocked_edges = []
        if(req.env == "bookWorld"):
            grid_dimensions = [(req.objtypes * 3)//self.cell_size]*2
        else:
            dimension_x = 3
            dimension_y = 12
            grid_dimensions = [dimension_x/self.cell_size,dimension_y/self.cell_size]
            if os.path.isfile(os.path.join(ROOT_DIR, 'config/env_precompute.pkl')):
                with open(os.path.join(ROOT_DIR, 'config/env_precompute.pkl'), 'rb') as envfile:
                    blocked_edges = pickle.load(envfile)
            
            blocked_edges_temp = (np.array(blocked_edges['map'])*float(blocked_edges['scale'])).tolist()
            blocked_edges = []

            for i in blocked_edges_temp:
                blocked_edges.append(tuple(i))

        env_json_setup(req.objtypes, req.objcount, req.env)

        maze_objects = json.load(open(os.path.join(ROOT_DIR,'config/maze.json'),'r'),
                               object_hook=JSONUtils('maze_objects').custom_dict_hook)
        
        self.cache_variables["grid_dimensions"] = copy.deepcopy(grid_dimensions)
        self.cache_variables["maze_objects"] = copy.deepcopy(maze_objects)
        self.cache_variables["cell_size"] = copy.deepcopy(self.cell_size)
        self.cache_variables["blocked_edges"] = copy.deepcopy(blocked_edges)
        self.cache_variables["seed"] = copy.deepcopy(req.seed)
        self.cache_variables["book_info"] = copy.deepcopy(self.book_info)
        self.cache_variables["env"] = copy.deepcopy(req.env)
        
        self.maze = MazeGenerator(grid_dimensions,
                                    maze_objects,
                                    self.cell_size,
                                    req.seed,
                                    blocked_edges,
                                    self.book_info,
                                    env=req.env)
        self.maze_default = MazeGenerator(grid_dimensions,
                                    maze_objects,
                                    self.cell_size,
                                    req.seed,
                                    blocked_edges,
                                    self.book_info,
                                    env=req.env)
        self.grid_size, self.cell_size, self.blocked_edges = self.maze.getMazeState()
        self.goalstates = self.maze.getGoalState()
        self.init_x, self.init_y, _ = self.maze.initState()
        time.sleep(5.0)

        if self.actionserver is None:
            self.actionserver = RobotActionsServer(self.maze.env_objects,
                                                ROOT_DIR,
                                                headless=True,
                                                random_seed=req.seed)
            self.actionserver.register_ros_functions()
        else:
            self.actionserver = RobotActionsServer(self.maze.env_objects,
                                                ROOT_DIR,
                                                headless=True,
                                                random_seed=req.seed)
                         

        return GenerateMazeResponse(0)

    def remove_blocked_edge(self, req):
        return RemoveBlockedEdgeMsgResponse(self.maze.delete_edge(req.objname))

def generate_maze(subjects, books, seed, headless, env):

    rospy.wait_for_service('generate_maze')

    response =  rospy.ServiceProxy('generate_maze', GenerateMaze) \
        (subjects, books, seed, headless, env)
        
    if response.done != 0:
        
        raise Exception("Cannot generate the maze")

def initialize_planning_server():

    fileHandle = open("/dev/null", "w")
    p = subprocess.Popen("rosrun hw3 server.py", shell=True)
    return p

def main():

    blocked_edges = []
    if os.path.isfile(os.path.join(ROOT_DIR, 'config/env_precompute.pkl')):
        with open(os.path.join(ROOT_DIR, 'config/env_precompute.pkl'), 'rb') as envfile:
            blocked_edges = pickle.load(envfile)

    with open(os.path.join(ROOT_DIR, 'config/book_metadata.json'),'r') as fp:
        book_info = json.load(fp)

    blocked_edges = (np.array(blocked_edges['map'])*float(blocked_edges['scale'])).tolist()

    Server(maze=None, blocked_edges=blocked_edges, book_info=book_info).start()

if __name__ == "__main__":
    main()
