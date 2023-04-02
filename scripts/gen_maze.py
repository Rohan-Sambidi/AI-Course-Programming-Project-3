#!/usr/bin/env python3
# encoding: utf-8

__copyright__ = "Copyright 2021, AAIR Lab, ASU"
__authors__ = ["Chirav Dave", "Kiran Prasad", "Naman Shah"]
__credits__ = ["Siddharth Srivastava"]
__license__ = "MIT"
__version__ = "1.0"
__maintainers__ = ["Naman Shah"]
__contact__ = "aair.lab@asu.edu"
__docformat__ = 'reStructuredText'

import json

import numpy as np
from abc import *
import os

import rospy

class MazeObject:
    __metaclass__ = ABCMeta
    '''
    Abstract class for maze objects
    __init__ requires coordinates of the object to be inserted
    add_object should generate the XML for the object being created
    add_object_description adds
    '''
    @abstractmethod
    def __init__(self, grid_dims, out_xml, dae, coords=(0,0,0), name="my_mesh", scale=(1,1,1), id=0, static=False):
        self.x, self.y,self.z = coords
        self.f_out = out_xml
        self.name = name
        self.dae = dae
        self.scale = scale
        self.collision = self.dae
        self.id=id
        self.static = int(static)
        if os.path.isfile(self.dae): 
            self.dae = os.path.abspath(self.dae)

    @abstractmethod
    def add_object(self):
        self.f_out.write('<model name="{}_{}">\n'.format(self.name, self.id)+\
                         '<pose frame=''>{} {} {} 0 0 0</pose>\n'.format(self.x, self.y, self.z)+\
                         '<scale>{} {} {}</scale>\n'.format(self.scale[0], self.scale[1], self.scale[2])+\
                         '<link name="link">\n'+\
                         '<pose frame=''>{} {} {} 0 0 0</pose>\n'.format(self.x, self.y, self.z)+\
                         '<velocity>0 0 0 0 -0 0</velocity>\n'+\
                         '<acceleration>0 0 0 0 -0 0</acceleration>\n'+\
                         '<wrench>0 0 0 0 -0 0</wrench>\n'+\
                         '</link>\n</model>')

    @abstractmethod
    def add_object_description(self):
        self.f_out.write('<model name="{}_{}">\n<pose>0 0 0 0 0 0</pose>\n'.format(self.name, self.id)+\
                         '<static>{}</static>\n<link name="body">\n'.format(self.static)+\
                         '<inertial><mass>0.005</mass><inertia></inertia><pose frame=> 0 0 0 0 -0 0</pose>\n'+\
                         '</inertial><self_collide>0</self_collide><kinematic>0</kinematic><gravity>1</gravity>'+\
                         '<collision name="body_collide">\n<geometry>\n<mesh>'+\
                         '<uri>file://{}</uri><scale> {} {} {} </scale></mesh>\n'.format(self.dae,self.scale[0],self.scale[1],self.scale[2])+\
                         '</geometry>'+\
                         '<surface><contact><ode/></contact><friction><ode><mu>1000</mu><mu2>1000</mu2></ode>\n</friction>\n</surface>\n</collision>\n'+\
                         '<visual name="visual"><geometry>\n<mesh>'+\
                         '<uri>file://{}</uri><scale> {} {} {} </scale></mesh>\n'.format(self.collision,self.scale[0],self.scale[1],self.scale[2])+\
                         '</geometry>'+\
                         '\n</visual>'+\
                         '\n</link>\n</model>\n')

class MazeGenerator(object):
    '''
    Object dict format:
    {
        'goal':MazeObject
        'obstacles':[]->list(MazeObject)
        'bounding':[]->list(MazeObject)
    }
    'bounding' objects will demarcate the areas of the maze; walls for example
    'bounding' needs to pass the pose coordinates as well. This will not be 
    supplied by the MazeGenerator
    'obstacles' will be randomly generated within the available areas of the maze
    'goal' will be generated at the end after all the obstacles have been created

    The position of the goal location will be determined from the available edges 
    after the obstacles have been placed

    '''
    def __init__(self,
                 grid_dims=None,
                 objects=[],
                 cell_size=0.5,
                 maze_seed=None,
                 blocked_edges=[],
                 book_info=None,
                 env="cafeWorld"):

        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                     os.path.pardir))
        self.maze_seed = maze_seed
        self.scale = cell_size
        self.cell_size = cell_size
        self.grid_size = grid_dims
        self.goalstate = []
        self.objects = self.validate_objects(objects)
        self.blocked_edges = blocked_edges # List of tuples of occupied edges
        self.book_info = book_info
        self.env = env
        self.env_objects = {"object":{}, "goal":{}}


        if grid_dims is None:
            raise ValueError("Grid dimensions undefined")

        self.init_setup = self.generate()

    def validate_objects(self, objects_dict):

        objects_dict.setdefault('obstacles',[])
        objects_dict.setdefault('bounding',[])
        objects_dict.setdefault('goal',[])

        return objects_dict

    def init_empty_world(self):
        #File containing empty world description
        f_in = open(os.path.join(self.root_dir,'worlds/empty_world.sdf'), 'r')
        #File to save your maze
        f_out = open(os.path.join(self.root_dir,'worlds/maze.sdf'), 'w')
        #Copying empty world description into output maze
        for line in f_in:
            f_out.write(line)
        f_in.close()
        return f_out

    def write_object_data(self):

        for key in self.env_objects:
            objtype = self.env_objects[key]
            self.env_objects[key] = {k:v for k,v in sorted(objtype.items())}

        with open(os.path.join(self.root_dir, 'objects.json'),'w') as ofp:
            json.dump(self.env_objects, ofp)

    def object_data_generator(self, obj_type, obj_idx, start_coords, goal_coords, subject):
        
        obj_size = ['small','large']
        if self.env == "bookWorld":
            if obj_type == 'object':
                obj = 'book_'
            else:
                obj = 'bin_'
        else:
            if obj_type == 'object':
                obj = subject.lower() + "_"
            else:
                obj = 'table_'

        data = {}
        data['load_loc'] = start_coords
        data['loc'] = goal_coords
        data['size'] = obj_size[obj_idx%2]
        data['obj_type'] = subject.lower()


        self.env_objects[obj_type][obj+str(obj_idx+1)] = data # +1 to match the object name in gazebo


    def get_goal_blocked_edges(self, coords):

        goal_edges = []
        goal_edges.append((coords[0]-self.scale,coords[1],coords[0],coords[1]))
        goal_edges.append((coords[0],coords[1]+self.scale,coords[0],coords[1]))
        goal_edges.append((coords[0],coords[1],coords[0]+self.scale,coords[1]))
        goal_edges.append((coords[0],coords[1],coords[0],coords[1]-self.scale))

        return goal_edges
    
    def get_obstacle_blocked_edges(self, coords):

        goal_edges = []
        goal_edges.append((coords[0],coords[1],coords[0]+self.scale,coords[1]))
        goal_edges.append((coords[0],coords[1],coords[0],coords[1]+self.scale))
        goal_edges.append((coords[0]+self.scale,coords[1],coords[0]+self.scale,coords[1]+self.scale))
        goal_edges.append((coords[0],coords[1]+self.scale,coords[0]+self.scale,coords[1]+self.scale))

        return goal_edges, [[coords[0],coords[1]],
                            [coords[0]+self.scale,coords[1]],
                            [coords[0],coords[1]+self.scale],
                            [coords[0]+self.scale,coords[1]+self.scale]]

    def add_blocked_edges(self, coords):
        blocked_list = []
        blocked_edges = []
        x = coords[0]
        y = coords[1]
        for i in range(2):
            for j in range(2):
                blocked_edges.append((x + i * self.cell_size, y + j * self.cell_size, x + i * self.cell_size + self.cell_size, y + j * self.cell_size))
                blocked_edges.append((x + i * self.cell_size, y + j * self.cell_size, x + i * self.cell_size, y + j * self.cell_size + self.cell_size))
                blocked_edges.append((x + i * self.cell_size, y + j * self.cell_size, x + i * self.cell_size, y + j * self.cell_size - self.cell_size))
                blocked_edges.append((x + i * self.cell_size - self.cell_size, y + j * self.cell_size, x + i * self.cell_size, y + j * self.cell_size))
        blocked_list = [(x - self.cell_size, y), (x, y - self.cell_size), (x - self.cell_size, y + self.cell_size), (x, y + self.cell_size * 2), (x + self.cell_size * 2, y + self.cell_size), (x + self.cell_size, y + self.cell_size * 2),
                        (x + self.cell_size * 2, y), (x + self.cell_size, y - self.cell_size)]
        return blocked_edges,blocked_list


    def generate(self):
        '''
        One time generation call to create the maze with 
        all the objects passed to the class
        '''
        self.f_out = self.init_empty_world()
        np.random.seed(self.maze_seed)
        self.xml_objects = {'bounding':[], 'goal':[], 'obstacles':[]}
        
        # Setting bounding type objects in the maze
        for bound_obj, config in self.objects['bounding'].items():
            bounding_dae = config.get('file')
            bounding_scale = self.__get_body_scale(config)
            for i in range(config['count']):
                # Only 1 object is expected to be set as the bounding type
                # If multiple must be set, use the mazeObjects subclass to define 
                # multiple of the same variety in a loop inside that class
                bound_instance = bound_obj(coords=(-0.1, 0.12, -0.1423),
                                 grid_dims=self.grid_size,
                                 scale=bounding_scale,
                                 out_xml=self.f_out,
                                 dae=bounding_dae)
                bound_instance.add_object()
                self.xml_objects['bounding'].append(bound_instance)

        # Setting obstacle type objects in the maze
        obstacle_names = []
        obstacle_ctr = 0
        for obstacle_obj, config in sorted(self.objects['obstacles'].items(),key=lambda x: x[0].__name__):
            obstacle_dae = config.get('file')
            obj_scale = self.__get_body_scale(config)
            obstacle_names.append(obstacle_obj.__name__)

            if self.env == "bookWorld":  # book sizes. Only 1 book object.
                iterator = range(config['count'] * config['subjects'] * 2)
                modop = config['count'] * 2
            else:  # portion sizes. Multiple food objects.
                iterator = range(config['count'] * 2)
                modop = 1

            for i in iterator:
                coords, edge = self.__sample_obstacle_location()
                while len(coords) == 0:
                    coords, edge = self.__sample_obstacle_location()

                if self.env == "bookWorld":
                    if i%(config['count']*2) < config['count']:
                        obstacle_scale = (0.4, 0.4, 0.4)
                    else:
                        obstacle_scale = (0.6, 0.6, 0.6)
                else:
                    if i%(config['count']*2) < config['count']:
                        obstacle_scale = (0.5, 0.5, 0.5)
                    else:
                        obstacle_scale = (0.8, 0.8, 0.8)
                obstacle_scale = tuple([x*y for x, y in zip(obj_scale, obstacle_scale)])

                if self.env == "bookWorld":
                    coords = [edge[0]+self.scale/2,edge[1]+self.scale/2,coords[2]]
                    obstacle_blocks,corners = self.get_obstacle_blocked_edges(edge[:2])
                    rospy.logdebug(obstacle_blocks)
                    self.blocked_edges.extend(obstacle_blocks)

                obstacle_instance = obstacle_obj(coords=coords,
                                                 grid_dims=self.grid_size,
                                                 scale=obstacle_scale,
                                                 out_xml=self.f_out,
                                                 dae=obstacle_dae,
                                                 id=obstacle_ctr+1,
                                                 color=obstacle_ctr // modop)

                obstacle_instance.add_object()
                self.xml_objects['obstacles'].append(obstacle_instance)

                if self.env == "bookWorld":                
                    obj_name = self.book_info['subject_list'][obstacle_ctr // modop]
                else:
                    obj_name = obstacle_instance.name

                rospy.logdebug("Obstacle: {} {} {} {}".format(obstacle_ctr, coords, edge, obj_name))

                self.object_data_generator(obj_type='object',
                                           obj_idx=obstacle_ctr,
                                           goal_coords=coords[:-1],
                                           start_coords=[edge[:2], edge[2:]],
                                           subject=obj_name)
                
                obstacle_ctr += 1

        # Generate a random goal location and set the goal objects in the maze
        goalctr = 0
        for goal_obj, config in self.objects['goal'].items():
            goal_dae = config.get('file')
            
            color_idx = -1
            for i in range(config['count']):
                if i % 2 == 0:
                    color_idx += 1
                
                # Even indexed tables are small, odd indexed tables are large
                goal_scale = (0.4, 0.4, 0.4)
                if i%2 == 0 and self.env == "bookWorld":
                    goal_scale = (0.3, 0.3, 0.3)

                goal_coords, goal_blocks, load_locations = self.__generate_goal_coords()
                rospy.logdebug("Goal: {} {} {}".format(goalctr, goal_coords, load_locations))
                goal_instance = goal_obj(coords=tuple(goal_coords),
                                         grid_dims=self.grid_size,
                                         scale=goal_scale,
                                         out_xml=self.f_out,
                                         dae=goal_dae,
                                         id=goalctr+1,
                                         color=self.book_info['color_list'][color_idx])

                goal_instance.add_object()
                self.xml_objects['goal'].append(goal_instance)
                self.blocked_edges.extend(goal_blocks)
                self.goalstate.append(tuple(goal_coords[:-1]))

                if self.env == "bookWorld":
                    obj_name = self.book_info['subject_list'][i // 2]
                else:
                    obj_name = obstacle_names[i//2]

                self.object_data_generator(obj_type='goal',
                                           obj_idx=i,
                                           goal_coords=goal_coords[:-1],
                                           start_coords=load_locations,
                                           subject=obj_name)
                goalctr += 1
        self.f_out.write('</state>')

        for goal in self.xml_objects['goal']:
            goal.add_object_description()
        for bound in self.xml_objects['bounding']:
            bound.add_object_description()
        for obstacle in self.xml_objects['obstacles']:
            obstacle.add_object_description()

        self.f_out.write('</world>\n</sdf>')
        self.f_out.close()
        print ("World generated")

        self.write_object_data()

        return self.blocked_edges

    def __get_body_scale(self, body_config, default=(1,1,1)):
        
        if body_config.get('scale') is not None:
            scale = tuple(body_config['scale'])
        else:
            scale = default
        
        return scale

    def __sample_obstacle_location(self):

        coords = tuple()
        edge = tuple()

        if self.env == "bookWorld":
            offset = np.random.uniform(0, 0.1*self.scale)
            flag = np.random.randint(0, 2)
            x = np.random.randint(0, self.grid_size[0]+1)*self.scale
            y = np.random.randint(0, self.grid_size[1]+1)*self.scale
            rounded_x = self.round(x)
            rounded_y = self.round(y)
            z = 0
            offset = 0.0
            
            if flag==0 \
            and (rounded_x+self.scale <= self.grid_size[0]*self.scale) \
            and ((rounded_x, rounded_y, rounded_x+self.scale, rounded_y) not in self.blocked_edges):
                coords = (x+self.scale/2+offset, y, z)
                edge = (rounded_x, rounded_y, rounded_x+self.scale, rounded_y)
            
            elif flag==1 \
            and (rounded_y+self.scale <= self.grid_size[1]*self.scale) \
            and ((rounded_x, rounded_y, rounded_x, rounded_y+self.scale) not in self.blocked_edges):
                coords = (x, y+self.scale/2-offset, z)
                edge = (rounded_x, rounded_y, rounded_x, rounded_y+self.scale)


        elif self.env == "cafeWorld":
            offset = np.random.uniform(0, 0.2*self.scale)
            locs = [
                (np.random.uniform(0.05, 0.1), np.random.uniform(6.3, 7.6)),
                (np.random.uniform(0.2, 1.3), np.random.uniform(7.7, 7.85))
            ]
            choice = np.random.randint(0,len(locs))
            x, y = locs[choice]
            rounded_x = 0.5
            rounded_y = 7.0
            z = 2.3 * 0.2
            
            if choice==0:
                coords = (x, y+offset, z)
                edge = (rounded_x, rounded_y, rounded_x+self.scale, rounded_y)
            
            elif choice==1:
                coords = (x+offset, y, z)
                edge = (rounded_x, rounded_y, rounded_x, rounded_y+self.scale)


        return coords, edge

    def round(self,num):
        return round(num*2)/2.0

    def __generate_goal_coords(self):

        if self.env == "bookWorld":
            grid_x = self.grid_size[0]*self.scale
            grid_y = self.grid_size[1]*self.scale
        elif self.env == "cafeWorld":
            grid_x = 3
            grid_y = 6

        goal_coords = [self.round(np.random.randint(1, grid_x+1)),
                       self.round(np.random.randint(1, grid_y+1)),
                       0]
        blocked_edges, load_locations = self.add_blocked_edges(goal_coords)

        while any([edge in self.blocked_edges for edge in blocked_edges]):

            goal_coords = [self.round(np.random.randint(1, grid_x+1)),
                           self.round(np.random.randint(1, grid_y+1)),
                           0]
            blocked_edges, load_locations = self.add_blocked_edges(goal_coords)

        return goal_coords, blocked_edges, load_locations

    def initState(self):
        return (0, 0, 'EAST')

    def getGoalState(self):
        '''
        Return the coords of the goal state in the maze
        '''
        # Define a default goal state in case no goal objects are
        # generated
        if self.goalstate == []:
            self.goalstate = [(self.grid_size[0]*self.scale, self.grid_size[1]*self.scale)]

        return self.goalstate

    def getMazeState(self):
        '''
        Return blocked edges 
        '''
        self.grid_size = [self.grid_size[0]*self.scale, self.grid_size[1]*self.scale]

        return self.grid_size, self.scale, self.blocked_edges
    
    def delete_edge(self, obj_name):
        locs = self.env_objects['object'][obj_name]['load_loc']
        # Since the loading edge starts with the (x,y) of the object
        # find all 4 edges surrouding it relative to x,y and remove them
        if obj_name.startswith("book_"):
            edge,_ = self.get_obstacle_blocked_edges(locs[0])
        else:
            edge = [tuple(locs[0]+locs[1])]
        self.update_blocked_edges(edge)
        return "1"

    def update_blocked_edges(self, edges):
        for edge in edges:
            if edge in self.blocked_edges:
                rospy.logdebug("Removing blocked edge: {}".format(edge))
                self.blocked_edges.remove(edge)
