#!/usr/bin/env python3
# encoding: utf-8

__copyright__ = "Copyright 2019, AAIR Lab, ASU"
__authors__ = ["Naman Shah"]
__credits__ = ["Siddharth Srivastava"]
__license__ = "MIT"
__version__ = "1.0"
__maintainers__ = ["Naman Shah", "Pulkit Verma"]
__contact__ = "aair.lab@asu.edu"
__docformat__ = 'reStructuredText'

import rospy
from hw3.srv import *
import json

class State:
    """
    This class defines the state of the TurtleBot.

    """
    
    def __init__(self,x,y,orientation):
        """
        :param x: current x-cordinate of turtlebot
        :type x: float
        :param y: current x-cordinate of turtlebot
        :type y: float
        :param orientation: current orientation of turtlebot, can be either NORTH, SOUTH, EAST, WEST
        :type orientation: str

        """    
        self.x  = x 
        self.y = y
        self.orientation = orientation

    def __eq__(self, other):
        if self.x == other.x and self.y == other.y and self.orientation == other.orientation:
            return True
        else:
            return False
            
    def __hash__(self):
    
        return hash(str(self))

    def __repr__(self):
        return "({}, {}, {})".format(str(self.x), str(self.y), str(self.orientation))
        
    def get_gazebo_repr(self):
        return [self.x, self.y, self.orientation]

    def get_x(self):
        """
            Returns
            ========
                int
                    The x-value of the state (-1 if the state is invalid).
        """
        
        return self.x
        
    def get_y(self):
        """
            Returns
            ========
                int
                    The y-value of the state (-1 if the state is invalid).
        """
        
        return self.y
        
    def get_orientation(self):
        """
            Returns
            ========
                str
                    The direction (orientation) of the state.
        """
        
        return self.orientation

class Helper:
    """
    A state here is represented as a dictionary defining the location of books, if they have been placed, 
    location of turtlebot and which book is present in the basket.

    Example:
        \{
            "robot":  { "x" : 0, "y" : 0, "orientation" : "EAST" },

            "basket": None,
            
            "book_1":  { "x" : 2, "y" : 1.5, "placed" : False },
            
            "book_2": ..
            
            .
            
            .
            
            "trolly_1": { "x" : 3, "y" : 3 },

        \}

    """        

    def get_current_state (self):
        """
        This function calls get_initial_state service to recive the initial state of the turtlebot.
        
        :returns: State Dictionary
        :rtype: dict

        :raises: ServiceException: When call to rospy fails.

        """
        rospy.wait_for_service('get_current_state')
        try:
            get_initial_state = rospy.ServiceProxy('get_current_state', GetInitialState)
            response = get_initial_state()
            return json.loads(response.state)
        except rospy.ServiceException as e:
            print ("Service call failed: %s" % e)


    def is_terminal_state(self, state):
        """
        This function accepts a dictionary representing the state.

        :param state: State represented as a dictionary.
        :type state: dictionary

        :returns: True if state is terminal state and returns false if state is not terminal state
        :rtype: Boolean

        :raises: ServiceException: When call to rospy fails.

        """

        rospy.wait_for_service('is_terminal_state')
        try:
            is_term_state = rospy.ServiceProxy('is_terminal_state', IsTerminalState)
            response = is_term_state(json.dumps(state))
            return response.value == 1
        except rospy.ServiceException as e:
            print ("Service call failed: %s" % e)


    def reset_world (self):
        """
        This function resets the running server state to the initial state

        :rtype: None

        :raises: ServiceException: When call to rospy fails.
        
        .. warning::
            This method works only in headless mode. It resets the state of the running server to the initial state. Use this at the start of each episode.

        """
        rospy.wait_for_service('reset_world')
        try:
            handle = rospy.ServiceProxy('reset_world', ResetWorldMsg)
            response = handle()
        except rospy.ServiceException as e:
            print ("Service call failed: %s" % e)


    def get_all_actions(self):
        """
        Returns all the actions that turtleBot can perform.
        
        :returns: A comma separated list of actions turtlebot can perform
        :rtype: list of strings

        :raises: ServiceException: When call to rospy fails.
        
        """
        rospy.wait_for_service('get_all_actions')
        try:
            all_actions = rospy.ServiceProxy('get_all_actions', GetActions)
            response = all_actions()
            return response.actions.split(",")
        except rospy.ServiceException as e:
            print ("Service call failed: %s" % e)


    def get_reward(self, state, action, next_state):
        """
        
        This function which reward for executing action in the given state and resulting in the next_state.
        This is equivalent to R(s,a,s') formulation.

        :param state: state where the action "action" is taken
        :type state: dictonary
        :param action: "action" being taken in state 
        :type action: str
        :param next_state: resulting state when action "action" performed in state "state"
        :type next_state: dictionary

        :returns: The reward for executing action given s and s' as usually expressed in R(s,a,s')
        :rtype: float

        :raises: ServiceException: When call to rospy fails.

        """
        rospy.wait_for_service('get_reward')
        try:
            get_reward = rospy.ServiceProxy('get_reward', GetReward)
            state = json.dumps(state)
            next_state = json.dumps(next_state)
            response = get_reward(state, action, next_state)
            return response.reward
        except rospy.ServiceException as e:
            print ("Service call failed: %s" % e)


    def execute_action(self, action, action_params):
        """
        This function executes the given action in the state being maintained by the server.

        :param action: an str representing the action to perform. This action is a valid defined in the action_config.json
        :type action: str
        :param action_params: a dictionary representing the parameters associated with the action as defined in the action_config.json
        :type action_params: dict

        Example:
            `action='careful_pick', action_params={'book_name':'book_1'}`

        :returns: True if action was successful and returns false otherwise
        :returns: Next state  represented as dictionary
        :rtype: Boolean, dictionary

        :raises: ServiceException: When call to rospy fails.

        .. note::
            action is a value defined in the action_config.json
            
            action_params is a dictionary specifying the parameters associated with the action as defined in
            
            action_config.json

        """
        
        rospy.wait_for_service('execute_action')
        try:
            execute_action = rospy.ServiceProxy('execute_action', ActionMsg)

            action_params = json.dumps(action_params)
            response = execute_action(action, action_params)

            return_val = True
            if response.success == -1:
                return_val = False
            return return_val, json.loads(response.next_state)
        except rospy.ServiceException as e:
            print ("Service call failed: %s" % e)

