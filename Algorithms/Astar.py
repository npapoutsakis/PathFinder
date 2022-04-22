import copy
import time
import sys
from abc import ABC
from typing import Tuple, Union, Dict, List, Any
import math
import numpy as np

from commonroad.scenario.trajectory import State

sys.path.append('../')
from SMP.maneuver_automaton.motion_primitive import MotionPrimitive
from SMP.motion_planner.node import Node, CostNode
from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.queue import FIFOQueue, LIFOQueue, PriorityQueue, Queue
from SMP.motion_planner.search_algorithms.base_class import SearchBaseClass
from SMP.motion_planner.utility import MotionPrimitiveStatus, initial_visualization, update_visualization

class SequentialSearch(SearchBaseClass, ABC):
    """
    Abstract class for search motion planners.
    """

    # declaration of class variables
    path_fig: Union[str, None]

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)

    def initialize_search(self, time_pause, cost=True):
        """
        initializes the visualizer
        returns the initial node
        """
        self.list_status_nodes = []
        self.dict_node_status: Dict[int, Tuple] = {}
        self.time_pause = time_pause
        self.visited_nodes = []

        # first node
        if cost:
            node_initial = CostNode(list_paths=[[self.state_initial]],
                                        list_primitives=[self.motion_primitive_initial],
                                        depth_tree=0, cost=0)
        else:
            node_initial = Node(list_paths=[[self.state_initial]],
                                list_primitives=[self.motion_primitive_initial],
                                depth_tree=0)
        initial_visualization(self.scenario, self.state_initial, self.shape_ego, self.planningProblem,
                              self.config_plot, self.path_fig)
        self.dict_node_status = update_visualization(primitive=node_initial.list_paths[-1],
                                                     status=MotionPrimitiveStatus.IN_FRONTIER,
                                                     dict_node_status=self.dict_node_status, path_fig=self.path_fig,
                                                     config=self.config_plot,
                                                     count=len(self.list_status_nodes), time_pause=self.time_pause)
        self.list_status_nodes.append(copy.copy(self.dict_node_status))
        return node_initial

    def take_step(self, successor, node_current, cost=True):
        """
        Visualizes the step of a successor and checks if it collides with either an obstacle or a boundary
        cost is equal to the cost function up until this node
        Returns collision boolean and the child node if it does not collide
        """
        # translate and rotate motion primitive to current position
        list_primitives_current = copy.copy(node_current.list_primitives)
        path_translated = self.translate_primitive_to_current_state(successor,
                                                                    node_current.list_paths[-1])
        list_primitives_current.append(successor)
        self.path_new = node_current.list_paths + [[node_current.list_paths[-1][-1]] + path_translated]
        if cost:
            child = CostNode(list_paths=self.path_new,
                                 list_primitives=list_primitives_current,
                                 depth_tree=node_current.depth_tree + 1,
                                 cost=self.cost_function(node_current))
        else:
            child = Node(list_paths=self.path_new, list_primitives=list_primitives_current,
                         depth_tree=node_current.depth_tree + 1)

        # check for collision, skip if is not collision-free
        if not self.is_collision_free(path_translated):

            position = self.path_new[-1][-1].position.tolist()
            self.list_status_nodes, self.dict_node_status, self.visited_nodes = self.plot_colliding_primitives(current_node=node_current,
                                                                                           path_translated=path_translated,
                                                                                           node_status=self.dict_node_status,
                                                                                           list_states_nodes=self.list_status_nodes,
                                                                                           time_pause=self.time_pause,
                                                                                           visited_nodes=self.visited_nodes)
            return True, child
        self.update_visuals()
        return False, child

    def update_visuals(self):
        """
        Visualizes a step on plot
        """
        position = self.path_new[-1][-1].position.tolist()
        if position not in self.visited_nodes:
            self.dict_node_status = update_visualization(primitive=self.path_new[-1],
                                                         status=MotionPrimitiveStatus.IN_FRONTIER,
                                                         dict_node_status=self.dict_node_status, path_fig=self.path_fig,
                                                         config=self.config_plot,
                                                         count=len(self.list_status_nodes), time_pause=self.time_pause)
            self.list_status_nodes.append(copy.copy(self.dict_node_status))
        self.visited_nodes.append(position)

    def goal_reached(self, successor, node_current):
        """
        Checks if the goal is reached.
        Returns True/False if goal is reached
        """
        path_translated = self.translate_primitive_to_current_state(successor,
                                                                    node_current.list_paths[-1])
        # goal test
        if self.reached_goal(path_translated):
            # goal reached
            self.path_new = node_current.list_paths + [[node_current.list_paths[-1][-1]] + path_translated]
            path_solution = self.remove_states_behind_goal(self.path_new)
            self.list_status_nodes = self.plot_solution(path_solution=path_solution, node_status=self.dict_node_status,
                                                        list_states_nodes=self.list_status_nodes, time_pause=self.time_pause)
            return True
        return False

    def get_obstacles_information(self):
        """
        Information regarding the obstacles.
        Returns a list of obstacles' information, each element
        contains information regarding an obstacle:
        [x_center_position, y_center_position, length, width]

        """
        return self.extract_collision_obstacles_information()

    def get_goal_information(self):
        """
        Information regarding the goal.
        Returns a list of the goal's information
        with the following form:
        [x_center_position, y_center_position, length, width]
        """
        return self.extract_goal_information()

    def get_node_information(self, node_current):
        """
        Information regarding the input node_current.
        Returns a list of the node's information
        with the following form:
        [x_center_position, y_center_position]
        """
        return node_current.get_position()

    def get_node_path(self, node_current):
        """
        Information regarding the input node_current.
        Returns the path starting from the initial node and ending at node_current.
        """
        return node_current.get_path()

    def cost_function(self, node_current):
        """
        Returns g(n) from initial to current node, !only works with cost nodes!
        """
        velocity = node_current.list_paths[-1][-1].velocity

        node_center = self.get_node_information(node_current)
        goal_center = self.get_goal_information()
        distance_x = goal_center[0] - node_center[0]
        distance_y = goal_center[1] - node_center[1]
        length_goal = goal_center[2]
        width_goal = goal_center[3]

        distance = 4.5
        if(abs(distance_x)<length_goal/2 and abs(distance_y)<width_goal/2):
            prev_x = node_current.list_paths[-2][-1].position[0]
            prev_y = node_current.list_paths[-2][-1].position[1]
            distance = goal_center[0] - length_goal / 2 - prev_x
        cost = node_current.cost + distance
        
        return cost

    def heuristic_function(self, node_current):
        """
        Enter your heuristic function h(x) calculation of distance from node_current to goal
        Returns the distance normalized to be comparable with cost function measurements
        """

        #Euclidean Distance
        node_center = self.get_node_information(node_current)
        goal_node = self.get_goal_information()
        
        distance_x = abs(node_center[0] - goal_node[0])
        distance_y = abs(node_center[1] - goal_node[1])

        distance = math.sqrt((distance_x**2) + (distance_y**2))
        
        return distance

    def evaluation_function(self, node_current, w):
        """
        f(x) = g(x) + w * h(x)
        """
        g = self.cost_function(node_current)
        h = self.heuristic_function(node_current)
        f = g + w*h
        return f

    def convert_node_path(self, path):
        list = []
        for item in path:
            list.append(item.tolist())
        
        f = open("output.txt", "a")
        f.write("\tPath: ")
        for node in list:
            f.write("(" + str(node[0]) + ", " + str(node[1]) + ")") 
            f.write("->")
        f.write("\n")
        f.close()
        return

    def a_star(self, node_start, weight):
        
        #Not Visited
        open_list = PriorityQueue()
        
        #Visited
        closed_list = []
        
        #We insert the start node in the open_list
        open_list.insert(node_start, self.evaluation_function(node_start, weight))

        while not open_list.empty():
            
            #Node with the Lowest Eval Function
            node_current = open_list.pop()

            #For each successor on the lowest_eval_func_node
            for successor in node_current.get_successors():
            
                if self.goal_reached(successor, node_current):
                    f = open("output.txt", "a")
                    f.write("A* (w = "+str(weight)+"):\n")
                    f.write("\tVisited Nodes Number: " + str(len(closed_list))+ "\n")
                    f.close()
                    self.convert_node_path(self.get_node_path(node_current))
                    f = open("output.txt", "a")
                    # f.write("Path: " + str(self.convert_node_path(self.get_node_path(node_current)))+ "\n")
                    f.write("\tHeuristic Cost: " + str(self.heuristic_function(node_current)) + "\n")
                    f.write("\tEstimated Cost: "+ str(self.cost_function(node_current)) + "\n")
                    f.close()
                    
                    return True

                collision_flag, child = self.take_step(successor, node_current)

                if collision_flag:
                    #Means that we have checked a node and it collided with an obstacle, so nodes_visited++
                    closed_list.append(child)
                    continue

                #If child is already in close_list, just ignore it
                if child in closed_list:
                    continue

                #Insert the child with in open_list -> pop() will get the lowest again
                open_list.insert(child, self.evaluation_function(child, weight)) 
            
            #We have visited the node -> move it to close list
            closed_list.append(node_current)
       
        print("Search Failed!")
        return False

    def execute_search(self, time_pause, weight) -> Tuple[Union[None, List[List[State]]], Union[None, List[MotionPrimitive]], Any]:
        node_initial = self.initialize_search(time_pause=time_pause)

        path = self.a_star(node_start= node_initial, weight= weight)

        return path


class Astar(SequentialSearch):
    """
    Class for Astar Search algorithm.
    """
    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)
 
