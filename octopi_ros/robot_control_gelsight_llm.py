#!/usr/bin/env python

'''
this is used specially to receive object commands based on the LLM input and move them
'''

import math
import rospy
import numpy as np
from std_msgs.msg import String, Float64
from geometry_msgs.msg import Pose, Twist, Transform, PoseArray
import ast
from franka_control_wrappers.panda_commander import PandaCommander
from dougsm_helpers.ros_control import ControlSwitcher
from franka_msgs.msg import FrankaState, Errors as FrankaErrors
import dougsm_helpers.tf_helpers as tfh
import time
from moveit_msgs.msg import RobotState, Constraints, OrientationConstraint, PositionConstraint
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import rosbag
import tf
from scipy.spatial.transform import Rotation as R
from sympy import Matrix
import os, shutil
import random

GOAL_TOLERANCE = 0.001
CONSTRAINT_TOLERANCE = 0.001
PLANNING_TIMEOUT=5

##TODO Update these start/goal poses as needed. These are the pre-computed joint state positions of the robot.
stored_poses = {
    'home' : [-0.002706399888474321, -0.47568586444018174, 0.03327264372425887, -2.3363230046765158, -0.007343809096193361, 1.907500267314152, 0.02785356081525068],
    'start': [
        [-0.026379698702117853, -0.44520214150244725, 0.25876896292703194, -2.723473448636241, 0.1096197988126013, 2.309840831173791, 0.09283879206614108],
        [-0.019110939854585274, -0.4175276752563945, 0.02848687166942839, -2.730343586303879, -0.07731417495115679, 2.314374106140262, 0.023807761389103333],
        [-0.01931760371832588, -0.3454632747382448, -0.2301061128321351, -2.6320139571003676, -0.14891644139257668, 2.2646462879710727, -0.19045249198564806]    
    ], 
    'goal': [
        [-0.04212008604250456, -0.009489816104342927, 0.1991638339324248, -2.1879149586999347, 0.012982576729118697, 2.2099936048299686, 0.14137550114079395],
        [-0.04214260097135247, -0.0009674228618346287, 0.06855321409894512, -2.177193069262069, -0.05062089080499723, 2.189568569989852, 0.04991614641171444],
        [-0.042152583257520235, -0.0231625909066629, -0.05723162048148942, -2.1385655101126058, -0.11946608895059804, 2.13208334546619, -0.053430909948266506]    
    ],
    'camera_pose': [-0.004120275633292022, -0.72693497881973, 0.05129479751984278, -2.1822514616793613, -0.009388850129312938, 1.4259686019155715, 0.03807281148832616]
}

class MoveRobot(object):
    def __init__(self):
        super(MoveRobot, self).__init__()
        rospy.init_node('robot_control', anonymous=False)
        self.gripper = 'robotiq'

        rospy.Subscriber('/gsmini_command', String, queue_size=1, callback=self.targetCallback)

        self.filename_pub = rospy.Publisher('/gsmini_command', String, queue_size=1)

        # cartesian control
        self.curr_velocity_publish_rate = 100.0  # Hz
        self.curr_velo_pub = rospy.Publisher("/cartesian_velocity_node_controller/cartesian_velocity", Twist, queue_size=1)
        self.max_velo = 0.10
        self.curr_velo = Twist()
        self.image_number = 0

        self.trial = 0
        self.last_obj = ""

        self.cs = ControlSwitcher(
            {
                "moveit": "position_joint_trajectory_controller",
                "velocity": "cartesian_velocity_node_controller",
            }
        )
        self.cs.switch_controller("moveit")
        self.ee = 'panda_link8'
        self.pc = PandaCommander(group_name="panda_arm", gripper=self.gripper, ee=self.ee)
        self.pc.create_scene()

        self.robot_state = None
        self.ROBOT_ERROR_DETECTED = False
        self.BAD_UPDATE = False
        rospy.Subscriber(
            "/franka_state_controller/franka_states",
            FrankaState,
            self.__robot_state_callback,
            queue_size=1,
        )

        self.O_T_EE = None
        self.target_pose = None
        self.ready_camera = False
        self.current_type = "default"
        self.tf_listener = tf.TransformListener()
        self.pc.print_debug_info()
        print('MOVE ROBOT is running')


        self.constraints = None
    
    def __robot_state_callback(self, msg):
        self.robot_state = msg
        self.O_T_EE = np.array(self.robot_state.O_T_EE).reshape([4,4]).T
        if any(self.robot_state.cartesian_collision):
            if not self.ROBOT_ERROR_DETECTED:
                rospy.logerr("Detected Cartesian Collision")
            self.ROBOT_ERROR_DETECTED = True
        for s in FrankaErrors.__slots__:
            if getattr(msg.current_errors, s):
                self.stop()
                if not self.ROBOT_ERROR_DETECTED:
                    rospy.logerr("Robot Error Detected")
                self.ROBOT_ERROR_DETECTED = True

    def touch_items(self, touch_index=None):
        success = self.pc.goto_joints(stored_poses['home'], velocity=0.1)
        start = 1
        if not success:
            print("could not go to home")
            return
        for pose in stored_poses['start']:
            success = self.pc.goto_joints(pose, velocity=0.1)
            if not success:
                print("could not go to designated start pose")
                return
            msg = String()
            msg.data = 'r {}'.format(start)
            self.filename_pub.publish(msg)
            rospy.sleep(2)
            msg.data = 'c'
            self.filename_pub.publish(msg)
            start += 1
            rospy.sleep(5)
            success = self.pc.goto_joints(stored_poses['home'], velocity=0.1)

        self.take_picture()

    def take_picture(self):
        success = self.pc.goto_joints(stored_poses['camera_pose'], velocity=0.1)
        m = String()
        m.data = "take pic"
        self.filename_pub.publish(m)
        rospy.sleep(3)
        success = self.pc.goto_joints(stored_poses['home'], velocity=0.1)

    def infer_singular(self):
        msg = String()
        msg.data = 'r 4'
        self.filename_pub.publish(msg)
        rospy.sleep(2)
        msg.data = 'c'
        self.filename_pub.publish(msg)

    def move_item(self, start, goal):
        ## moves an item from the start position (left/right/middle) to the goal position.
        ## 3 calls of this is used to perform the sorting.

        success = self.pc.goto_joints(stored_poses['home'], velocity=0.1)
        success = self.pc.goto_joints(stored_poses['start'][start], velocity=0.1)
        if not success:
            print("could not go to designated start pose")
            return
        rospy.sleep(1)
        self.pc.gripper.grasp(0, force=90)
        success = self.pc.goto_joints(stored_poses['home'], velocity=0.1)
        rospy.sleep(1)
        success = self.pc.goto_joints(stored_poses['goal'][goal], velocity=0.1)
        if not success:
            print("could not go to designated start pose")
            return
        rospy.sleep(1)
        self.pc.gripper.set_gripper(0.1)
        success = self.pc.goto_joints(stored_poses['home'], velocity=0.1)

    def targetCallback(self, msg):
        
        print('{} is receieved.'.format(msg.data))
        if msg.data == 'item touch':
            self.touch_items()
        elif msg.data == 'infer':
            self.infer_singular()
        elif msg.data == 'take_pic':
            self.take_picture()
        elif 'movesort' in msg.data:
            _, start, stop = msg.data.split(' ')
            self.move_item(int(start) - 1, int(stop))
        elif msg.data[0] == 'c':
            self.collect_data(msg.data)
        elif msg.data[0] == 'r' and 'reset' not in msg.data:
            self.trial = 0
            self.last_obj = msg.data.split(' ')[1]
        elif 'trace' in msg.data:
            self.trace()

    def make_transformation_matrix(self, trans, rot):
        pos_x, pos_y, pos_z = trans
        or_x, or_y, or_z, or_w = rot
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = R.from_quat([or_x, or_y, or_z, or_w]).as_matrix()
        transformation_matrix[:3, 3] = [pos_x, pos_y, pos_z]
        return transformation_matrix
    

    def TF_matrix(self, i,dh):
        # Define Transformation matrix based on DH params
        alpha = dh[i][0]
        a = dh[i][1]
        d = dh[i][2]
        q = dh[i][3]
        
        TF = Matrix([[math.cos(q),-math.sin(q), 0, a],
                    [math.sin(q)*math.cos(alpha), math.cos(q)*math.cos(alpha), -math.sin(alpha), -math.sin(alpha)*d],
                    [math.sin(q)*math.sin(alpha), math.cos(q)*math.sin(alpha),  math.cos(alpha),  math.cos(alpha)*d],
                    [   0,  0,  0,  1]])
        return TF
    

    def compute_fk(self, joint_angles):
        M_PI = math.pi

        dh = [[ 0,      0,        0.333,   joint_angles[0]],
                [-M_PI/2,   0,        0,       joint_angles[1]],
                [ M_PI/2,   0,        0.316,   joint_angles[2]],
                [ M_PI/2,   0.0825,   0,       joint_angles[3]],
                [-M_PI/2,  -0.0825,   0.384,   joint_angles[4]],
                [ M_PI/2,   0,        0,       joint_angles[5]],
                [ M_PI/2,   0.088,    0.107,   joint_angles[6]]]

        T_01 = self.TF_matrix(0, dh)
        T_12 = self.TF_matrix(1, dh)
        T_23 = self.TF_matrix(2, dh)
        T_34 = self.TF_matrix(3, dh)
        T_45 = self.TF_matrix(4, dh)
        T_56 = self.TF_matrix(5, dh)
        T_67 = self.TF_matrix(6, dh)

        T_07 = T_01*T_12*T_23*T_34*T_45*T_56*T_67 
        
        translations = T_07.col(-1)
        translations.row_del(-1)
        translations = list(translations.T)
        T_07.row_del(-1)
        T_07.col_del(-1)

        #quaternions = R.from_matrix(T_07).as_quat()
        quaternions = [0.993, 0.095, -0.005, -0.067]

        return translations, quaternions

    def convert_gripper_val(self, x):
        return 0.135*( (255.0 - x) / 255.0 )
    

    def move_along_z(self, n=3, velocity=0.1, z=0.1):
        pose1 = self.pc.get_current_pose()

        self.init_path_constraints(pose1)
        self.enable_path_constraints()

        # note poses must have same orientation
        x1,y1,z1 = pose1.position.x, pose1.position.y, pose1.position.z
        z2 = z + z1

        waypoints = []
        for i in range(1, n+1, 1):
            new_pose = Pose()
            new_pose.orientation.x = pose1.orientation.x
            new_pose.orientation.y = pose1.orientation.y
            new_pose.orientation.z = pose1.orientation.z
            new_pose.orientation.w = pose1.orientation.w
            new_pose.position.x = x1
            new_pose.position.y = y1
            new_pose.position.z = z1 + (z2-z1)*i/n
            waypoints.append(new_pose)

        success = self.pc.goto_pose_cartesian_waypoints(poses=waypoints, velocity=velocity)

        

        self.disable_path_constraints()

        return success
    
    def move_along_z_new(self, distance=0.1, velocity_coeff=0.01):
        '''
        
        This moves along z by time by time (1.5 seconds)

        '''

        start_time =  time.time()
        

        self.cs.switch_controller('velocity')


        curr_pose = self.pc.get_current_pose()
        z_curr = curr_pose.position.z
        z_target = curr_pose.position.z + distance

        if np.sign(distance) == 1:
            new_distance = z_curr - z_target
        else:
            new_distance = z_target - z_curr
        

        v = Twist()
        v.linear.z = velocity_coeff*np.sign(distance)

        # cartesian_contact = any(self.robot_state.cartesian_contact)
        # robot_error = self.ROBOT_ERROR_DETECTED

        
        start_time = time.time()
        while (
            new_distance  <= 0
            and not any(self.robot_state.cartesian_contact)
            and not self.ROBOT_ERROR_DETECTED
        ):
            self.curr_velo_pub.publish(v)
            z_curr = self.pc.get_current_pose().position.z

            if np.sign(distance) == 1:
                new_distance = z_curr - z_target
            else:
                new_distance = z_target - z_curr
            rospy.sleep(0.0001)
        
        v = Twist()
        self.curr_velo_pub.publish(v)
        rospy.sleep(0.5)
        if self.ROBOT_ERROR_DETECTED:
            print('Robot Error Detected')
            self.__recover_robot_from_error()
            self.move_res.publish('error')
            return  time.time()-start_time, False
        

        z_curr = self.pc.get_current_pose().position.z
        if np.sign(distance) == 1:
            new_distance = z_curr - z_target
        else:
            new_distance = z_target - z_curr

        print('Cartesian path controller ended with {} delta.'.format(new_distance))

        self.cs.switch_controller('moveit')

        return time.time()-start_time, True

    def collect_data(self, name):
        ### instructs the robot to close the gripper for 4 seconds and opens it.
        self.pc.gripper.set_gripper(0)
        rospy.sleep(4)
        self.pc.gripper.set_gripper(0.1)
        self.trial += 1

    def init_path_constraints(self,pose):

        self.constraints = Constraints()
        self.constraints.name = "pre_grasp_to_grasp"
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = "panda_link0"
        # orientation_constraint.header = pose.header
        orientation_constraint.link_name = self.ee
        orientation_constraint.orientation = pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = CONSTRAINT_TOLERANCE
        orientation_constraint.absolute_y_axis_tolerance = CONSTRAINT_TOLERANCE
        orientation_constraint.absolute_z_axis_tolerance = CONSTRAINT_TOLERANCE
        orientation_constraint.weight = 1

        self.constraints.orientation_constraints.append(orientation_constraint)

        return
    
    def enable_path_constraints(self):
        self.pc.active_group.set_path_constraints(self.constraints)
        return

    def disable_path_constraints(self):
        self.pc.active_group.set_path_constraints(None)
        return

    def get_distance(self, current_pose):

        a = np.array([self.target_pose.position.x, self.target_pose.position.y, self.target_pose.position.z])
        b = np.array([current_pose.position.x, current_pose.position.y, current_pose.position.z])
        
        return np.linalg.norm(a-b)

    def move_to_home(self):
        self.pc.goto_home(velocity=0.1)

    def __recover_robot_from_error(self):
        rospy.logerr("Recovering")
        self.pc.recover()
        self.cs.switch_controller("moveit")
        self.move_to_home()
        # self.pc.goto_home()
        # self.pc.goto_saved_pose("home", velocity=0.1)
        rospy.logerr("Done")
        self.ROBOT_ERROR_DETECTED = False

def main():
    m_g = MoveRobot()
    rospy.spin()

if __name__ == '__main__':
    main()
