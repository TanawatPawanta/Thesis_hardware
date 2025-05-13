#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from cascade_controller_interfaces.msg import MotorControl, MotorControlGroup, SetPoint
from std_msgs.msg import Header, Int16
from sensor_msgs.msg import JointState
from pydrake.trajectories import PiecewisePolynomial
import math
import numpy as np
from enum import Enum


class WalkStates(Enum):
    stand = "stand"
    walk = "walk"
    init = "init"
    R_stance = "R stance"
    L_stance = "L stance"

class MotorControlGroupPublisher(Node):
    def __init__(self):
        super().__init__('motor_control_group_publisher')
        self.get_logger().info("Motor Control Group Publisher started")
        self.timer_period = 0.02  
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.start_time = self.get_clock().now().nanoseconds * 1e-9
        self.LoadDatas(file_name="step10cmV4_2")
        self.DefineVariables()

        self.publisher_ = self.create_publisher(MotorControlGroup,
                                                'mit_mode_control', 10)

        self.walk_command_sub = self.create_subscription(
            Int16,
            '/walk_command',
            self.walk_command_callback,
            10
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

    def LoadDatas(self, file_name):
        self.loaded_q_data = np.load(
            f'/home/tanawatp/dynamixel_ws/src/cascade_controller/include/q_data_{file_name}.npz') 
        self.loaded_qd_data = np.load(
            f'/home/tanawatp/dynamixel_ws/src/cascade_controller/include/qd_data_{file_name}.npz')
        self.loaded_init_data = np.load(
            f'/home/tanawatp/dynamixel_ws/src/cascade_controller/include/q_init_data_{file_name}.npz')
        self.q_init_L = self.loaded_init_data['arr_0']
        self.q_init_R = self.loaded_init_data['arr_1']
        self.qd_init_L = self.loaded_init_data['arr_2']
        self.qd_init_R = self.loaded_init_data['arr_3']

        self.sw_leg_q = self.loaded_q_data['arr_0'].T
        self.st_leg_q = self.loaded_q_data['arr_1'].T
        self.sw_leg_qd = self.loaded_qd_data['arr_0'].T
        self.st_leg_qd = self.loaded_qd_data['arr_1'].T

        self.st_leg_q[:,-1] = self.st_leg_q[:,-1] + (self.st_leg_q[:,-1] - self.sw_leg_q[:,0])*0.5
        self.sw_leg_q[:,-1] = self.sw_leg_q[:,-1] + (self.sw_leg_q[:,-1] - self.st_leg_q[:,0])*0.5

    def DefineVariables(self):
        self.num_motors = 12
        self.current_position   = [0.0] * self.num_motors
        self.current_velocity   = [0.0] * self.num_motors
        self.current_effort     = [0.0] * self.num_motors

        self.statics_frictions_ff = [0.14,0.14,0.14,0.14,0.14,0.2,  # L
                                     0.14,0.14,0.14,0.14,0.14,0.2]  # R
        self.ind = 0
        self.L_leg_q  = self.q_init_L[:,self.ind]
        self.R_leg_q  = self.q_init_R[:,self.ind]
        self.L_leg_qd = self.qd_init_L[:,self.ind]
        self.R_leg_qd = self.qd_init_R[:,self.ind]
        self.L_stance = False 
        self.walk_state = WalkStates.init
        self.num_joints = 12 
        self.walk_command = -1
        self.controller_enable = True
        self.operation_state = "disable_controller"
        self.home_state = "init_trajectories"
        self.home_interval = 3.0 # [sec]

    def joint_state_callback(self, msg: JointState):
        for i, joint_name in enumerate(msg.name):
            if i < self.num_motors:
                self.current_position[i] = msg.position[i]
                self.current_velocity[i] = msg.velocity[i]
                self.current_effort[i] = msg.effort[i]

    def timer_callback(self):
        current_time = self.get_clock().now().nanoseconds * 1e-9
        msg = MotorControlGroup()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "" 
        print("operation state : ",self.operation_state)
        print("walk_command    : ",self.walk_command)
        print("-"*50)
        match self.operation_state:
            case "disable_controller":
                self.controller_enable = False
                if self.walk_command == -1:
                    self.operation_state = "disable_controller"
                elif self.walk_command == 0:
                    self.operation_state = "home"
                    self.home_state = "init_trajectories"
                elif self.walk_command == 1:
                    self.operation_state = "task_execution"
                elif self.walk_command == 2:
                    self.operation_state = "hold_position"
                    
            case "hold_position":
                self.controller_enable = True
                self.L_leg_q  = np.array(self.current_position)[0:6]
                self.R_leg_q  = np.array(self.current_position)[6:12]
                self.L_leg_qd = np.zeros(6)
                self.R_leg_qd = np.zeros(6)
                if self.walk_command == -1:
                    self.operation_state = "disable_controller"
                elif self.walk_command == 0:
                    self.operation_state = "home"
                    self.home_state = "init_trajectories"
                elif self.walk_command == 1:
                    self.operation_state = "task_execution"
                elif self.walk_command == 2:
                    self.operation_state = "hold_position"

            case "home":
                match self.home_state:
                    case "init_trajectories":
                        self.controller_enable = False
                        self.ind = 0
                        self.trajectory_time = 0.0
                        q_L_f  = self.q_init_L[:,self.ind]
                        q_R_f   = self.q_init_R[:,self.ind]
                        qd_L_f  = self.qd_init_L[:,self.ind]
                        qd_R_f  = self.qd_init_R[:,self.ind]
                        q_0 = np.array(self.current_position)
                        P_L = np.array([q_0[0:6],
                                        q_L_f]).T
                        V_L = np.array([np.zeros(6),
                                        np.zeros(6),]).T
                        self.traj_L = InitCubicTrajectories(P=P_L,
                                                       V=V_L,
                                                       Times=[0,self.home_interval],
                                                       n=6)
                        P_R = np.array([q_0[6:12],
                                        q_R_f]).T
                        V_R = np.array([np.zeros(6),
                                        np.zeros(6),]).T
                        self.traj_R = InitCubicTrajectories(P=P_R,
                                                       V=V_R,
                                                       Times=[0,self.home_interval],
                                                       n=6)
                        self.home_state = "evaluate_trajectories"
                    case "evaluate_trajectories":
                        self.controller_enable = True
                        if self.trajectory_time <= self.home_interval:
                            for i in range(len(self.traj_L)):
                                self.L_leg_q[i]  = self.traj_L[i].value(self.trajectory_time)[0][0]
                                self.R_leg_q[i]  = self.traj_R[i].value(self.trajectory_time)[0][0]
                                self.L_leg_qd[i] = self.traj_L[i].EvalDerivative(self.trajectory_time, 1)[0][0]
                                self.R_leg_qd[i] = self.traj_R[i].EvalDerivative(self.trajectory_time, 1)[0][0]
                            self.trajectory_time += self.timer_period
                        else: 
                            self.home_state = "done"
                    case "done":
                        self.operation_state = "hold_position"
                        self.controller_enable = True
                        pass
                
                if self.walk_command == -1:
                    self.operation_state = "disable_controller"
                elif self.walk_command == 0:
                    self.operation_state = "home"
                elif self.walk_command == 1:
                    self.operation_state = "task_execution"
                elif self.walk_command == 2:
                    self.operation_state = "hold_position"

            case "task_execution":
                self.controller_enable = True
                self.UpdateDesirejointStates()
                if self.walk_command == -1:
                    self.operation_state = "disable_controller"
                elif self.walk_command == 0:
                    self.operation_state = "home"
                    self.home_state = "init_trajectories"
                elif self.walk_command == 1:
                    self.operation_state = "task_execution"
                elif self.walk_command == 2:
                    self.operation_state = "hold_position"

        q_desire = np.concatenate((np.array([self.L_leg_q]), np.array([self.R_leg_q])),axis=1)
        qd_desire = np.concatenate((np.array([self.L_leg_qd]), np.array([self.R_leg_qd])),axis=1)
        motor_ids = [1,2,3,4,5,6,11,12,13,14,15,16]
        if self.controller_enable:

            kps = [0.0, 0.0, 0.0, 0.0, 0.0, 3.0,   # L
                   0.0, 0.0, 0.0, 0.0, 0.0, 2.0]   # R
            
            kds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.01,  # L
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.02
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   ]   # R
        else:
            kps = np.zeros(12)
            kds = np.zeros(12)
        print("kps :\n", kps)
        print("kds :\n", kds)
        for id in range(len(motor_ids)):
            motor = MotorControl()
            motor.motor_id = motor_ids[id]
            if motor_ids[id] == 1 or motor_ids[id] == 11:
                motor.motor_serie = "XM430"
            else:
                motor.motor_serie = "XM540"
            motor.control_mode = 0  
            motor.set_point.position = q_desire[0,id]
            motor.set_point.velocity = qd_desire[0,id]
            motor.set_point.effort = self.statics_frictions_ff[id]
            motor.set_point.kp = kps[id]
            motor.set_point.kd = kds[id]
            msg.motor_controls.append(motor)

        self.publisher_.publish(msg)
    
    def walk_command_callback(self, msg:Int16):
        self.walk_command = msg.data

    def UpdateDesirejointStates(self):
        match self.walk_state:
            case WalkStates.stand:
                pass
            case WalkStates.init:
                if self.ind <= self.q_init_L.shape[1] - 200:
                    self.L_leg_q = self.q_init_L[:,self.ind]
                    self.R_leg_q = self.q_init_R[:,self.ind]

                    self.L_leg_qd = self.qd_init_L[:,self.ind]
                    self.R_leg_qd = self.qd_init_R[:,self.ind]

                    self.ind += 1
                else:
                    self.L_leg_qd = np.zeros(6)
                    self.R_leg_qd = np.zeros(6)                
                    # self.walk_state = WalkStates.walk
                    # self.ind = 0
            case WalkStates.walk:
                if self.ind > self.sw_leg_q.shape[1] - 200:
                    self.ind = 0
                    for i in [1,5]:
                        self.st_leg_q[i,:] = self.st_leg_q[i,:]*-1.0
                        self.sw_leg_q[i,:] = self.sw_leg_q[i,:]*-1.0

                        self.st_leg_qd[i,:] = self.st_leg_qd[i,:]*-1.0
                        self.sw_leg_qd[i,:] = self.sw_leg_qd[i,:]*-1.0

                    self.L_stance = not self.L_stance

                if self.L_stance:
                    self.L_leg_q = self.st_leg_q[:,self.ind]
                    self.R_leg_q = self.sw_leg_q[:,self.ind]

                    self.L_leg_qd = self.st_leg_qd[:,self.ind]
                    self.R_leg_qd = self.sw_leg_qd[:,self.ind]
                else:
                    self.L_leg_q = self.sw_leg_q[:,self.ind]
                    self.R_leg_q = self.st_leg_q[:,self.ind] 

                    self.L_leg_qd = self.sw_leg_qd[:,self.ind]
                    self.R_leg_qd = self.st_leg_qd[:,self.ind]                   
                self.ind += 1

def InitCubicTrajectories(P, V, Times, n):
    trajs = []
    for i in range(n):
        trajs.append(PiecewisePolynomial.CubicHermite(np.array(Times), 
                                                      np.array([P[i,:]]), 
                                                      np.array([V[i,:]])))
    return trajs

def main(args=None):
    rclpy.init(args=args)
    node = MotorControlGroupPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
