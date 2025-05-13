#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Int16MultiArray
from std_msgs.msg import Float64MultiArray as DesiredJointMsg
from cascade_controller_interfaces.msg import MotorControl, MotorControlGroup, SetPoint
import numpy as np
# from Utils.UtilsFunc import *
# 0.083 A -> Current Operate
# 2.69 mA
# Kt = 1.75 [Nm] / [A]
# setpoint      weight (g)      joint state     current
# 10


class MotorControllerNode(Node):
    def __init__(self):
        super().__init__('mit_controller_node')

        self.num_motors = 12

        self.motor_series = ["XM430","XM540","XM540","XM540","XM540","XM540",
                             "XM430","XM540","XM540","XM540","XM540","XM540"]
        self.XM540_torque_constant = 10.6/4.4
        self.XM430_torque_constant = 4.1/2.3

        self.kp = [0.0] * self.num_motors
        self.kd = [0.0] * self.num_motors

        self.current_position   = [0.0] * self.num_motors
        self.current_velocity   = [0.0] * self.num_motors
        self.current_effort     = [0.0] * self.num_motors

        self.desired_position   = [0.0] * self.num_motors
        self.desired_velocity   = [0.0] * self.num_motors
        self.desired_effort     = [0.0] * self.num_motors
        
        self.tau_g = np.zeros(self.num_motors)

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.desired_sub = self.create_subscription(
            MotorControlGroup,
            '/mit_mode_control',
            self.desired_joints_callback,
            10
        )
        self.tau_g_sub = self.create_subscription(
            Float64MultiArray,
            '/tau_g',
            self.tau_g_callback,
            10
        )

        self.iq_pub = self.create_publisher(Int16MultiArray, '/group_goal_current', 10)
        self.com_maker_pub = self.create_publisher(Marker,'/com_marker',10)
        self.joint_position_errors_pub = self.create_publisher(Float64MultiArray,
                                                               '/joint_position_errors',10)
        self.joint_velocity_errors_pub = self.create_publisher(Float64MultiArray,
                                                               '/joint_velocity_errors',10)
        self.control_timer = self.create_timer(0.02, self.control_loop)  # 50 Hz

        self.get_logger().info('Motor Controller Node has been started.')


    def joint_state_callback(self, msg: JointState):
        for i, joint_name in enumerate(msg.name):
            if i < self.num_motors:
                self.current_position[i] = msg.position[i]
                self.current_velocity[i] = msg.velocity[i]
                self.current_effort[i] = msg.effort[i]

    def desired_joints_callback(self, msg: MotorControlGroup):
        data = msg.motor_controls
        if len(data) == self.num_motors:
            for i in range(self.num_motors):
                # self.motor_series[i] = data[i].motor_serie
                self.desired_position[i] = data[i].set_point.position
                self.desired_velocity[i] = data[i].set_point.velocity
                self.desired_effort[i] = data[i].set_point.effort
                self.kp[i] = data[i].set_point.kp
                self.kd[i] = data[i].set_point.kd
    
    def tau_g_callback(self,msg: Float64MultiArray):
        self.tau_g = msg.data

    def control_loop(self):
        iq_list = []
        q_errors = []
        qd_errors = []
        for i in range(self.num_motors):
            error_position = self.desired_position[i] - self.current_position[i]
            error_velocity = self.desired_velocity[i] - self.current_velocity[i]
            q_errors.append(error_position)
            qd_errors.append(error_velocity)
            static_friction_ff = 0.0
            if abs(error_position) <= 0.001:
                error_position = 0.0 
                static_friction_ff = 0.0
            else:
                if error_position >= 0.0:
                    static_friction_ff = self.desired_effort[i]
                elif error_position < 0.0:
                    static_friction_ff = -1.0*self.desired_effort[i]
            self.PubJoinErrors(q_errors=q_errors,
                               qd_errors=qd_errors)
            if self.motor_series[i] == "XM540":
                torque_constant = self.XM540_torque_constant
            elif self.motor_series[i] == "XM430":
                torque_constant = self.XM430_torque_constant
            else:
                torque_constant = 2.0
            tau_u = (self.kp[i] * error_position) + (self.kd[i] * error_velocity) + static_friction_ff
            if self.kp[i] == 0:
                tau_u = 0.0
            torques = self.tau_g[i] + tau_u
            iq_ref = (torques / torque_constant) 
            iq_ref = iq_ref / (0.00269)

            iq_int_ref = int(iq_ref)
            iq_ff = 1
            if iq_int_ref > 0:
                iq_int_ref = iq_int_ref + iq_ff
            elif iq_int_ref < 0:
                iq_int_ref = iq_int_ref - iq_ff
            iq_list.append(iq_int_ref)
        iq_command = Int16MultiArray()
        iq_command.data = iq_list
        self.iq_pub.publish(iq_command)

    def PubJoinErrors(self,q_errors,qd_errors):
        position_errors = Float64MultiArray()
        position_errors.data = q_errors
        self.joint_position_errors_pub.publish(position_errors)
        velocity_errors = Float64MultiArray()
        velocity_errors.data = qd_errors
        self.joint_velocity_errors_pub.publish(velocity_errors)


def main(args=None):
    rclpy.init(args=args)
    node = MotorControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
