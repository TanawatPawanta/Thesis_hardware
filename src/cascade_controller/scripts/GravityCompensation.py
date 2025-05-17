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
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
import os


class GravityCompensationNode(Node):
    def __init__(self):
        super().__init__('gravity_compensation_node')

        self.num_motors = 12
        
        self.current_position   = [0.0] * self.num_motors
        self.current_velocity   = [0.0] * self.num_motors
        self.current_effort     = [0.0] * self.num_motors

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.foot_P_com_pub = self.create_publisher(Float64MultiArray,
                                                    '/foot_P_com', 10)
        self.com_maker_pub = self.create_publisher(Marker,
                                                   '/com_marker',10)
        self.tau_g_pub = self.create_publisher(Float64MultiArray,
                                               '/tau_g',10)
        
        self.update_event_timer = self.create_timer(0.02, self.UpdateEvent)  # 50 Hz

        self.get_logger().info('Gravity Compensation Node has been started.')
        self.PreparePlant()
        self.stance_leg = "double_stance"

    def joint_state_callback(self, msg: JointState):
        for i, joint_name in enumerate(msg.name):
            if i < self.num_motors:
                self.current_position[i] = msg.position[i]
                self.current_velocity[i] = msg.velocity[i]
                self.current_effort[i] = msg.effort[i]

    def UpdateEvent(self):
        q = np.array(self.current_position)
        
        tau_g = self.CalcGravityCompensation(q=q)
        print("<< stance leg >> :", self.stance_leg,"<========================")
        print("Torque g L: \n",tau_g[0:6])
        print("Torque g R: \n",tau_g[6:12])
        print("-"*100)
        self.PubTauG(tau_g=list(tau_g))

    def PreparePlant(self):
        self.plant_half, dump = DefinePlant(model_name="humanoid_half_body",
                             is_fixed=True,
                             fixed_frame="torso")
        self.plant_half_context = self.plant_half.CreateDefaultContext()       

        self.plant_R_fixed, dump = DefinePlant(model_name="humanoid_R_fixed",
                             is_fixed=True,
                             fixed_frame="R_foot_contact")
        self.plant_R_fixed_context = self.plant_R_fixed.CreateDefaultContext()

        self.plant_L_fixed, dump = DefinePlant(model_name="humanoid_L_fixed",
                             is_fixed=True,
                             fixed_frame="L_foot_contact")
        self.plant_L_fixed_context = self.plant_L_fixed.CreateDefaultContext()

        self.plant_double_stance, dump = DefinePlant(model_name="humanoid_only_leg",
                             is_fixed=True,
                             fixed_frame="L_foot_contact")
        self.plant_double_stance_context = self.plant_double_stance.CreateDefaultContext()

    def CalcGravityCompensation(self,q):
        self.UpdateStanceLeg(q=q) 
        q_L = q[0:6]
        q_R = q[6:12]
        qd = np.zeros(12)
        match self.stance_leg:
            case("double_stance"):
                self.plant_double_stance.SetPositions(self.plant_double_stance_context,
                                                    -np.array(q_L)[::-1])
                tau_g_L = self.plant_double_stance.CalcGravityGeneralizedForces(context=self.plant_double_stance_context)
                tau_g_L = [0.45,0.45,0.2,0.45,0.45,0.45]*np.array(tau_g_L)[::-1]
                self.plant_double_stance.SetPositions(self.plant_double_stance_context,
                                                    -np.array(q_R)[::-1])
                tau_g_R = self.plant_double_stance.CalcGravityGeneralizedForces(context=self.plant_double_stance_context)
                # [0.3,0.3,0.3,0.3,0.3,0.3] [0.5,0.5,0.5,0.5,0.7,0.6]
                tau_g_R = [0.45,0.45,0.2,0.45,0.45,0.45]* np.array(tau_g_R)[::-1]
            case("R_stance"):
                self.plant_R_fixed.SetPositions(self.plant_R_fixed_context, 
                                                np.concatenate((-q_R[::-1],q_L)))
                tau_g = 1*self.plant_R_fixed.CalcGravityGeneralizedForces(context=self.plant_R_fixed_context)
                tau_g_L = -0.5*tau_g[6:]
                tau_g_R = 0.5*tau_g[0:6][::-1]
            case("L_stance"): 
                self.plant_L_fixed.SetPositions(self.plant_L_fixed_context, 
                                                np.concatenate((-q_L[::-1],q_R)))
                tau_g = 1*self.plant_L_fixed.CalcGravityGeneralizedForces(context=self.plant_L_fixed_context)
                tau_g_L = 0.5*tau_g[0:6][::-1] 
                tau_g_R = -0.5*tau_g[6:]        

        torques = np.concatenate((tau_g_L,tau_g_R))
        return torques
        
    def UpdateStanceLeg(self, q):
        self.plant_half.SetPositions(self.plant_half_context, q)
        W_P_com = self.plant_half.CalcCenterOfMassPositionInWorld(self.plant_half_context)
        L_foot_contact = self.plant_half.GetFrameByName("L_foot_contact")
        torso_T_L = L_foot_contact.CalcPoseInWorld(self.plant_half_context)
        L_P_com = torso_T_L.inverse().multiply(W_P_com)
        R_foot_contact = self.plant_half.GetFrameByName("R_foot_contact")
        R_P_com = R_foot_contact.CalcPoseInWorld(self.plant_half_context).inverse().multiply(W_P_com)
        offset = 10
        foot_P_com = Float64MultiArray()
        foot_P_com.data = list(np.concatenate((L_P_com,R_P_com)))
        self.foot_P_com_pub.publish(foot_P_com)
        self.PubCoMMarker(torso_P_com=W_P_com)
        torso_high = -1*torso_T_L.translation()[2]
        print("torso_high : ",torso_high)
        print("L_P_com    : ",L_P_com)
        print("R_P_com    : ",R_P_com)
        L_foot_rect = np.array([-69+offset,-27+offset,
                                 96-offset, 63-offset]) * 1e-3
        R_foot_rect = np.array([-69+offset,-63+offset,
                                 96-offset, 27-offset]) * 1e-3
        # if IsInsideRectangle(point=L_P_com[0:2], rect=L_foot_rect):
        #     self.stance_leg = "L_stance"
        # elif IsInsideRectangle(point=R_P_com[0:2], rect=R_foot_rect):
        #     self.stance_leg = "R_stance"
        # else:
        #     self.stance_leg = "double_stance"

    def PubCoMMarker(self, torso_P_com):
        marker = Marker()
        marker.header.frame_id = "torso"  # Or your root link
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "center_of_mass"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = torso_P_com[0]
        marker.pose.position.y = torso_P_com[1]
        marker.pose.position.z = torso_P_com[2]
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.015
        marker.scale.y = 0.015
        marker.scale.z = 0.015
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.1
        self.com_maker_pub.publish(marker)

    def PubTauG(self,tau_g):
        msg = Float64MultiArray()
        msg.data = tau_g
        self.tau_g_pub.publish(msg)


def DefinePlant(model_name:str,is_fixed:bool,fixed_frame:str)->MultibodyPlant:
    plant = MultibodyPlant(time_step=0.0)
    parser = Parser(plant)
    urdf = model_name+".urdf"
    dir = "/home/tanawatp/dynamixel_ws/src/cascade_controller/scripts/robot_model"
    robot_urdf_path = os.path.join(dir,urdf)
    robot_instant_index = parser.AddModels(robot_urdf_path)
    if is_fixed:
        robot_fixed_frame = plant.GetFrameByName(fixed_frame)
        init_pos = [0.0, 0.0, 0.0] 
        init_orien = np.asarray([0, 0, 0])
        X_WRobot = RigidTransform(
        RollPitchYaw(init_orien * np.pi / 180), p=init_pos)
        plant.WeldFrames(plant.world_frame(),robot_fixed_frame, X_WRobot)
    plant.Finalize()
    return plant, robot_instant_index

def IsInsideRectangle(point, rect):
    """
    Check if a point is inside a rectangle.

    Parameters:
        point: np.array([x, y])
        rect: np.array([x_min, y_min, x_max, y_max])

    Returns:
        True if point is inside or on the edge of the rectangle.
    """
    x, y = point
    x_min, y_min, x_max, y_max = rect
    return x_min <= x <= x_max and y_min <= y <= y_max

def main(args=None):
    rclpy.init(args=args)
    node = GravityCompensationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
