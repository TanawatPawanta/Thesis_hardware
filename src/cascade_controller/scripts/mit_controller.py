#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

# Message มาตรฐานสำหรับ JointState
from sensor_msgs.msg import JointState

# สมมติว่าเรามีข้อความสั่งงานกระแสมอเตอร์เป็นแบบ Float64MultiArray
# โดยใน field data จะเก็บค่า iq_ref ของแต่ละมอเตอร์
from std_msgs.msg import Float64MultiArray, Int16MultiArray

# ตัวอย่างข้อความสั่งงานเป้าหมาย (desired angle, desired speed)
# สามารถเปลี่ยนเป็น message ชนิดอื่นได้ตามต้องการ
# ในที่นี้สมมติใช้ Float64MultiArray โดยให้ data เรียงดังนี้:
# [desired_angle_motor0, desired_speed_motor0,
#  desired_angle_motor1, desired_speed_motor1,
#  ...]
# หรืออาจจะแยกเป็นคนละ Topic ก็ได้
from std_msgs.msg import Float64MultiArray as DesiredJointMsg

# 0.083 A -> Current Operate
# 2.69 mA
# Kt = 1.75 [Nm] / [A]
# setpoint      weight (g)      joint state     current
# 10


class MotorControllerNode(Node):
    def __init__(self):
        super().__init__('motor_controller_node')

        # ประกาศ parameter Kp, Kd (ค่าเริ่มต้นปรับได้)
        self.declare_parameter('kp', 1.0)
        self.declare_parameter('kd', 0.1)

        # อ่านค่าพารามิเตอร์มาเก็บ
        self.kp = self.get_parameter('kp').value
        self.kd = self.get_parameter('kd').value

        # จำนวนมอเตอร์ (ปรับตามจำนวนจริง)
        self.num_motors = 1

        # ตัวแปรเก็บสถานะ joint ปัจจุบัน
        # จะเก็บเป็น list ตามจำนวนมอเตอร์
        self.current_angle = [0.0] * self.num_motors
        self.current_speed = [0.0] * self.num_motors
        self.current_effort = [0.0] * self.num_motors

        # ตัวแปรเก็บค่าที่สั่ง (desired) จากภายนอก
        self.desired_angle = [0.0] * self.num_motors
        self.desired_speed = [0.0] * self.num_motors
        self.desired_effort = [0.0] * self.num_motors

        # สร้าง Subscriber สำหรับ joint state (ตำแหน่งและความเร็วที่วัดได้จริง)
        # สมมติ topic ชื่อ /joint_states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # สร้าง Subscriber สำหรับรับค่ามุมและความเร็วอ้างอิง (desired)
        # สมมติ topic ชื่อ /desired_joints
        self.desired_sub = self.create_subscription(
            DesiredJointMsg,
            '/desired_joints',
            self.desired_joints_callback,
            10
        )

        # สร้าง Publisher สำหรับสั่งกระแส (iq) ไปยังไดร์ฟมอเตอร์
        # สมมติ topic ชื่อ /motor_currents
        self.iq_pub = self.create_publisher(Int16MultiArray, '/group_goal_current', 10)
        
        # ตั้ง timer ให้ทำงานคำนวณควบคุมเป็นระยะ
        self.control_timer = self.create_timer(1/60, self.control_loop)  # 100 Hz

        self.get_logger().info('Motor Controller Node has been started.')

    def joint_state_callback(self, msg: JointState):
        """
        Callback สำหรับอ่านค่าตำแหน่ง (position) และความเร็ว (velocity) ของมอเตอร์
        โดย msg.name จะเป็นรายชื่อ joint
        msg.position, msg.velocity เป็น list ตำแหน่งและความเร็วตามลำดับ
        จะต้อง mapping ให้ถูกกับ self.current_angle, self.current_speed
        """
        # ตัวอย่างสมมติว่ามี joint 2 ตัว ชื่อ "joint0", "joint1"
        # ถ้าทำงานกับหลายมอเตอร์ ต้องตรวจสอบให้แน่ใจว่า index เรียงถูกต้อง
        for i, joint_name in enumerate(msg.name):
            if i < self.num_motors:
                self.current_angle[i] = msg.position[i]
                self.current_speed[i] = msg.velocity[i]
                self.current_effort[i] = msg.effort[i]

    def desired_joints_callback(self, msg: DesiredJointMsg):
        """
        Callback สำหรับรับค่า desired angle และ desired speed
        ในตัวอย่างนี้ใช้ Float64MultiArray ที่เรียงข้อมูลแบบ
        [angle0, speed0, angle1, speed1, ...]
        """
        data = msg.data
        if len(data) == 2 * self.num_motors:
            for i in range(self.num_motors):
                self.desired_angle[i] = data[2 * i]
                self.desired_speed[i] = data[2 * i + 1]

    def control_loop(self):
        """
        ฟังก์ชันควบคุมหลัก ทำงานทุก ๆ timer callback
        1. คำนวณ error ของ angle และ speed
        2. หาค่า torque = Kp*error_angle + Kd*error_speed
        3. แปลง torque เป็น iq (สมมติว่ามี torque_constant = 1.0 เพื่อความง่าย)
        4. ส่งค่า iq_ref ไปยัง publisher
        """
        iq_command = Int16MultiArray()
        iq_list = []

        torque_constant = 1.0

        self.kp = 5.0
        self.kd = 0.50
        for i in range(self.num_motors):
            error_angle = self.desired_angle[i] - self.current_angle[i]
            error_speed = self.desired_speed[i] - self.current_speed[i]

            torque = (self.kp * error_angle) + (self.kd * error_speed)

            iq_ref = torque / torque_constant
            iq_ref = iq_ref * 10.0
            iq_list.append(int(iq_ref))

        iq_command.data = iq_list
        self.iq_pub.publish(iq_command)


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
