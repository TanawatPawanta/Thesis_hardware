import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Int16MultiArray
from sensor_msgs.msg import JointState

class MotorController:
    """คลาสสำหรับเก็บ state และ PID ของแต่ละมอเตอร์"""
    def __init__(self, motor_id, outer_params, inner_params):
        self.motor_id = motor_id
        self.target_position = 0.0
        self.current_position = 0.0
        self.current_velocity = 0.0

        # สำหรับ outer loop (position -> velocity setpoint)
        self.outer_Kp = outer_params.get('Kp', 1.0)
        self.outer_Ki = outer_params.get('Ki', 0.1)
        self.outer_Kd = outer_params.get('Kd', 0.05)
        self.outer_integral = 0.0
        self.outer_previous_error = 0.0
        self.velocity_setpoint = 0.0  # ผลลัพธ์จาก outer loop

        # สำหรับ inner loop (velocity -> control effort)
        self.inner_Kp = inner_params.get('Kp', 100.0)
        self.inner_Ki = inner_params.get('Ki', 0.1)
        self.inner_Kd = inner_params.get('Kd', 0.05)
        self.inner_integral = 0.0
        self.inner_previous_error = 0.0

    def compute_outer(self, dt):
        """
        Outer loop: คำนวณ velocity setpoint จาก target position กับ current position
        """
        error_outer = 2.0 - self.current_position
        self.outer_integral += error_outer * dt
        derivative_outer = (error_outer - self.outer_previous_error) / dt if dt > 0 else 0.0
        self.velocity_setpoint = (self.outer_Kp * error_outer +
                                  self.outer_Ki * self.outer_integral +
                                  self.outer_Kd * derivative_outer)
        self.outer_previous_error = error_outer
        return self.velocity_setpoint

    def compute_inner(self, dt):
        """
        Inner loop: คำนวณ control effort จาก velocity setpoint (ที่ได้จาก outer loop) กับ current velocity
        """
        error_inner = self.velocity_setpoint - self.current_velocity
        # error_inner = 1.5 - self.current_velocity
        self.inner_integral += error_inner * dt
        derivative_inner = (error_inner - self.inner_previous_error) / dt if dt > 0 else 0.0
        control_effort = (self.inner_Kp * error_inner +
                          self.inner_Ki * self.inner_integral +
                          self.inner_Kd * derivative_inner)
        self.inner_previous_error = error_inner
        return control_effort

class MultiMotorCascadeController(Node):
    def __init__(self):
        super().__init__('multi_motor_cascade_controller')
        # Parameter: รายการ motor_ids ที่จะควบคุม
        self.declare_parameter('motor_ids', ['motor_41', 'motor_42'])
        motor_ids = self.get_parameter('motor_ids').get_parameter_value().string_array_value

        # PID parameters เบื้องต้น (สามารถปรับให้เป็น parameter เพิ่มเติมได้)
        outer_params = {'Kp': 0.50, 'Ki': 0.0, 'Kd': 0.0}
        inner_params = {'Kp': 4.0, 'Ki': 1.5, 'Kd': 0.05}

        self.motors = {}
        self.target_subs = {}
        # Publisher สำหรับส่งคำสั่งควบคุมแบบ group (Int16MultiArray)
        self.group_goal_pub = self.create_publisher(Int16MultiArray, '/group_goal_current', 10)

        # สร้าง subscriber สำหรับ target position ของแต่ละ motor
        for motor_id in motor_ids:
            self.motors[motor_id] = MotorController(motor_id, outer_params, inner_params)
            target_topic = f'/{motor_id}/target_position'
            self.target_subs[motor_id] = self.create_subscription(
                Float64,
                target_topic,
                lambda msg, motor_id=motor_id: self.target_callback(msg, motor_id),
                10)

        # Subscriber สำหรับ joint_states ที่ publish โดย joint_state_pub
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        # Parameter สำหรับ limit saturation ของ control effort
        self.declare_parameter('control_effort_min', -1000)
        self.declare_parameter('control_effort_max', 1000)
        self.control_effort_min = self.get_parameter('control_effort_min').get_parameter_value().integer_value
        self.control_effort_max = self.get_parameter('control_effort_max').get_parameter_value().integer_value

        # Timer สำหรับ outer loop (50 Hz)
        self.outer_timer_period = 0.05  # 20 ms
        self.outer_timer = self.create_timer(self.outer_timer_period, self.outer_loop)

        # Timer สำหรับ inner loop (100 Hz)
        self.inner_timer_period = 0.02  # 10 ms
        self.inner_timer = self.create_timer(self.inner_timer_period, self.inner_loop)

    def target_callback(self, msg: Float64, motor_id: str):
        self.motors[motor_id].target_position = msg.data
        self.get_logger().debug(f"[{motor_id}] Received target_position: {msg.data}")

    def joint_state_callback(self, msg: JointState):
        # สมมุติว่า msg.name เป็น list ของชื่อ joint ที่ตรงกับ motor_id
        for i, joint_name in enumerate(msg.name):
            if joint_name in self.motors:
                self.motors[joint_name].current_position = msg.position[i]
                if msg.velocity:
                    self.motors[joint_name].current_velocity = msg.velocity[i]
                self.get_logger().debug(
                    f"[{joint_name}] Updated from joint_states: position={msg.position[i]}, velocity={msg.velocity[i] if msg.velocity else 0.0}")

    def outer_loop(self):
        """
        Outer loop: คำนวณ velocity setpoint สำหรับแต่ละ motor (ทำงานที่ 50 Hz)
        """
        for motor_id, motor in self.motors.items():
            vel_sp = motor.compute_outer(self.outer_timer_period)
            self.get_logger().debug(f"[{motor_id}] Outer loop velocity_setpoint: {vel_sp:.2f}")

    def inner_loop(self):
        """
        Inner loop: คำนวณ control effort จาก velocity setpoint กับ current velocity 
        แล้ว publish group command (ทำงานที่ 100 Hz)
        """
        control_efforts = []
        for motor_id, motor in self.motors.items():
            control_effort = motor.compute_inner(self.inner_timer_period)
            # แปลงเป็น integer พร้อมทำ limit saturation
            int_control_effort = int(round(control_effort))
            int_control_effort = max(self.control_effort_min, min(int_control_effort, self.control_effort_max))
            control_efforts.append(int_control_effort)
            self.get_logger().debug(
                f"[{motor_id}] Inner loop control_effort: {control_effort:.2f} -> {int_control_effort}")
        
        group_msg = Int16MultiArray()
        group_msg.data = control_efforts
        self.group_goal_pub.publish(group_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MultiMotorCascadeController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
