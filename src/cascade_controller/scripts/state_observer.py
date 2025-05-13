#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from cascade_controller_interfaces.msg import MotorControl, MotorControlGroup, SetPoint
import numpy as np

# Adaptive Kalman Filter สำหรับ state vector = [position, velocity, current]
class AdaptiveKalmanFilter3:
    def __init__(self, A, B, C, Q, R, P, x, auto_tune=True, lambda_R=0.99):
        self.A = A        # Matrix A (3x3)
        self.B = B        # Matrix B (3x1)
        self.C = C        # Matrix C (3x3)
        self.Q = Q        # Process noise covariance (3x3)
        self.R = R        # Measurement noise covariance (3x3)
        self.P = P        # Error covariance (3x3)
        self.x = x        # Initial state estimate (3x1)
        self.auto_tune = auto_tune
        self.lambda_R = lambda_R  # Forgetting factor สำหรับ auto-tuning

    def predict(self, u):
        # ขั้นตอน prediction: x[k+1] = A*x[k] + B*u
        self.x_pred = self.A @ self.x + self.B * u
        self.P_pred = self.A @ self.P @ self.A.T + self.Q
        return self.x_pred

    def update(self, y):
        # คำนวณ innovation (ความแตกต่างระหว่างการวัดกับการคาดการณ์)
        innovation = y - self.C @ self.x_pred
        S = self.C @ self.P_pred @ self.C.T + self.R
        K = self.P_pred @ self.C.T @ np.linalg.inv(S)
        self.x = self.x_pred + K @ innovation
        I = np.eye(self.A.shape[0])
        self.P = (I - K @ self.C) @ self.P_pred

        # Auto-tune ค่า R จาก innovation covariance
        if self.auto_tune:
            innovation_cov = innovation @ innovation.T
            C_P_pred_Ct = self.C @ self.P_pred @ self.C.T
            R_new = self.lambda_R * self.R + (1 - self.lambda_R) * (innovation_cov - C_P_pred_Ct)
            R_new_diag = np.maximum(np.diag(R_new), 1e-6)
            self.R = np.diag(R_new_diag)
            # print(R_new_diag)
        return self.x

    def get_state(self):
        return self.x

# Node สำหรับ State Estimation (แยกจาก node ควบคุมมอเตอร์)
class StateEstimatorNode(Node):
    def __init__(self):
        super().__init__('state_estimator_node')
        
        # Subscriber รับ joint_states ที่ publish จาก node ควบคุมมอเตอร์
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )
        
        # Subscriber รับคำสั่งควบคุม (MotorControlGroup)
        self.command_sub = self.create_subscription(
            MotorControlGroup,
            'mit_mode_control',
            self.command_callback,
            10
        )
        
        # Publisher สำหรับส่งออก estimated state (ใช้ JointState message เป็นตัวอย่าง)
        self.estimated_state_pub = self.create_publisher(JointState, 'estimated_states', 10)
        
        # Timer loop สำหรับทำงานที่ 100 Hz
        self.timer_period = 0.001  # 10 ms
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.last_time = self.get_clock().now()
        
        # เก็บข้อมูลล่าสุดจากการวัดและคำสั่งในรูปแบบ dictionary
        # key: motor_id, value: dict สำหรับ measurement หรือ command
        self.latest_measurements = {}  # จาก topic "joint_states"
        self.latest_commands = {}      # จาก topic "motor_group_command"
        
        # สมมติว่าระบบมีมอเตอร์ 6 ตัว (ปรับตามจริง)
        self.motors = [41]
        
        # สร้าง Adaptive Kalman Filter สำหรับแต่ละมอเตอร์
        self.kalman_filters = {}
        dt = self.timer_period
        # พารามิเตอร์ประมาณ: กำหนด k และ tau (ปรับจูนได้)
        k = 1.0    # Gain จาก current ไปสู่ acceleration
        tau = 1.0  # Time constant ของระบบไฟฟ้า
        # นิยาม state-space โดยใช้ Forward Euler:
        A = np.array([[1, dt,      0],
                      [0,  1, dt * k],
                    #   [0,  0, 1 - dt/tau]])
                      [0,  0, 1]])
        B = np.array([[0],
                      [0],
                      [dt/tau]])
        C = np.eye(3)
        Q = np.eye(3) * 0.01  # Process noise covariance
        R = np.eye(3)   # Measurement noise covariance (เริ่มต้น)
        R[0,0] = 10.1
        R[1,1] = 10.1
        R[2,2] = 10.1
        P = np.eye(3) * 100   # Initial error covariance
        x0 = np.zeros((3,1))   # State เริ่มต้น: [0, 0, 0]
        for motor_id in self.motors:
            self.kalman_filters[motor_id] = AdaptiveKalmanFilter3(
                A.copy(), B.copy(), C.copy(), Q.copy(), R.copy(), P.copy(), x0.copy(),
                auto_tune=False, lambda_R=0.99
            )
        
        self.get_logger().info("StateEstimatorNode initialized.")
    
    def joint_state_callback(self, msg: JointState):
        """
        รับข้อมูล joint_states จาก node ควบคุมมอเตอร์  
        สมมติว่า msg.name มีรูปแบบ "motor_<id>" และ
        msg.position, msg.velocity, msg.effort (ซึ่ง effort แทน current)
        """
        for i, name in enumerate(msg.name):
            try:
                motor_id = int(name.split('_')[-1])
            except ValueError:
                continue
            self.latest_measurements[motor_id] = {
                'position': msg.position[i],
                'velocity': msg.velocity[i],
                'current': msg.effort[i]
            }
    
    def command_callback(self, msg: MotorControlGroup):
        """
        รับคำสั่งควบคุมจาก topic "motor_group_command"  
        สมมติว่า msg.motor_controls เป็น array ที่มีฟิลด์: position, velocity, effort, kp, kd  
        โดยลำดับใน array สอดคล้องกับลำดับของมอเตอร์ใน self.motors
        """
        if len(msg.motor_controls) != len(self.motors):
            self.get_logger().warn("Received command length does not match number of motors.")
            return
        for i, motor_id in enumerate(self.motors):
            ctrl = msg.motor_controls[i]
            self.latest_commands[motor_id] = {
                'pos_cmd': ctrl.set_point.position,
                'vel_cmd': ctrl.set_point.velocity,
                'torque_cmd': ctrl.set_point.effort,
                'kp': ctrl.set_point.kp,
                'kd': ctrl.set_point.kd
            }
    
    def timer_callback(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        
        estimated_states = {}
        # สำหรับแต่ละมอเตอร์ หากมีทั้ง measurement และ command ให้ประมวลผล
        for motor_id in self.motors:
            if motor_id in self.latest_measurements and motor_id in self.latest_commands:
                meas = self.latest_measurements[motor_id]
                cmd = self.latest_commands[motor_id]
                # นิยาม effective input: u_eff = kp*(pos_cmd - measured_pos) + kd*(vel_cmd - measured_vel) + torque_cmd
                u_eff = (cmd['kp'] * (cmd['pos_cmd'] - meas['position']) +
                         cmd['kd'] * (cmd['vel_cmd'] - meas['velocity']) +
                         cmd['torque_cmd'])
                # ดึง Kalman Filter สำหรับ motor นี้
                kf = self.kalman_filters[motor_id]
                kf.predict(np.array([[u_eff]]))
                # วัดได้: y = [position, velocity, current]
                y = np.array([[meas['position']],
                              [meas['velocity']],
                              [meas['current']]])
                x_est = kf.update(y)
                estimated_states[motor_id] = x_est
                # self.get_logger().info(f"Motor {motor_id} est. state: pos={x_est[0,0]:.4f}, vel={x_est[1,0]:.4f}, cur={x_est[2,0]:.4f}")
        
        # สร้างและ publish JointState สำหรับ estimated states
        js_msg = JointState()
        js_msg.header.stamp = now.to_msg()
        names = []
        positions = []
        velocities = []
        efforts = []  # publish estimated current ใน field effort
        for motor_id in self.motors:
            names.append(f"motor_{motor_id}")
            if motor_id in estimated_states:
                x_est = estimated_states[motor_id]
                positions.append(x_est[0,0])
                velocities.append(x_est[1,0])
                efforts.append(x_est[2,0])
            else:
                positions.append(0.0)
                velocities.append(0.0)
                efforts.append(0.0)
        js_msg.name = names
        js_msg.position = positions
        js_msg.velocity = velocities
        js_msg.effort = efforts
        self.estimated_state_pub.publish(js_msg)

def main(args=None):
    rclpy.init(args=args)
    node = StateEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
