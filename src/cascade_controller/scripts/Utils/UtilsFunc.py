from scipy.spatial.transform import Rotation as R
from IPython.display import SVG, display
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import MultibodyForces
from pydrake.systems.primitives import LogVectorOutput
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import pydot

def quaternion_to_euler(Q)->list:
    w = Q[0]
    x = Q[1]
    y = Q[2]
    z = Q[3]
    r = R.from_quat([x, y, z, w])  # Note: SciPy uses [x, y, z, w] order
    out = r.as_euler('xyz', degrees=True)
    return [float(out[0]),float(out[1]),float(out[2])] # Returns roll, pitch, yaw

def Jleg_position(J_hip_y_q=0.0, J_hip_r_q=0.0, J_hip_p_q=0.0, J_knee_q=0.0, J_ankle_p_q=0.0, J_ankle_r_q=0.0):
    """
    Args:
        joint angles (float, optional): joint angles in deg. Defaults to 0.0.

    Returns:
        np.array: joint angles in rad
    """
    return np.array([[J_hip_y_q, J_hip_r_q, J_hip_p_q, J_knee_q, J_ankle_p_q, J_ankle_r_q]]) * np.pi/180.0

def Jarm_position(JL_shoulder_p_q=0.0, JL_shoulder_r_q=0.0, JL_elbow_q=0.0):
    return np.array([[JL_shoulder_p_q, JL_shoulder_r_q, JL_elbow_q]]) * np.pi/180.0

def Jhead_position(J_neck_q=0.0, J_cam_q=0.0):
    return np.array([[J_neck_q, J_cam_q]]) * np.pi/180.0

def show_diagram(diagram, show=True):
    if show:
        display(
        SVG(
            pydot.graph_from_dot_data(
                diagram.GetGraphvizString(max_depth=1))[0].create_svg()))
    else:
        pass

def ListJointsName(plant,model_instance_index):
    joint_ind = plant.GetJointIndices(model_instance_index[0]) # all joint index in plant
    print("GetJointIndices: ",joint_ind)
    for i in joint_ind:
        joint = plant.get_joint(i)  # get constant joint reference(read only) from plant
        print(f"Joint {i}: {joint.name()}")

def ListActuatorName(plant,model_instance_index):
    joint_ind = plant.GetJointActuatorIndices(model_instance_index) # all actuate joint index in plant
    for i in joint_ind:
        actuator = plant.get_joint_actuator(i)  # get constant actuate joint reference(read only) from plant
        print(f"Actuator {i}: {actuator.name()}")

def ListPlantOutputPort(plant):
    num_ports = plant.num_output_ports()    # number of output port of Multibody plant
    for i in range(num_ports):
        output_port = plant.get_output_port(i)
        print(f"Output Port {i}: {output_port.get_name()}, Type: {output_port.get_data_type()}")    

def ListStatesNames(plant):
    states_names = np.array([plant.GetStateNames()]).T
    print("Num state: ",states_names.size," states")
    print("Num position: ",plant.num_positions())
    print("Num velocity: ",plant.num_velocities())
    print(states_names)

def ListFrameNames(plant,model_instance_index):
    """
    Returns a list of all frame names in the given MultibodyPlant.

    Args:
        plant: An instance of pydrake.multibody.plant.MultibodyPlant.

    Returns:
        A list of strings representing the names of all frames in the plant.
    """
    frameID = plant.GetFrameIndices(model_instance_index[0])
    print("Robot frames name")
    for i in frameID:
        print(i,"->",plant.get_frame(i).name())

def ListBodyNames(plant, model_instance_index):
    for i in plant.GetBodyIndices(model_instance=model_instance_index[0]):
        body = plant.get_body(i)
        print(f"{i}: {body.name()}")

def RigidTransformToMatrix(rigid_transform: RigidTransform) -> np.ndarray:
    """
    Converts a Drake RigidTransform to a 4x4 NumPy homogeneous transformation matrix.

    Args:
        rigid_transform (RigidTransform): The RigidTransform to convert.

    Returns:
        np.ndarray: A 4x4 NumPy array representing the homogeneous transformation matrix.
    """
    # Access the rotation matrix and convert to NumPy array
    rotation_matrix_np = rigid_transform.rotation().matrix()  # Shape: (3, 3)

    # Access the translation vector as a NumPy array
    translation_vector_np = rigid_transform.translation()     # Shape: (3,)

    # Initialize a 4x4 identity matrix
    homogeneous_matrix = np.eye(4)

    # Assign the rotation matrix to the top-left 3x3 block
    homogeneous_matrix[:3, :3] = rotation_matrix_np

    # Assign the translation vector to the top-right 3x1 block
    homogeneous_matrix[:3, 3] = translation_vector_np

    return homogeneous_matrix

def DisplayGait(x_data, y_data, rectangle_centers, rectangle_size, description):
    """
    Plot a trajectory and rectangles in the same plot with equal resolution and a description below the plot.

    Parameters:
        x_data (list or array): X-axis data for the trajectory.
        y_data (list or array): Y-axis data for the trajectory.
        rectangle_centers (list of tuples): List of (x, y) coordinates for rectangle centers.
        rectangle_size (tuple): Size of rectangles (width, height).
        description (str): Description to display below the plot.

    Returns:
        None: Displays the plot.
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the trajectory
    ax.scatter(x_data, y_data, label="CoM Trajectory", color="blue",s=0.1, linewidth=1)


    # Add rectangles
    for center in rectangle_centers:
        rect_x = center[0] - rectangle_size[0] / 2  # Bottom-left corner x
        rect_y = center[1] - rectangle_size[1] / 2  # Bottom-left corner y
        rectangle = patches.Rectangle(
            (rect_x, rect_y),  # Bottom-left corner
            rectangle_size[0],  # Width
            rectangle_size[1],  # Height
            edgecolor="red",  # Rectangle edge color
            facecolor="none",  # Transparent fill
            linewidth=2,
        )
        ax.add_patch(rectangle)

    # Set equal scaling for x and y axes
    ax.set_aspect("equal")

    # Set equal resolution by matching axis limits
    x_min, x_max = min(x_data) - 0.1, max(x_data) + 0.1
    y_min, y_max = min(y_data) - 0.1, max(y_data) + 0.1
    max_range = max(x_max - x_min, y_max - y_min)

    # Set tick steps to 0.1
    x_ticks = np.arange(x_min, x_max, 0.1)
    y_ticks = np.arange(-0.1, 0.15, 0.1)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Add description below the plot
    # fig.text(
    #     0.5, 0.32, description, ha="center", va="center", fontsize=12
    # )

    # Set plot labels and legend
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('LIP CoM Trajectory : '+description)
    ax.legend()
    ax.grid(True)

    # Display the plot
    plt.show()

def CreateLoggers(builder, system, port_num:list[int], name:list[str]):
    out = []
    n = 0
    for i in port_num:
        logger = LogVectorOutput(system.get_output_port(i), builder)
        logger.set_name(name[n])
        n+=1
        out.append(logger)
    return out

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

def PlotRectangleAndPoint(rect, point,resolution = 0.01):
    """
    Plot the rectangle and the point.

    Parameters:
        rect: np.array([x_min, y_min, x_max, y_max])
        point: np.array([x, y])
    """
    x_min, y_min, x_max, y_max = rect
    px, py = point

    fig, ax = plt.subplots()

    # Create rectangle patch
    width = x_max - x_min
    height = y_max - y_min
    rectangle = patches.Rectangle((x_min, y_min), width, height,
                                  linewidth=2, edgecolor='blue', facecolor='none')
    ax.add_patch(rectangle)

    # Plot point
    color = 'go' if IsInsideRectangle(point, rect) else 'ro'
    ax.plot(px, py, color)

    # Adjust plot limits
    ax.set_xlim(min(x_min, px) - resolution, max(x_max, px) + resolution)
    ax.set_ylim(min(y_min, py) - resolution, max(y_max, py) + resolution)
    ax.set_aspect('equal')
    ax.grid(True)

    plt.title("Point Inside Rectangle" if IsInsideRectangle(point, rect) else "Point Outside Rectangle")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()     