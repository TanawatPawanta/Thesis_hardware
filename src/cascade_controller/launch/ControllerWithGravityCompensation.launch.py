from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    mit_controller_node = Node(
        package='cascade_controller',
        executable='MitControllerV2.py',
        output='screen'
    )

    gravity_compensation_node = Node(
        package='cascade_controller',
        executable='GravityCompensation.py',
        output='screen'
    )

    return LaunchDescription([mit_controller_node,
                              gravity_compensation_node])
