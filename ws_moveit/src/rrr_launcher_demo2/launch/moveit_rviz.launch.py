from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_moveit_rviz_launch


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("rrr_arm", package_name="rrr_launcher_demo2").to_moveit_configs()
    print('xxxxxxx' , moveit_config)
    return generate_moveit_rviz_launch(moveit_config)
