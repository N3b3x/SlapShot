from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import time
import os
import math

# User can select the robot model: "UR5" or "UR3"
ROBOT_MODEL = "UR3"  # Change to "UR3" if needed


# Define table dimensions and z height as variables
table_length = 1.2
table_width = 0.7
table_height = 0.02
z_height = 0.01  # Table's base height from the ground
table_top_z = z_height + table_height  # Top of the table

# Define additional parameters
rail_thickness = 0.02
rail_height = 0.05
rail_z = table_height + (rail_height / 3)  # Place the bottom of the rail at the table's top surface
puck_radius = 0.075
puck_height = 0.02
puck_mass = 0.17
goal_width = 0.2  # Width of the goal opening
robot_base_width = 0.15  # Physical width of the robot base

def setup_scene():
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    
    print('Connected to CoppeliaSim:', sim)

    sim.stopSimulation()
    sim.startSimulation()
    time.sleep(1)
    
    # Create hockey table
    table = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [table_length, table_width, table_height], 0)  # Use 0 for default options
    sim.setObjectPosition(table, [0, 0, z_height], sim.handle_parent)  # Set table position
    sim.setObjectInt32Param(table, sim.shapeintparam_respondable, 1)

    # Side rails (top/bottom) - flush with table edge
    rail_data = [
        ([table_length, rail_thickness, rail_height], [0, table_width / 2 - rail_thickness / 2, rail_z]),
        ([table_length, rail_thickness, rail_height], [0, -table_width / 2 + rail_thickness / 2, rail_z]),

        # Left goal side rails
        ([rail_thickness, (table_width - goal_width) / 2, rail_height],
        [-table_length / 2 + rail_thickness / 2, (goal_width + (table_width - goal_width) / 2) / 2, rail_z]),
        ([rail_thickness, (table_width - goal_width) / 2, rail_height],
        [-table_length / 2 + rail_thickness / 2, -(goal_width + (table_width - goal_width) / 2) / 2, rail_z]),

        # Right goal side rails
        ([rail_thickness, (table_width - goal_width) / 2, rail_height],
        [table_length / 2 - rail_thickness / 2, (goal_width + (table_width - goal_width) / 2) / 2, rail_z]),
        ([rail_thickness, (table_width - goal_width) / 2, rail_height],
        [table_length / 2 - rail_thickness / 2, -(goal_width + (table_width - goal_width) / 2) / 2, rail_z])
    ]
    for size, pos in rail_data:
        rail = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size, 2)  # Enable sharp edges
        if rail == -1:  # Check if the rail creation failed
            raise RuntimeError(f"[ERROR] Failed to create rail with size {size} at position {pos}.")
        sim.setShapeColor(rail, None, sim.colorcomponent_ambient_diffuse, [0, 0, 0])  # Black color
        sim.setObjectParent(rail, table, True)  # Parent rail to table
        sim.setObjectPosition(rail, pos, sim.handle_parent)
        # make the rails respondable but keep them static
        sim.setObjectInt32Param(rail, sim.shapeintparam_respondable, 1)

    # Add red hockey puck
    #puck_z_position = table_top_z + puck_height / 2  # Position puck on top of the table
    puck_z_position = table_top_z + 1.0*(puck_height)  # Ensure the puck sits right on top of the table
    puck = sim.createPrimitiveShape(sim.primitiveshape_cylinder, [puck_radius, puck_radius, puck_height], 8)  # Correct puck size
    sim.setObjectPosition(puck, [0, 0, puck_z_position], sim.handle_parent)  # Puck position
    sim.setShapeColor(puck, None, sim.colorcomponent_ambient_diffuse, [1, 0, 0])  # Red color
    sim.setObjectAlias(puck, "HockeyPuck")

    # Allow using the selected robot model
    robot_model = f"{ROBOT_MODEL}.ttm"
    robot_path = os.path.join(os.path.dirname(__file__), robot_model)

    # Load left robot
    robot_left = sim.loadModel(robot_path)
    if robot_left == -1:
        raise RuntimeError(f"[ERROR] Failed to load left {ROBOT_MODEL} robot model.")
    sim.setObjectPosition(robot_left, -1, [-0.6, 0, z_height + table_height])  # Left arm position on top of the table

    # Load right robot
    robot_right = sim.loadModel(robot_path)
    if robot_right == -1:
        raise RuntimeError(f"[ERROR] Failed to load right {ROBOT_MODEL} robot model.")
    sim.setObjectPosition(robot_right, -1, [0.6, 0, z_height + table_height])  # Right arm position on top of the table

    if robot_left == -1 or robot_right == -1:
        raise RuntimeError("[ERROR] Failed to load robot models.")

    # Adjust robot base positions to be on the table behind the goals
    robot_z = table_top_z  # On table
    robot_offset = 0.2  # Distance behind the goal
    robot_left_pos = [-table_length / 2 - robot_offset, 0, robot_z]
    robot_right_pos = [table_length / 2 + robot_offset, 0, robot_z]

    sim.setObjectPosition(robot_left, -1, robot_left_pos)
    sim.setObjectPosition(robot_right, -1, robot_right_pos)

    # Retrieve effector handles dynamically
    def get_effector_handle(sim, robot_handle, effector_name):
        # Retrieve all objects in the robot's hierarchy tree
        objects = sim.getObjectsInTree(robot_handle, sim.handle_all, 0)
        for obj in objects:
            # Check if the alias matches the effector name
            if effector_name == sim.getObjectAlias(obj):
                return obj
        raise RuntimeError(f"[ERROR] Failed to find effector '{effector_name}' for robot.")

    effector_left = get_effector_handle(sim, robot_left, "link7_visible")
    effector_right = get_effector_handle(sim, robot_right, "link7_visible") 

    # Attach paddles to the robot arms
    def attach_paddle(sim, effector_handle, color, name):
        # Create a flat cylindrical paddle (disc shape)
        paddle = sim.createPrimitiveShape(sim.primitiveshape_cylinder, [0.1, 0.1, 0.01], 8)  # [diameter_x, diameter_y, height]
        
        # Set color
        sim.setShapeColor(paddle, None, sim.colorcomponent_ambient_diffuse, color)
        
        # Parent to effector (without keeping world pose, so we can set local position/orientation)
        keepInPlace = False
        sim.setObjectParent(paddle, effector_handle, keepInPlace)
        
        # Position relative to effector: raise slightly above the effector
        sim.setObjectPosition(paddle, [-0.0275, 0, 0] ,sim.handle_parent)  # Local Z offset
        
        # Orientation: rotate 90° around Y so paddle lies flat in XZ plane
        sim.setObjectOrientation(paddle, [0, math.radians(90), 0], sim.handle_parent)
        # Set readable alias
        sim.setObjectAlias(paddle, name)
        sim.setObjectInt32Param(puck, sim.shapeintparam_respondable, 1)

    attach_paddle(sim, effector_left, [0, 0, 1], "LeftPaddle")  # Blue paddle for left arm
    attach_paddle(sim, effector_right, [0, 1, 0], "RightPaddle")  # Green paddle for right arm


    sim.setObjectInt32Param(puck, sim.shapeintparam_static, 0)
    sim.setObjectInt32Param(puck, sim.shapeintparam_respondable, 1)

    robot_left_alias = sim.getObjectAlias(robot_left, 2)
    robot_right_alias = sim.getObjectAlias(robot_right, 2)

    max_ang_vel = np.deg2rad(180)
    max_ang_accel = np.deg2rad(40)
    max_ang_jerk = np.deg2rad(80)
    
    #initialize joint positions
    num_joints = 6
    sim.moveToConfig({
        'joints': [sim.getObject(f'{robot_left_alias}/joint', {'index': idx}) 
                   for idx in range(num_joints)],
        'maxVel': num_joints * [max_ang_vel],
        'maxAccel': num_joints * [max_ang_accel],
        'maxJerk': num_joints * [max_ang_jerk],
        'targetPos': np.deg2rad([-90, 60, 45, -15, -90, 90]).tolist()
    })

    sim.moveToConfig({
        'joints': [sim.getObject(f'{robot_right_alias}/joint', {'index': idx}) 
                   for idx in range(num_joints)],
        'maxVel': num_joints * [max_ang_vel],
        'maxAccel': num_joints * [max_ang_accel],
        'maxJerk': num_joints * [max_ang_jerk],
        'targetPos': np.deg2rad([90, 60, 45, -15, -90, 90]).tolist()
    })

    # Add a top-down perspective camera
    camera_handle = sim.createVisionSensor(
        2,  # options: bit 1 set for perspective mode
        [640, 480, 0, 0],  # intParams: high resolution (1280x720), reserved params set to 0
        [0.1, 10, 60, 0.03, 0, 0, 0, 0, 0, 0, 0]  # floatParams: near clipping, far clipping, FOV, sensor size, and null pixel settings
    )
    sim.setObjectAlias(camera_handle, "TopCamera")

    # Position the camera above the scene
    camera_position = [0, 0, 0.5]  # X, Y, Z position (2 meters above the center of the table)
    camera_orientation = [0, math.radians(-180), 0]  # Rotate -90° around X-axis to look straight down
    sim.setObjectPosition(camera_handle, -1, camera_position)
    sim.setObjectOrientation(camera_handle, camera_orientation, -1)

    # Create a floating view for the camera
    floating_view = sim.floatingViewAdd(
        0.1, 0.1, 0.3, 0.3,  # Centered position and size
        0  # Default options (close button enabled, resizable, etc.)
    )

    # Link the floating view to the camera
    # Attach the camera to the floating view
    res = sim.adjustView(
        floating_view,  # Handle of the floating view
        camera_handle,  # Handle of the camera
        64,  # Option to remove the floating view at simulation end
        "Top-Down Camera View"  # Label for the floating view
    )

    # Check if adjustment was successful
    if res > 0:
        print("View successfully adjusted.")
    else:
        print("Failed to adjust the view.")

    return sim, {
        'camera': camera_handle,
        'robot_left': robot_left,
        'robot_right': robot_right,
        'effector_left': effector_left,
        'effector_right': effector_right,
        'puck': puck
    }
    
def get_robot_base_positions(table_length, table_width, robot_offset):
    """
    Calculate the base positions of the robots relative to the center of the table.

    Args:
        table_length (float): Length of the table.
        table_width (float): Width of the table.
        robot_offset (float): Distance behind the goal where the robot base is positioned.

    Returns:
        dict: A dictionary containing the positions of the left and right robot bases.
    """
    # Left robot base position (behind the left goal)
    robot_left_pos = [-table_length / 2 - robot_offset, 0]  # (x, y)

    # Right robot base position (behind the right goal)
    robot_right_pos = [table_length / 2 + robot_offset, 0]  # (x, y)

    return {
        'robot_left': robot_left_pos,
        'robot_right': robot_right_pos
    }

def main():
    sim, handles = setup_scene()

    print("[TEST] Scene setup complete.")
    print("Camera handle:", handles['camera'])
    print("Effector (left) handle:", handles['effector_left'])
    print("Effector (right) handle:", handles['effector_right'])
    print("Robot left:", handles['robot_left'])
    print("Robot right:", handles['robot_right'])
    print("Puck handle:", handles['puck'])

    input("[TEST] Press Enter to stop simulation...")
    
    # sim_file = os.path.join(os.path.dirname(__file__), 'air_hockey.ttt')
    # print(f'Saving sim to {sim_file}')
    # sim.saveScene(sim_file)
    sim.stopSimulation()

if __name__ == "__main__":
    main()
