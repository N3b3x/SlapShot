from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import time
import os
import math


# User can select the robot model: "UR5" or "UR3"
ROBOT_MODEL = "UR3"  # Change to "UR3" if needed

def setup_scene():
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    
    print('Connected to CoppeliaSim:', sim)

    sim.stopSimulation()
    sim.startSimulation()
    time.sleep(1)
    
    # Create hockey table
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

    # Create hockey table
    table = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [table_length, table_width, table_height], 0)  # Use 0 for default options
    sim.setObjectPosition(table, [0, 0, z_height], sim.handle_parent)  # Set table position

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

    # Add red hockey puck
    #puck_z_position = table_top_z + puck_height / 2  # Position puck on top of the table
    puck_z_position = table_top_z #+ (puck_height / 2)  # Ensure the puck sits right on top of the table
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

    attach_paddle(sim, effector_left, [0, 0, 1], "LeftPaddle")  # Blue paddle for left arm
    attach_paddle(sim, effector_right, [0, 1, 0], "RightPaddle")  # Green paddle for right arm

    # Add a top-down camera
    camera = sim.createVisionSensor(
        0,  # options: 0 for default settings (no additional options enabled)
        [640, 480, 0, 0],  # intParams: sensor resolution 640x480, reserved params set to 0
        [0.1, 10, 60, 0.7, 0, 0, 0, 0, 0, 0, 0]  # floatParams: near clipping plane, far clipping plane, FOV, sensor size, and null pixel settings
    )
    sim.setObjectAlias(camera, "TopCamera")

    # Position the camera above the scene
    camera_position = [0, 0, 2]  # X, Y, Z position (2 meters above the center of the table)
    camera_orientation = [0, math.radians(-90), 0]  # Rotate -90° around X-axis to look down
    sim.setObjectPosition(camera, -1, camera_position)
    sim.setObjectOrientation(camera, camera_orientation, -1)

    return sim, {
        'camera': camera,
        'robot_left': robot_left,
        'robot_right': robot_right,
        'effector_left': effector_left,
        'effector_right': effector_right,
        'puck': puck
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
    sim.stopSimulation()

if __name__ == "__main__":
    main()
