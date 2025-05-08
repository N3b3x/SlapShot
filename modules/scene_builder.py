from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import time
import os
import math
import random

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
            # ============================================================
            # ============================================================
            #   BASIC HELPERS AND CONSTANTS
            # ============================================================
            # ============================================================
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

#---------------------------------------------
# Calculate inertia for a solid cylinder
#---------------------------------------------
def calculate_puck_inertia(mass, radius, height):
    """
    Calculate the inertia for a solid cylinder (puck).

    Args:
        mass (float): Mass of the puck.
        radius (float): Radius of the puck.
        height (float): Height of the puck.

    Returns:
        tuple: Inertia values (Ixx, Iyy, Izz).
    """
    inertia_z = 0.5 * mass * radius**2  # Rotational inertia around the puck's vertical axis
    inertia_x = inertia_y = (1/12) * mass * (3 * radius**2 + height**2)  # Rotational inertia around horizontal axes
    return inertia_x, inertia_y, inertia_z

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
            # ============================================================
            # ============================================================
            #   CONSTANTS AND PARAMETERS FOR THE AIR HOCKEY TABLE SCENE
            # ============================================================
            # ============================================================
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

#---------------------------------------------
# ROBOT MODEL
#---------------------------------------------
ROBOT_MODEL = "UR3"  # Robot model being used. 
                     # Can select the robot model: "UR5" or "UR3"

#---------------------------------------------
# TABLE DIMENSIONS AND PROPERTIES
#---------------------------------------------
# Define table dimensions and properties
table_length = 1.2  # Length of the air hockey table in meters
table_width = 0.7   # Width of the air hockey table in meters
table_height = 0.02 # Thickness of the table in meters
z_height = 0.01     # Height of the table's base from the ground in meters
table_top_z = z_height + table_height  # Height of the table's top surface from the ground

# Define table physical properties
table_friction = 0.0001       # Friction coefficient for the table surface
table_restitution = 0.8     # Coefficient of restitution (bounciness) for the table surface

# Define additional parameters
rail_thickness = 0.02                       # Thickness of the table's side rails in meters
rail_height = 0.08                          # Height of the table's side rails in meters
rail_z = table_height + (rail_height / 3)   # Z-position of the rail's bottom, slightly above the table's top

# Define table rails physical properties
table_rails_friction = 0.0001      # Friction coefficient for the table rails surface
table_rails_restitution = 0.2     # Coefficient of restitution (bounciness) for the table surface

# Define goal parameters
goal_width = 0.2  # Width of the goal opening in meters

#---------------------------------------------
# PUCK DIMENSIONS AND PROPERTIES
#---------------------------------------------
puck_radius = 0.075         # Radius of the puck in meters
puck_height = 0.02          # Height (thickness) of the puck in meters
puck_mass = 0.17            # Mass of the puck in kilograms

puck_inertia_x, puck_inertia_y, puck_inertia_z = calculate_puck_inertia(puck_mass, puck_radius, puck_height)

# Puck physical properties
puck_restitution = 0.9          # Coefficient of restitution (bounciness) for the puck
puck_friction = 0.0001          # Friction coefficient for the puck's interaction with the table
puck_rolling_friction = 0.0000  # Rolling friction coefficient for the puck
puck_linear_damping = 0.00      # Linear damping to reduce puck's velocity over time
puck_angular_damping = 0.00     # Angular damping to reduce puck's spin over time

#---------------------------------------------
# CAMERA PARAMETERS
#---------------------------------------------
camera_resolution = [640, 480]  # Resolution: width x height
camera_near_clipping = 0.1  # Near clipping plane
camera_far_clipping = 10  # Far clipping plane
camera_fov = 60  # Field of view in degrees
camera_sensor_size = 0.03  # Sensor size

camera_position = [0, 0, 0.5]  # X, Y, Z position (2 meters above the center of the table)
camera_orientation = [0, math.radians(180), 0]  # Rotate -90° around X-axis to look straight down
#---------------------------------------------
# ROBOT PLACEMENT PROPERTIES
#---------------------------------------------
robot_base_width = 0.15  # Width of the robot's base in meters
robot_z = table_top_z    # Z-position of the robot's base, aligned with the table's top surface
robot_offset = 0.2       # Distance of the robot's base from the goal in meters

paddle_friction = 0.001  # Friction coefficient for the paddle
paddle_restitution = 0.5  # Coefficient of restitution (bounciness) for the paddle

# Export variables for external access
__all__ = [
    "table_length", "table_width", "table_height", "z_height", "table_top_z",
    "table_friction", "table_restitution", "rail_thickness", "rail_height", "rail_z",
    "table_rails_friction", "table_rails_restitution", "goal_width",
    "puck_radius", "puck_height", "puck_mass", "puck_inertia_x", "puck_inertia_y", "puck_inertia_z",
    "puck_restitution", "puck_friction", "puck_rolling_friction", "puck_linear_damping", "puck_angular_damping",
    "robot_base_width", "robot_z", "robot_offset", "paddle_friction", "paddle_restitution",
    "camera_resolution", "camera_near_clipping", "camera_far_clipping", "camera_fov", "camera_sensor_size",
    "camera_position", "camera_orientation"
]

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
            # ============================================================
            # ============================================================
            #   SIM HELPERS AND PHYSICS PROPERTIES APPLIER
            # ============================================================
            # ============================================================
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

#---------------------------------------------
# Object Mass and Inertia Helpers
#---------------------------------------------
def set_mass_and_inertia(sim, shape, mass, inertia_diag):
    """
    Sets mass and simplified diagonal inertia for a given shape.

    Args:
        sim: Simulation object.
        shape (int): Handle to the shape.
        mass (float): Mass in kg.
        inertia_diag (list): Diagonal elements [Ixx, Iyy, Izz] of the inertia matrix.
    """
    sim.setShapeMass(shape, mass)

    # Convert to 3x3 inertia matrix in row-major order
    inertia_matrix = [
        inertia_diag[0], 0, 0,
        0, inertia_diag[1], 0,
        0, 0, inertia_diag[2]
    ]

    # Identity transformation matrix: COM at origin
    transformation_matrix = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0
    ]

    sim.setShapeInertia(shape, inertia_matrix, transformation_matrix)

#---------------------------------------------
# Physics Properties Applier
#---------------------------------------------
def apply_physics_properties(sim, handle, properties: dict):
    """
    Apply Bullet physics and common properties to a shape.

    Args:
        sim: Simulation object.
        handle: Shape handle.
        properties (dict): Keys are property names, values are bool/float.
    """
    handle_name = sim.getObjectAlias(handle, 2)  # Get the handle's name
    for k, v in properties.items():
        print(f"Setting property '{k}' to value '{v}' for handle {handle} (name: {handle_name})")
        if isinstance(v, bool):
            sim.setBoolProperty(handle, k, v)
        elif isinstance(v, float):
            sim.setFloatProperty(handle, k, v)
        elif isinstance(v, list):
            sim.setVector3Property(handle, k, v)

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
            # ============================================================
            # ============================================================
            #               CREATE AIR HOCKEY TABLE 
            #             (table, rails, puck, camera)
            # ============================================================
            # ============================================================
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

#---------------------------------------------
# Complete hockey table creator
#---------------------------------------------
def create_hockey_table(sim):
    """
    Creates a hockey table simulation environment.

    This function initializes a hockey table by first creating the table itself,
    adding the necessary rails, and then creating the puck. The table, rails, 
    and puck are added to the provided simulation object.

    Args:
        sim: The simulation object where the hockey table and its components
             will be created and added.

    Returns:
        A tuple containing the created table object, puck object, camera object, and goal sections.
    """
    table = create_table(sim)
    create_rails(sim, table)
    puck = create_puck(sim)
    camera = create_top_camera(sim, table)
    goal_sections = create_goal_sections(sim, table)
    return table, puck, camera, goal_sections

#---------------------------------------------
# Random puck positioning initializer
#---------------------------------------------
def initialize_puck_randomly(sim, puck):
    """
    Initializes the puck randomly in either the left or right player's alley, closer to their goals.

    Args:
        sim: The simulation object.
        puck: The handle of the puck object.
    """
    # Define the x-coordinate range for left and right alleys
    left_x_range = (-table_length / 2 + 0.1, -table_length / 2 + 0.3)
    right_x_range = (table_length / 2 - 0.3, table_length / 2 - 0.1)

    # Define the y-coordinate range for the puck's position
    y_range = (-goal_width / 2 + 0.05, goal_width / 2 - 0.05)

    # Randomly choose left or right alley
    if random.choice(["left", "right"]) == "left":
        x_pos = random.uniform(*left_x_range)
    else:
        x_pos = random.uniform(*right_x_range)

    # Randomly choose a y-coordinate within the range
    y_pos = random.uniform(*y_range)

    # Set the puck's position
    sim.setObjectPosition(puck, sim.handle_parent, [x_pos, y_pos, table_top_z + puck_height / 2])
    print(f"[INFO] Puck initialized at position: ({x_pos:.2f}, {y_pos:.2f}, {table_top_z + puck_height / 2:.2f})")

#---------------------------------------------
# Table Base Creator
#---------------------------------------------
def create_table(sim):
    """
    Creates the air hockey table as a cuboid and places it in the scene.

    Args:
        sim: The CoppeliaSim simulation object.

    Returns:
        int: The handle of the created table object.
    """
    table = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [table_length, table_width, table_height], 0)
    if table == -1:
        raise RuntimeError("[ERROR] Failed to create table.")
    sim.setObjectPosition(table, [0, 0, z_height], sim.handle_parent)
    sim.setObjectInt32Param(table, sim.shapeintparam_respondable, 1)
    sim.setObjectAlias(table, "AirHockeyTable")
    
    apply_physics_properties(sim, table, {
            'respondable': True,
            'dynamic': False,
            'dynamic': False,
            'applyShowEdges': True,
            'bullet.friction': table_friction,
            'bullet.frictionOld': table_friction,
            'bullet.stickyContact': False
        })
    
    return table

#---------------------------------------------
# Table Rails Creator
#---------------------------------------------
def create_rails(sim, table):
    """
    Creates all side and goal rails around the table.

    Args:
        sim: The simulation object.
        table: The handle of the hockey table.
    """
    def _add_rail(size, pos):
        rail = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size, 2)
        if rail == -1:
            raise RuntimeError(f"[ERROR] Failed to create rail at {pos}")
        sim.setShapeColor(rail, None, sim.colorcomponent_ambient_diffuse, [0, 0, 0])
        sim.setObjectParent(rail, table, True)
        sim.setObjectPosition(rail, pos, sim.handle_parent)

        apply_physics_properties(sim, rail, {
            'respondable': True,
            'dynamic': False,
            'bullet.friction': table_rails_friction,
            'bullet.frictionOld': table_rails_friction,
            'bullet.restitution': table_rails_restitution,  # High bounce off rail
            'bullet.stickyContact': False
        })

    # Side and goal rails
    half_goal_space = (table_width - goal_width) / 2
    y_pos = (goal_width + half_goal_space) / 2
    rail_specs = [
        # Side rails (top/bottom)
        ([table_length, rail_thickness, rail_height], [0,  table_width/2 - rail_thickness/2, rail_z]),
        ([table_length, rail_thickness, rail_height], [0, -table_width/2 + rail_thickness/2, rail_z]),
        # Goal side rails
        ([rail_thickness, half_goal_space, rail_height], [-table_length/2 + rail_thickness/2,  y_pos, rail_z]),
        ([rail_thickness, half_goal_space, rail_height], [-table_length/2 + rail_thickness/2, -y_pos, rail_z]),
        ([rail_thickness, half_goal_space, rail_height], [ table_length/2 - rail_thickness/2,  y_pos, rail_z]),
        ([rail_thickness, half_goal_space, rail_height], [ table_length/2 - rail_thickness/2, -y_pos, rail_z])
    ]

    for size, pos in rail_specs:
        _add_rail(size, pos)
        
#---------------------------------------------
# Air Hockey Puck Creator
#---------------------------------------------
def create_puck(sim):
    """
    Creates and returns a dynamic hockey puck with realistic physical properties.

    Args:
        sim: Simulation object.

    Returns:
        int: Handle to the puck object.
    """
    puck = sim.createPrimitiveShape(sim.primitiveshape_cylinder, [puck_radius, puck_radius, puck_height], 8)
    if puck == -1:
        raise RuntimeError("[ERROR] Failed to create puck.")
    
    # Set puck position above the table
    sim.setObjectPosition(puck, sim.handle_parent, [0, 0, table_top_z + puck_height / 2])
    sim.setShapeColor(puck, None, sim.colorcomponent_ambient_diffuse, [1, 0, 0])
    sim.setObjectAlias(puck, "HockeyPuck")
    
    # Set puck mass and inertia
    set_mass_and_inertia(sim, puck, puck_mass, [puck_inertia_x, puck_inertia_y, puck_inertia_z])

    # Apply physics properties
    apply_physics_properties(sim, puck, {
        'dynamic': True,
        'respondable': True,
        'bullet.friction': puck_friction,
        'bullet.frictionOld': puck_friction,
        'bullet.stickyContact': False,
        'bullet.restitution': puck_restitution,
        'bullet.linearDamping': puck_linear_damping,
        'bullet.angularDamping': puck_angular_damping,
        # 'bullet.customCollisionMarginEnabled': True,
        # 'bullet.customCollisionMarginValue': 0.001,
    })

    return puck

#------------------------------------------------
# Top Camera - board visualization - Creator
#------------------------------------------------
def create_top_camera(sim, table):
    """
    Creates and positions a top-down vision sensor above the hockey table.

    Args:
        sim: Simulation object.
        table: Handle to the hockey table.

    Returns:
        int: Handle to the created camera.
    """
    # Add a top-down perspective camera
    camera_handle = sim.createVisionSensor(
        2,  # options: bit 1 set for perspective mode
        [*camera_resolution, 0, 0],  # intParams: resolution, reserved params set to 0
        [camera_near_clipping, camera_far_clipping, camera_fov, camera_sensor_size, 0, 0, 0, 0, 0, 0, 0]  # floatParams: clipping, FOV, sensor size, and null pixel settings
    )
    sim.setObjectAlias(camera_handle, "TopCamera")

    # Position the camera above the scene
    sim.setObjectPosition(camera_handle, -1, camera_position)
    sim.setObjectOrientation(camera_handle, camera_orientation, -1)

    # Create a floating view for the camera
    floating_view = sim.floatingViewAdd(
        0.1, 0.1, 0.3, 0.3,  # Centered position and size
        0  # Default options (close button enabled, resizable, etc.)
    )

    # Link the floating view to the camera
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
        
    return camera_handle

#---------------------------------------------
# Goal Sections - with goal sensor - Creator
#---------------------------------------------
def create_goal_sections(sim, table):
    """
    Creates goal sections behind the goal openings with a base and yellow rail edges.
    Ensures the right goal's exterior aligns correctly with the interior.
    Uses ray-type proximity sensors to detect if a puck breaks the goal line.

    Args:
        sim: The simulation object.
        table: The handle of the hockey table.

    Returns:
        dict: A dictionary containing the handles of the goal sections and sensors:
            - left_goal_base
            - right_goal_base
            - left_goal_rails
            - right_goal_rails
            - left_goal_sensor
            - right_goal_sensor
    """
    goal_depth = 0.2  # Depth of the goal section
    goal_height = table_height + 0.05  # Height of the goal section
    goal_z = z_height + goal_height / 2  # Center of the goal section in the Z-axis
    goal_extension_factor = 0.4  # Extend only 40% of the way to the robots
    sensor_offset = 0.025  # Offset to position the sensors deeper into the goals

    def _create_goal_base(position, alias):
        # Create the goal base as a cuboid
        goal_base = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [goal_depth * goal_extension_factor, goal_width, table_height], 0)
        if goal_base == -1:
            raise RuntimeError(f"[ERROR] Failed to create goal base: {alias}")
        sim.setObjectPosition(goal_base, position, sim.handle_parent)
        sim.setObjectAlias(goal_base, alias)
        sim.setShapeColor(goal_base, None, sim.colorcomponent_ambient_diffuse, [0.5, 0.5, 0.5])  # Gray color
        sim.setObjectParent(goal_base, table, True)

        apply_physics_properties(sim, goal_base, {
            'respondable': True,
            'dynamic': False,
            'bullet.friction': 0.1,
            'bullet.frictionOld': 0.1,
            'bullet.stickyContact': True,
            'bullet.restitution': 0.6
        })
        return goal_base

    def _create_goal_rails(position, alias, side=0):
        """
        Create the goal rails as cuboids.

        Args:
            position (list): Position of the goal base.
            alias (str): Alias for the goal rails.
            side (int): 0 for left, 1 for right.

        Returns:
            list: Handles of the created goal rails.
        """
        rail_thickness = 0.02
        rail_height = 0.05
        offset = -goal_depth * goal_extension_factor / 2 if side == 0 else goal_depth * goal_extension_factor / 2
        rail_specs = [
            ([rail_thickness, goal_width, rail_height], [position[0] + offset, position[1], goal_z]),
            ([goal_depth * goal_extension_factor, rail_thickness, rail_height], [position[0], position[1] + goal_width / 2, goal_z]),
            ([goal_depth * goal_extension_factor, rail_thickness, rail_height], [position[0], position[1] - goal_width / 2, goal_z])
        ]

        rails = []
        for size, pos in rail_specs:
            rail = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size, 0)
            if rail == -1:
                raise RuntimeError(f"[ERROR] Failed to create goal rail at {pos}")
            sim.setObjectPosition(rail, pos, sim.handle_parent)
            sim.setShapeColor(rail, None, sim.colorcomponent_ambient_diffuse, [1, 1, 0])  # Yellow color
            sim.setObjectParent(rail, table, True)

            apply_physics_properties(sim, rail, {
                'respondable': True,
                'dynamic': False,
                'bullet.friction': table_rails_friction,
                'bullet.frictionOld': table_rails_friction,
                'bullet.restitution': table_rails_restitution,  # High bounce off rail
                'bullet.stickyContact': False
            })
            rails.append(rail)
        return rails

    def _create_goal_sensor(start_position, end_position, alias):
        # Define proximity sensor parameters for a ray-type sensor
        sensor_type = sim.proximitysensor_ray_subtype
        sub_type = 16  # Deprecated, set to 16
        options = 0  # Explicitly handled
        int_params = [0, 0, 0, 0, 0, 0, 0, 0]  # Not used for ray-type sensors
        float_params = [
            0.02,  # Offset
            np.linalg.norm(np.array(end_position) - np.array(start_position))-2*0.02,  # Range (distance between start and end)
            0.0,  # X size (not used for ray-type sensors)
            0.0,  # Y size (not used for ray-type sensors)
            0.0,  # X size far (not used for ray-type sensors)
            0.0,  # Y size far (not used for ray-type sensors)
            0.0,  # Inside gap
            0.0,  # Radius (not used for ray-type sensors)
            0.0,  # Radius far (not used for ray-type sensors)
            0.0,  # Angle (not used for ray-type sensors)
            0.0,  # Threshold angle for limited angle detection
            0.0,  # Smallest detection distance
            0.01,  # Sensing point size
            0.0,  # Reserved
            0.0   # Reserved
        ]

        # Create the proximity sensor
        sensor = sim.createProximitySensor(sensor_type, sub_type, options, int_params, float_params)
        if sensor == -1:
            raise RuntimeError(f"[ERROR] Failed to create proximity sensor: {alias}")
        sim.setObjectPosition(sensor, start_position, sim.handle_parent)
        sim.setObjectAlias(sensor, alias)
        sim.setObjectParent(sensor, table, True)

        # Orient the sensor to point from start_position to end_position using Euler angles
        direction = np.array(end_position) - np.array(start_position)
        direction = direction / np.linalg.norm(direction)  # Normalize the direction vector
        angle_x = -math.atan2(direction[1], direction[0])  # Pitch (horizontal orientation)
        angle_y = -math.atan2(direction[2], math.sqrt(direction[0]**2 + direction[1]**2))  # Roll
        angle_z = 0  # Yaw
        sim.setObjectOrientation(sensor, [angle_x, angle_y, angle_z], sim.handle_parent)

        return sensor

    # Create left goal base, rails, and sensor
    left_goal_base = _create_goal_base([-table_length / 2 - goal_depth * goal_extension_factor / 2, 0, z_height], "LeftGoalBase")
    left_goal_rails = _create_goal_rails([-table_length / 2 - goal_depth * goal_extension_factor / 2, 0, z_height], "LeftGoalRails", 0)
    left_goal_sensor = _create_goal_sensor(
        [-table_length / 2 - sensor_offset, -goal_width / 2, table_top_z + puck_height / 2],
        [-table_length / 2 - sensor_offset, goal_width / 2, table_top_z + puck_height / 2],
        "LeftGoalSensor"
    )

    # Create right goal base, rails, and sensor
    right_goal_base = _create_goal_base([table_length / 2 + goal_depth * goal_extension_factor / 2, 0, z_height], "RightGoalBase")
    right_goal_rails = _create_goal_rails([table_length / 2 + goal_depth * goal_extension_factor / 2, 0, z_height], "RightGoalRails", 1)
    right_goal_sensor = _create_goal_sensor(
        [table_length / 2 + sensor_offset, -goal_width / 2, table_top_z + puck_height / 2],
        [table_length / 2 + sensor_offset, goal_width / 2, table_top_z + puck_height / 2],
        "RightGoalSensor"
    )

    return {
        'left_goal_base': left_goal_base,
        'right_goal_base': right_goal_base,
        'left_goal_rails': left_goal_rails,
        'right_goal_rails': right_goal_rails,
        'left_goal_sensor': left_goal_sensor,
        'right_goal_sensor': right_goal_sensor
    }

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
            # ============================================================
            # ============================================================
            #           LOAD ROBOTS WITH PADDLES AND PLACE THEM
            #   (robot_left, robot_right, effector_left, effector_right)
            # ============================================================
            # ============================================================
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

#---------------------------------------------
# Create robots with paddels and place them
#---------------------------------------------
def load_and_place_robot_pair(sim, attach_paddles=True, disable_attached_scripts=True):
    """
    Loads and places two robot arms symmetrically on either side of the table, with their paddles
    positioned inside their respective goals and slightly above the board.

    Args:
        sim: CoppeliaSim simulation object.
        attach_paddles (bool): If True, paddles will be attached to both end-effectors.
        disable_scripts (bool): If True, removes child scripts from robots after loading.

    Returns:
        dict: Dictionary of handles:
            - robot_left (int)
            - robot_right (int)
            - effector_left (int)
            - effector_right (int)
    """
    def _load_robot_at(position, alias):
        robot_model = f"{ROBOT_MODEL}.ttm"
        robot_path = os.path.join(os.path.dirname(__file__), robot_model)
        if not os.path.isfile(robot_path):
            raise FileNotFoundError(f"[ERROR] Could not find {robot_model} at {robot_path}")
        
        handle = sim.loadModel(robot_path)
        if handle == -1:
            raise RuntimeError(f"[ERROR] Failed to load robot model: {robot_model}")
        
        if disable_attached_scripts:
            try:
                script_handle = sim.getScript(sim.scripttype_childscript, handle)
                sim.removeObjects([script_handle], False)
                print(f"[INFO] Removed script from {alias}")
            except Exception:
                print(f"[WARN] No script found on {alias} to remove.")
                
        sim.setObjectPosition(handle, -1, position)
        sim.setObjectAlias(handle, alias)

        return handle

    # Define left and right robot base positions
    left_robot_base_pos = [-table_length / 2 - robot_offset, 0, robot_z]
    right_robot_base_pos = [table_length / 2 + robot_offset, 0, robot_z]

    # Load both robots
    robot_left = _load_robot_at(left_robot_base_pos, "RobotLeft")
    robot_right = _load_robot_at(right_robot_base_pos, "RobotRight")

    # Get end effectors (assumes "link7_visible" as the end)
    eff_left = sim.getObject(f"{sim.getObjectAlias(robot_left, 2)}/link7_visible")
    eff_right = sim.getObject(f"{sim.getObjectAlias(robot_right, 2)}/link7_visible")

    if attach_paddles:
        # Attach paddles and position them inside respective goals
        attach_paddle(sim, eff_left, [0, 0, 1], "LeftPaddle")
        attach_paddle(sim, eff_right, [0, 1, 0], "RightPaddle")

    # Get end effectors (assumes "link7_visible" as the end)
    eff_left = sim.getObject(f"{sim.getObjectAlias(robot_left, 2)}/link7_visible/LeftPaddle")
    eff_right = sim.getObject(f"{sim.getObjectAlias(robot_right, 2)}/link7_visible/RightPaddle")
    
    return {
        'robot_left': robot_left,
        'robot_right': robot_right,
        'effector_left': eff_left,
        'effector_right': eff_right
    }

#---------------------------------------------
# Attach paddles to end-effectors
#---------------------------------------------
def attach_paddle(sim, effector_handle, color, name):
    """
    Creates and attaches a paddle to the robot's end-effector.

    Args:
        sim: Simulation object.
        effector_handle (int): Handle to the robot's end-effector.
        color (list): RGB color values for the paddle.
        name (str): Alias name for paddle.
    """
    paddle = sim.createPrimitiveShape(sim.primitiveshape_cylinder, [0.1, 0.1, 0.01], 8)
    sim.setShapeColor(paddle, None, sim.colorcomponent_ambient_diffuse, color)
    sim.setObjectParent(paddle, effector_handle, False)
    
    sim.setObjectPosition(paddle, [-0.0275, 0, 0], sim.handle_parent)  # Slight offset to lie flat
    sim.setObjectOrientation(paddle, [0, -math.radians(90), 0], sim.handle_parent)
    sim.setObjectAlias(paddle, name)
    
    apply_physics_properties(sim, paddle, {
        'dynamic': False,  # Paddles are typically static since they are controlled by the robot
        'respondable': True,
        'bullet.friction': paddle_friction,
        'bullet.frictionOld': paddle_friction,
        'bullet.stickyContact': False,
        'bullet.restitution': paddle_restitution,
    })

#---------------------------------------------
# Function to initialize robot joints
#---------------------------------------------
def initialize_robot_joints(sim, robot, target_positions_deg):
    """
    Sets the robot joint configuration to desired angles using trajectory motion.

    Args:
        sim: Simulation object.
        robot: Handle to robot.
        target_positions_deg (list): Target joint angles in degrees.
    """
    print(f"[DEBUG] Initializing robot joints for robot handle: {robot}")
    print(f"[DEBUG] Target joint positions (degrees): {target_positions_deg}")

    num_joints = 6
    robot_alias = sim.getObjectAlias(robot, 2)
    print(f"[DEBUG] Robot alias: {robot_alias}")

    joints = []
    for i in range(num_joints):
        joint_handle = sim.getObject(f'{robot_alias}/joint', {'index': i})
        if joint_handle == -1:
            print(f"[ERROR] Failed to get handle for joint {i} of robot {robot_alias}")
        else:
            print(f"[DEBUG] Joint {i} handle: {joint_handle}")
        joints.append(joint_handle)

    target_positions_rad = np.deg2rad(target_positions_deg).tolist()
    print(f"[DEBUG] Target joint positions (radians): {target_positions_rad}")

    try:
        sim.moveToConfig({
            'joints': joints,
            'targetPos': target_positions_rad,
            'maxVel': [np.deg2rad(180)] * num_joints,
            'maxAccel': [np.deg2rad(40)] * num_joints,
            'maxJerk': [np.deg2rad(80)] * num_joints
        })
        print("[DEBUG] moveToConfig executed successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to move robot joints: {e}")

#---------------------------------------------
# Functions to get robots positions
#---------------------------------------------
def get_robot_position(sim, robot_handle):
    """
    Get the base position of a robot dynamically from its handle.

    Args:
        sim: The simulation object.
        robot_handle: The handle of the robot.

    Returns:
        list: A list containing the x, y, z position of the robot's base.
    """
    if robot_handle == -1:
        raise ValueError("[ERROR] Invalid robot handle provided.")
    return sim.getObjectPosition(robot_handle, sim.handle_world)

def get_robot_left_position(sim, handles):
    """
    Get the base position of the left robot dynamically.

    Args:
        sim: The simulation object.
        handles: Dictionary of simulation handles.

    Returns:
        list: A list containing the x, y, z position of the left robot's base.
    """
    return get_robot_position(sim, handles['robot_left'])

def get_robot_right_position(sim, handles):
    """
    Get the base position of the right robot dynamically.

    Args:
        sim: The simulation object.
        handles: Dictionary of simulation handles.

    Returns:
        list: A list containing the x, y, z position of the right robot's base.
    """
    return get_robot_position(sim, handles['robot_right'])

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
            # ==================================================================
            # ==================================================================
            #           SETUP simIK ENVIRONMENT FOR ROBOT ARMS
            #   (simIK, robot_left, robot_right, effector_left, effector_right)
            # ==================================================================
            # ==================================================================
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

#----------------------------------------------------------
# SETS UP INVERSE KINEMATICS ENVIRONMENT FOR ROBOT ARMS
#----------------------------------------------------------
def setup_simIK_environment(sim, simIK, robot_handles):
    """
    Sets up a simIK environment for the 6DOF UR3 robot arms and positions their paddles
    in the middle of their respective goals.

    Args:
        sim: The simulation object.
        simIK: The simIK module.
        robot_handles (dict): Dictionary containing robot and effector handles.

    Returns:
        dict: A dictionary containing IK environment, groups, joint handles, and dummies.
    """
    # Initialize simIK environment
    ik_environment = simIK.createEnvironment()

    def configure_robot_ik(sim, simIK, ik_environment, robot_handle, effector_handle, target_position, alias):
        """
        Configures simIK for a 6-DOF arm, ensuring the end-effector is always flat to the table.

        Args:
            sim: The simulation API.
            simIK: The simIK API.
            ik_environment: IK environment handle.
            robot_handle: Handle to the robot base.
            effector_handle: Handle to the end-effector (link7_visible).
            target_position: Target world position [x, y, z].
            alias: String alias for naming target dummy.

        Returns:
            Tuple (ik_group, joint_handles, dummy_handle)
        """
        # Create IK group
        ik_group = simIK.createGroup(ik_environment)
        simIK.setGroupCalculation(
            ik_environment, ik_group,
            simIK.method_damped_least_squares, 0.03, 100
        )
        
        def create_downward_facing_dummy(sim, name, size=0.01):
            """
            Creates a dummy facing downward in -Z and returns its handle.

            Args:
            sim: The simulation object.
            name: Alias name to give the dummy.
            size: Size of the dummy.

            Returns:
            int: Handle to the dummy.
            """
            dummy = sim.createDummy(size)
            sim.setObjectAlias(dummy, name)

            # Orientation: facing -Z (identity with 180° rotation around X)
            downward_orientation = [0, -math.pi, 0]  # Euler XYZ (180° about X axis)
            sim.setObjectOrientation(dummy, downward_orientation, sim.handle_world)

            return dummy

        # Create downward-facing dummy
        target_dummy = create_downward_facing_dummy(sim, f"{alias}_IK_Target", size=0.01)
        sim.setObjectPosition(target_dummy, sim.handle_world, target_position)

        # Get joint handles
        joint_handles = []
        for i in range(6):
            joint_handle = sim.getObject(f"{sim.getObjectAlias(robot_handle, 2)}/joint", {'index': i})
            joint_handles.append(joint_handle)

        # Add IK element
        simIK.addElementFromScene(
            ik_environment,
            ik_group,
            robot_handle,       # base
            effector_handle,    # tip
            target_dummy,       # target
            simIK.constraint_pose
        )

        # Solve and sync to sim
        apply_ik_to_sim(simIK, ik_environment, ik_group)

        print(f"[INFO] {alias} IK configured. Target: {target_position}")
        return ik_group, joint_handles, target_dummy

    # Paddle default start positions
    left_paddle_position = [-table_length / 2 + 0.03, 0, table_top_z + puck_height*1/4]
    right_paddle_position = [table_length / 2 - 0.03, 0, table_top_z + puck_height*1/4]

    # Configure both robots
    left_ik_group, left_joints, left_dummy = configure_robot_ik(
        sim, simIK, ik_environment,
        robot_handles['robot_left'],
        robot_handles['effector_left'],
        left_paddle_position,
        "LeftRobot"
    )

    right_ik_group, right_joints, right_dummy = configure_robot_ik(
        sim, simIK, ik_environment,
        robot_handles['robot_right'],
        robot_handles['effector_right'],
        right_paddle_position,
        "RightRobot"
    )

    return {
        'ik_environment': ik_environment,
        'left_ik_group': left_ik_group,
        'right_ik_group': right_ik_group,
        'left_joints': left_joints,
        'right_joints': right_joints,
        'left_target_dummy': left_dummy,
        'right_target_dummy': right_dummy
    }


#---------------------------------------------
# Apply IK to the robot arms
#---------------------------------------------
def apply_ik_to_sim(simIK, ik_env, ik_group):
    """
    Applies inverse kinematics by syncing from simulation, handling the IK group, and syncing back to simulation.

    Args:
        simIK: The simIK module.
        ik_env: The IK environment handle.
        ik_group: The IK group handle.

    Returns:
        int: The result of the IK group handling.
    """
    simIK.syncFromSim(ik_env, [ik_group])
    result = simIK.handleGroup(ik_env, ik_group)
    simIK.syncToSim(ik_env, [ik_group])
    return result

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
            # ============================================================
            # ============================================================
            #     INVERSE KINEMATICS BASED POSITIONING FOR ROBOT ARMS
            # ============================================================
            # ============================================================
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

#---------------------------------------------
# Move the end-effector to a target position
#---------------------------------------------
def move_effector_to(sim, simIK, ik_environment, ik_group, dummy_target, target_position):
    """
    Moves the robot's end-effector to a target position using IK,
    while keeping the orientation facing downward.

    Args:
        sim: The sim module.
        simIK: The simIK module.
        ik_environment: The IK environment handle.
        ik_group: The IK group handle.
        dummy_target: The dummy handle used as the target.
        target_position: The 3D [x, y, z] world position to move to.
    """
    # Set dummy position (orientation already points downward)
    sim.setObjectPosition(dummy_target, sim.handle_world, target_position)

    # Perform IK and sync joints
    apply_ik_to_sim(simIK, ik_environment, ik_group)


def move_effector_to_2(ik_environment, ik_group, dummy_target, target_position):
    """
    Moves the robot's end-effector to a target position using IK,
    while keeping the orientation facing downward.

    Args:
        sim: The sim module.
        simIK: The simIK module.
        ik_environment: The IK environment handle.
        ik_group: The IK group handle.
        dummy_target: The dummy handle used as the target.
        target_position: The 3D [x, y, z] world position to move to.
    """
    client = RemoteAPIClient()
    sim = client.require('sim')
    simIK = client.require('simIK')
    move_effector_to(sim, simIK, ik_environment, ik_group, dummy_target, target_position)

#---------------------------------------------
# Drag the paddle along a path
#---------------------------------------------
def drag_paddle_along_path(sim, simIK, ik_env, ik_group, target_dummy, waypoints):
    """
    Moves the paddle through a series of waypoints on the table using simIK,
    ensuring the end-effector stays flat (downward-facing) throughout.

    Args:
        sim: CoppeliaSim remote API object
        simIK: simIK module
        ik_env: simIK environment handle
        ik_group: simIK group handle
        target_dummy: The target dummy used in IK
        waypoints: A list of [x, y, z] positions
    """
    for point in waypoints:
        sim.setObjectPosition(target_dummy, sim.handle_world, point)
        apply_ik_to_sim(simIK, ik_env, ik_group)
        time.sleep(0.05)  # Let sim settle (adjust duration for your frame rate)

#---------------------------------------------
# Generate a straight path between two points
#---------------------------------------------
def generate_straight_path(start, end, num_points):
    """
    Generates a straight line path between two points.

    Args:
        start: Starting [x, y, z] position.
        end: Ending [x, y, z] position.
        num_points: Number of points along the path.

    Returns:
        list: A list of [x, y, z] positions along the path.
    """
    return [[np.linspace(start[i], end[i], num_points).tolist() for i in range(3)]]

#---------------------------------------------
# Move dummies randomly
#---------------------------------------------
def move_dummies_randomly(sim, simIK, ik_handles):
    left_bounds = {
        'x': (-table_length/2 + 0.05, -0.35),  # Adjusted bounds to prevent overlap with table edges
        'y': (-table_width/2 + 0.1, table_width/2 - 0.1)
    }

    right_bounds = {
        'x': (0.35, table_length/2 - 0.05),  # Adjusted bounds to prevent overlap with table edges
        'y': (-table_width/2 + 0.1, table_width/2 - 0.1)
    }

    z = table_top_z + puck_height * 0.5  # Adjusted Z to ensure dummies stay above the table

    print("[TEST] Moving targets randomly... Press Ctrl+C to stop.")
    try:
        while True:
            left_x = random.uniform(*left_bounds['x'])
            left_y = random.uniform(*left_bounds['y'])

            right_x = random.uniform(*right_bounds['x'])
            right_y = random.uniform(*right_bounds['y'])

            move_effector_to(sim, simIK, ik_handles['ik_environment'], ik_handles['left_ik_group'], ik_handles['left_dummy'], [left_x, left_y, z])
            move_effector_to(sim, simIK, ik_handles['ik_environment'], ik_handles['right_ik_group'], ik_handles['right_dummy'], [right_x, right_y, z])

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[TEST] Movement loop interrupted.")

#---------------------------------------------
# Start real-time IK tracking
#---------------------------------------------
def start_realtime_ik_tracking(sim, simIK, ik_env, left_ik_group, right_ik_group, left_target_dummy, right_target_dummy, left_bounds, right_bounds, axis='x', frequency=0.5, amplitude=0.1):
    """
    Continuously moves the IK target dummies for both arms back and forth smoothly in one direction,
    and updates the IK groups in real time to track the movement.

    Args:
        sim: CoppeliaSim remote API object.
        simIK: simIK module.
        ik_env: IK environment handle.
        left_ik_group: IK group handle for the left arm.
        right_ik_group: IK group handle for the right arm.
        left_target_dummy: Target dummy for the left arm.
        right_target_dummy: Target dummy for the right arm.
        left_bounds: Dict with 'x' and 'y' tuple bounds for the left arm.
        right_bounds: Dict with 'x' and 'y' tuple bounds for the right arm.
        axis: Which axis to animate (default: 'x').
        frequency: Oscillation frequency in Hz.
        amplitude: How far to move from center in meters.
    """
    left_center_x = sum(left_bounds['x']) / 2
    left_center_y = sum(left_bounds['y']) / 2
    right_center_x = sum(right_bounds['x']) / 2
    right_center_y = sum(right_bounds['y']) / 2
    z = table_top_z + puck_height * 0.25

    start_time = sim.getSimulationTime()

    print("[IK TRACKER] Real-time IK tracking started for both arms...")

    try:
        while True:
            current_time = sim.getSimulationTime() - start_time
            phase = 2 * math.pi * frequency * current_time
            delta = amplitude * math.sin(phase)

            # Update left arm target position
            if axis == 'x':
                left_pos = [left_center_x + delta, left_center_y, z]
            else:
                left_pos = [left_center_x, left_center_y + delta, z]
            sim.setObjectPosition(left_target_dummy, sim.handle_world, left_pos)

            # Update right arm target position
            if axis == 'x':
                right_pos = [right_center_x + delta, right_center_y, z]
            else:
                right_pos = [right_center_x, right_center_y + delta, z]
            sim.setObjectPosition(right_target_dummy, sim.handle_world, right_pos)

            # Handle IK groups for both arms
            left_result = apply_ik_to_sim(simIK, ik_env, left_ik_group)
            right_result = apply_ik_to_sim(simIK, ik_env, right_ik_group)

            # Debugging logs
            #print(f"[DEBUG] Left target position: {left_pos}, IK result: {left_result}")
            #print(f"[DEBUG] Right target position: {right_pos}, IK result: {right_result}")

            # Let the sim breathe a bit, but fast enough (~60 Hz)
            time.sleep(0.016)
    except KeyboardInterrupt:
        print("[IK TRACKER] Stopped.")
        return
        
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
            # ============================================================
            # ============================================================
            #           FULL SCENE SETUP FOR SIMULATION
            #   (table, puck, camera, goal_sections, simIK environment)
            # ============================================================
            # ============================================================
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

#---------------------------------------------
# Setup the complete simulation scene
#---------------------------------------------
def setup_scene():
    """
    Sets up the complete simulation scene, including the table, puck, robots, and simIK environment.

    Returns:
        tuple: The sim object and dictionary of handles.
    """
    client = RemoteAPIClient()
    sim = client.require('sim')
    simIK = client.require('simIK')
    print('[setup_scene] [INFO] Connected to CoppeliaSim')

    sim.stopSimulation()
    print('[setup_scene] [INFO] Stopped any running simulation.')
    sim.startSimulation()
    print('[setup_scene] [INFO] Started simulation.')
    time.sleep(1)
    
    # Load and place the robot pair with paddles attached
    print('[setup_scene] [INFO] Loading and placing robot pair...')
    robot_handles = load_and_place_robot_pair(sim, attach_paddles=True, disable_attached_scripts=True)
    print(f'[setup_scene] [DEBUG] Robot handles: {robot_handles}')
    
    # Wait for a moment to ensure the table and components are fully initialized
    print('[setup_scene] [INFO] Waiting for components to initialize...')
    time.sleep(1)
    
    # Create the hockey table, puck, camera, and goal sections
    print('[setup_scene] [INFO] Creating hockey table, puck, camera, and goal sections...')
    table, puck, camera, goal_sections = create_hockey_table(sim)
    print(f'[setup_scene] [DEBUG] Table handle: {table}, Puck handle: {puck}, Camera handle: {camera}')
    print(f'[setup_scene] [DEBUG] Goal sections: {goal_sections}')

    # Initialize the puck randomly in one of the player's alleys
    print('[setup_scene] [INFO] Initializing puck randomly...')
    initialize_puck_randomly(sim, puck)

    # Initialize robot joints to default positions
    print('[setup_scene] [INFO] Initializing robot joints to default positions...')
    initialize_robot_joints(sim, robot_handles['robot_left'], [-90, 60, 45, -15, -90, 180])
    print('[setup_scene] [INFO] Left robot joints initialized.')
    initialize_robot_joints(sim, robot_handles['robot_right'], [90, 60, 45, -15, -90, 90])
    print('[setup_scene] [INFO] Right robot joints initialized.')
    
    # Introduce a delay to ensure all objects are properly initialized
    print('[setup_scene] [INFO] Waiting for objects to settle...')
    time.sleep(2)

    # Setup simIK environment for the robots
    print('[setup_scene] [INFO] Setting up simIK environment...')
    ik_handles = setup_simIK_environment(sim, simIK, robot_handles)
    print(f'[setup_scene] [DEBUG] IK handles: {ik_handles}')

    handles = {
        'table': table,
        'camera': camera,
        'effector_left': robot_handles['effector_left'],
        'effector_right': robot_handles['effector_right'],
        'robot_left': robot_handles['robot_left'],
        'robot_right': robot_handles['robot_right'],
        'puck': puck,
        'left_goal_base': goal_sections['left_goal_base'],
        'right_goal_base': goal_sections['right_goal_base'],
        'left_goal_rails': goal_sections['left_goal_rails'],
        'right_goal_rails': goal_sections['right_goal_rails'],
        'left_goal_sensor': goal_sections['left_goal_sensor'],
        'right_goal_sensor': goal_sections['right_goal_sensor'],
        'ik_environment': ik_handles['ik_environment'],
        'left_ik_group': ik_handles['left_ik_group'],
        'right_ik_group': ik_handles['right_ik_group'],
        'left_joints': ik_handles['left_joints'],
        'right_joints': ik_handles['right_joints'],
        'left_target_dummy': ik_handles['left_target_dummy'],
        'right_target_dummy': ik_handles['right_target_dummy'],
    }

    print("[TEST] Scene setup complete.")
    print("Table handle:", handles['table'])
    print("Camera handle:", handles['camera'])
    print("Effector (left) handle:", handles['effector_left'])
    print("Effector (right) handle:", handles['effector_right'])
    print("Robot left:", handles['robot_left'])
    print("Robot right:", handles['robot_right'])
    print("Puck handle:", handles['puck'])
    print("Left goal base handle:", handles['left_goal_base'])
    print("Right goal base handle:", handles['right_goal_base'])
    print("Left goal rails handles:", handles['left_goal_rails'])
    print("Right goal rails handles:", handles['right_goal_rails'])
    print("Left goal sensor handle:", handles['left_goal_sensor'])
    print("Right goal sensor handle:", handles['right_goal_sensor'])

    return sim, simIK, handles

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
            # ============================================================
            # ============================================================
            #          TESTING THE SCENE BUILDING & SIMULATION
            # ============================================================
            # ============================================================
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

#---------------------------------------------
# Main function to setup simulation and run
# just a back and forth movement
#---------------------------------------------
def main(sim_file=None):
    sim, simIK, handles = setup_scene()

    # Define bounds for left and right robots
    left_bounds = {
        'x': (-table_length / 2 + 0.05, -0.05),
        'y': (-table_width / 2 + 0.05, table_width / 2 - 0.05)
    }
    right_bounds = {
        'x': (0.05, table_length / 2 - 0.05),
        'y': (-table_width / 2 + 0.05, table_width / 2 - 0.05)
    }

    # Prompt user to select test type
    print("Select test type:")
    print("1. Real-time IK tracking (back and forth movement)")
    print("2. Random dummy movement")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        start_realtime_ik_tracking(
            sim,
            simIK,
            handles['ik_environment'],
            handles['left_ik_group'],
            handles['right_ik_group'],
            handles['left_target_dummy'],
            handles['right_target_dummy'],
            left_bounds=left_bounds,
            right_bounds=right_bounds,
            axis='y',         # Move back and forth in Y
            frequency=0.5,    # 0.5 Hz (1 cycle every 2s)
            amplitude=0.15    # ±0.15 m movement
        )
    elif choice == "2":
        move_dummies_randomly(sim, simIK, {
            'ik_environment': handles['ik_environment'],
            'left_ik_group': handles['left_ik_group'],
            'right_ik_group': handles['right_ik_group'],
            'left_dummy': handles['left_target_dummy'],
            'right_dummy': handles['right_target_dummy'],
        })
    else:
        print("Invalid choice. Exiting.")

    if sim_file:
        print(f'Saving sim to {sim_file}')
        sim.saveScene(sim_file)
    sim.stopSimulation()

if __name__ == "__main__":
    sim_file = os.path.join(os.path.dirname(__file__), 'air_hockey.ttt')  # Example default file path
    main(sim_file=sim_file)

