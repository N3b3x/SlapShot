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

# Define table physical properties
table_friction = 0.01  # Friction coefficient for the table
table_restitution = 0.6  # Coefficient of restitution for the table

# Define additional parameters
rail_thickness = 0.02
rail_height = 0.05
rail_z = table_height + (rail_height / 3)  # Place the bottom of the rail at the table's top surface

# Define puck parameters
puck_radius = 0.075
puck_height = 0.02
puck_mass = 0.17

# Calculate inertia for a solid cylinder
puck_inertia_z = 0.5 * puck_mass * puck_radius**2  # Inertia around the cylinder's axis
puck_inertia_x = puck_inertia_y = (1/12) * puck_mass * (3 * puck_radius**2 + puck_height**2)  # Inertia around x and y axes

# Puck physical properties
puck_restitution = 0.8   # Coefficient of restitution for the puck
puck_friction = 0.02      # Friction coefficient
puck_rolling_friction = 0.0005   # Static friction coefficient
puck_linear_damping = 0.005  # Linear damping for the puck
puck_angular_damping = 0.005  # Angular damping for the puck

# Define goal parameters
goal_width = 0.2  # Width of the goal opening

# Define robot parameters
robot_base_width = 0.15  # Physical width of the robot base
robot_z = table_top_z  # On table
robot_offset = 0.2  # Distance behind the goal


#---------------------------------------------
# Helpers
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

#---------------------------------------------
# Create the table, rails, and puck  
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
            'bullet.friction': table_friction,
            'bullet.restitution': table_restitution
        })
    
    return table

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
            'bullet.friction': 0.01,   # Rails should barely resist puck
            'bullet.restitution': 0.9  # High bounce off rail
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
    
    sim.setObjectPosition(puck, [0, 0, table_top_z + puck_height*2], sim.handle_parent)
    sim.setShapeColor(puck, None, sim.colorcomponent_ambient_diffuse, [1, 0, 0])
    sim.setObjectAlias(puck, "HockeyPuck")
    
    def apply_puck_physics(sim, puck_handle):
        inertia = [
            puck_inertia_x,
            puck_inertia_y,
            puck_inertia_z
        ]
        inertia_matrix = [
            inertia[0], 0, 0,
            0, inertia[1], 0,
            0, 0, inertia[2]
        ]
        sim.setShapeMass(puck_handle, puck_mass)
        sim.setShapeInertia(puck_handle, inertia_matrix, [1,0,0,0, 0,1,0,0, 0,0,1,0])  # COM at origin

        apply_physics_properties(sim, puck_handle, {
            'dynamic': True,
            'respondable': True,
            'bullet.friction': puck_friction,
            'bullet.restitution': puck_restitution,
            'bullet.linearDamping': puck_linear_damping,
            'bullet.angularDamping': puck_angular_damping,
            'bullet.customCollisionMarginEnabled': True,
            'bullet.customCollisionMarginValue': 0.001
        })
        
    apply_puck_physics(sim, puck)

    return puck

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
        A tuple containing the created table object, puck object, and camera object.
    """
    table = create_table(sim)
    create_rails(sim, table)
    puck = create_puck(sim)
    camera = create_top_camera(sim, table)
    return table, puck, camera

#---------------------------------------------
# Create robots and place them
#---------------------------------------------

def load_and_place_robot_pair(sim, attach_paddles=True, disable_attached_scripts=True):
    """
    Loads and places two robot arms symmetrically on either side of the table.

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

    # Define left and right robot positions
    left_pos  = [-table_length / 2 - robot_offset, 0, robot_z]
    right_pos = [ table_length / 2 + robot_offset, 0, robot_z]

    # Load both robots
    robot_left = _load_robot_at(left_pos, "RobotLeft")
    robot_right = _load_robot_at(right_pos, "RobotRight")

    # Get end effectors (assumes "link7_visible" as the end)
    eff_left = sim.getObject(f"{sim.getObjectAlias(robot_left, 2)}/link7_visible")
    eff_right = sim.getObject(f"{sim.getObjectAlias(robot_right, 2)}/link7_visible")

    if attach_paddles:
        attach_paddle(sim, eff_left, [0, 0, 1], "LeftPaddle")
        attach_paddle(sim, eff_right, [0, 1, 0], "RightPaddle")

    return {
        'robot_left': robot_left,
        'robot_right': robot_right,
        'effector_left': eff_left,
        'effector_right': eff_right
    }


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
    sim.setObjectOrientation(paddle, [0, math.radians(90), 0], sim.handle_parent)
    sim.setObjectAlias(paddle, name)


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

def get_robot_left_position():
    """
    Calculate the base position of the left robot relative to the center of the table.

    Returns:
        list: A list containing the x, y position of the left robot base.
    """
    return [-table_length / 2 - 0.2, 0]  # 0.2 is the global robot_offset


def get_robot_right_position():
    """
    Calculate the base position of the right robot relative to the center of the table.

    Returns:
        list: A list containing the x, y position of the right robot base.
    """
    # Left robot base position (behind the left goal)
    robot_left_pos = [-table_length / 2 - robot_offset, 0]  # (x, y)

    # Right robot base position (behind the right goal)
    robot_right_pos = [table_length / 2 + robot_offset, 0]  # (x, y)

#---------------------------------------------
# Setup the scene 
#---------------------------------------------
def setup_scene():
    """
    Sets up the complete simulation scene, including the table, puck, and robots.

    Returns:
        tuple: The sim object and dictionary of handles.
    """
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    print('[INFO] Connected to CoppeliaSim')

    sim.stopSimulation()
    sim.startSimulation()
    time.sleep(1)

    robot_handles = load_and_place_robot_pair(sim, attach_paddles=True)
    table, puck, camera = create_hockey_table(sim)

    initialize_robot_joints(sim, robot_handles['robot_left'], [-90, 60, 45, -15, -90, 180])
    initialize_robot_joints(sim, robot_handles['robot_right'], [90, 60, 45, -15, -90, 90])

    handles = {\
        'table': table,
        'camera': camera,
        'effector_left': robot_handles['effector_left'],
        'effector_right': robot_handles['effector_right'],
        'robot_left': robot_handles['robot_left'],
        'robot_right': robot_handles['robot_right'],
        'puck': puck
    }

    print("[TEST] Scene setup complete.")
    print("Table handle:", handles['table'])
    print("Camera handle:", handles['camera'])
    print("Effector (left) handle:", handles['effector_left'])
    print("Effector (right) handle:", handles['effector_right'])
    print("Robot left:", handles['robot_left'])
    print("Robot right:", handles['robot_right'])
    print("Puck handle:", handles['puck'])

    return sim, handles
#---------------------------------------------

def main(sim_file=None):
    sim, handles = setup_scene()
    
    input("[TEST] Press Enter to stop simulation...")
    
    if sim_file:
        print(f'Saving sim to {sim_file}')
        sim.saveScene(sim_file)
    sim.stopSimulation()

if __name__ == "__main__":
    sim_file = os.path.join(os.path.dirname(__file__), 'air_hockey.ttt')  # Example default file path
    main(sim_file=sim_file)

