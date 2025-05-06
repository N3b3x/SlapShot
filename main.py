from modules.scene_builder import setup_scene
from modules.puck_tracker import PuckTracker
from modules.strike_planner import StrikePlanner
import time
import os
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# Flags to enable/disable components
ENABLE_PUCK_TRACKER = False
ENABLE_STRIKE_PLANNER = True  # Only relevant if ENABLE_PUCK_TRACKER is True

def main():

    # sim, handles = setup_scene()
    
    client = RemoteAPIClient()
    sim = client.getObject('sim')

    sim_file = os.path.join(os.path.dirname(__file__), 'modules/air_hockey.ttt')
    
    sim.stopSimulation()
    sim.loadScene(sim_file)
    sim.startSimulation()

    handles =  {
        'camera': sim.getObject('./TopCamera'),
        'robot_left': sim.getObject('./UR3[0]'),
        'robot_right': sim.getObject('./UR3[1]'),
        'effector_left': sim.getObject('./LeftPaddle'),
        'effector_right': sim.getObject('./RightPaddle'),
        'puck': sim.getObject('./HockeyPuck')
    }
    tracker = None
    striker = None

    if ENABLE_PUCK_TRACKER:
        tracker = PuckTracker(sim, handles['camera'])
    if ENABLE_STRIKE_PLANNER:
        striker = StrikePlanner(sim, handles['robot_left'])

    prev_pos = None
    print("[INFO] Starting puck tracking and interception...")

    
    # starting_puck_vel = [-0.1]

    # dynamically enable the puck
    # sim.setObjectFloatParam(handles['puck'], sim.shapefloatparam_init_velocity_x, -1.0)
    sim.addForce(handles['puck'], [0, 0, 0], [1, 0, 0])
    try:
        while True:
            # pos = tracker.get_puck_position()
            # if pos and prev_pos:
            #     strike_point, strike_vel = striker.plan_strike(prev_pos, pos)
            #     if strike_point:
            #         striker.execute_strike(strike_point, strike_vel)
            # prev_pos = pos

            puck_pos, puck_vel = striker.extract_data_from_sim(handles['puck'])
            joint_angles = striker.plan_strike(puck_pos, puck_vel)

            if joint_angles is not None:
                striker.execute_strike(joint_angles)

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping simulation...")
        sim.stopSimulation()
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sim.stopSimulation()

if __name__ == "__main__":
    main()