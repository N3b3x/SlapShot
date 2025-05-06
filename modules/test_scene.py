from scene_builder import setup_scene  # Adjust import if in subfolder like 'modules'

def main():
    sim, handles = setup_scene()

    print("[TEST] Scene setup complete.")
    print("Camera handle:", handles['camera'])
    print("Puck handle:", handles['effector_left'])
    print("Robot left:", handles['robot_left'])
    print("Robot right:", handles['robot_right'])

    input("[TEST] Press Enter to stop simulation...")
    sim.stopSimulation()

if __name__ == "__main__":
    main()
