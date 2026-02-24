import yaml
from pathlib import Path
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode
# from lerobot.motors.configs import FeetechMotorsBusConfig

# Map the YAM data to LeRobot's FeetechMotorsBusConfig
# A typical GELLO setup uses IDs 1-6 for joints and 7 for the gripper
motors = {
    f"joint_{i+1}": Motor(model="sts3215", id=i+1, norm_mode=MotorNormMode.DEGREES) for i in range(6)
}
motors["gripper"] = Motor(model="sts3215", id=7, norm_mode=MotorNormMode.DEGREES)

# Initialize and connect the bus
bus = FeetechMotorsBus(port ="/dev/tty.usbmodem5AE60805531", motors=motors)
bus.connect()

try:
    print("Reading precise position and current feedback (7 motors)...")
    while True:
        # 'sync_read' returns a list/tensor of values for all 7 motors at once
        positions = bus.sync_read("Present_Position")
        currents = bus.sync_read("Present_Current")
        
        # Output example: Gripper (Index 6)
        print(f"Joint 1 Pos: {positions[0]} | Gripper Current: {currents[6]} mA")
        
        # High-frequency loop for haptic feedback (e.g., 50Hz)
        import time
        time.sleep(0.02)

except KeyboardInterrupt:
    bus.disconnect()
    print("\nBus disconnected.")