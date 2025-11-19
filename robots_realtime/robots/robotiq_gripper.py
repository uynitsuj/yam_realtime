"""pyRobotiqGripper: Python Driver for Robotiq Grippers via Modbus RTU

pyRobotiqGripper is a Python library designed to facilitate control of Robotiq\
grippers using Modbus RTU communication via serial port.

This module provides documentation in two formats:

- Docstrings: Embedded within the code for easy access.
- Online Documentation: Extensive documentation available at\
    <https://pyrobotiqgripper.readthedocs.io/en/latest/>.
"""

# General information
__author__ = "Benoit CASTETS"
__email__ = "opensourceeng@robotiq.com"
__license__ = "Apache License, Version 2.0"
__url__ = "https://github.com/castetsb/pyRobotiqGripper"
__version__ = "1.0.0"

# Iport libraries
import logging
import time
from typing import Optional, Tuple

import minimalmodbus as mm
import serial
import serial.tools.list_ports
from robots_realtime.robots.utils import Rate
from typing import Any, Dict
import numpy as np
from i2rt.robots.robot import Robot
import threading
from i2rt.utils.utils import RateRecorder
from dataclasses import dataclass

from robots_realtime.utils.performance_utils import set_realtime_and_pin

# Constants
BAUDRATE = 115200
BYTESIZE = 8
PARITY = "N"
STOPBITS = 1
TIMEOUT = 0.2
AUTO_DETECTION = "auto"

logger = logging.getLogger(__name__)


class RobotiqGripper(mm.Instrument):
    """Object control Robotiq grippers (2F85, 2F140 or hande).

    Suppose that the gripper is connected via the USB/RS485 adapter to the PC\
    executing this code.

    Modbus RTU function code supported by robotiq gripper

    =======================================  ====================
    Description                              Modbus function code
    =======================================  ====================
    Read registers                           4
    Write registers                          16
    Master read & write multiple registers   23
    =======================================  ====================

    For more information for gripper communication please check gripper manual
    on Robotiq website.
    https://robotiq.com/support/2f-85-2f-140

    .. note::
        This object cannot be use to control epick, 3F or powerpick.
    """

    def __init__(self, portname=AUTO_DETECTION, slaveAddress=9):
        """Create a RobotiqGripper object whic can be use to control Robotiq\
        grippers using modbus RTU protocol USB/RS485 connection.

        Args:
            - portname (str, optional): The serial port name, for example\
                /dev/ttyUSB0 (Linux), /dev/tty.usbserial (OS X) or COM4\
                (Windows). It is necesary to allowpermission to access this\
                connection using the bash comman sudo chmod 666 /dev/ttyUSB0.\
                By default the portname is set to "auto". In this case the\
                connection is done with the first gripper found as connected\
                to the PC.
            - slaveaddress (int, optional): Address of the gripper (integer)\
                usually 9.
        """
        # Gripper salve address
        self.slaveAddress = slaveAddress
        # Port on which is connected the gripper
        if portname == "auto":
            self.portname = self._autoConnect()
            if self.portname is None:
                raise Exception("No gripper detected")
        else:
            self.portname = portname

        # Create a pyserial object to connect to the gripper
        ser = serial.Serial(self.portname, BAUDRATE, BYTESIZE, PARITY, STOPBITS, TIMEOUT)

        # Create the object using parent class contructor
        super().__init__(ser, self.slaveAddress, mm.MODE_RTU, close_port_after_each_call=False, debug=False)

        # Attribute to monitore if the gripper is processing an action
        self.processing = False

        # Maximum allowed time to perform and action
        self.timeOut = 10

        # Dictionnary where are stored description of each register state
        self.registerDic = {}
        self._buildRegisterDic()

        # Dictionnary where are stored register values retrived from the gripper
        self.paramDic = {}
        self.readAll()

        # Attributes to store open and close distance state information

        # Distance between the fingers when gripper is closed
        self.close_value = None
        # Position in bit when gripper is closed
        self.closebit = None

        # Distance between the fingers when gripper is open
        self.open_value = None
        # Position in bit when gripper is open
        self.openbit = None

        self._aCoef = None
        self._bCoef = None

    def _autoConnect(self):
        """Return the name of the port on which is connected the gripper"""
        ports = serial.tools.list_ports.comports()
        portName = None

        for port in ports:
            try:
                # Try opening the port
                ser = serial.Serial(port.device, BAUDRATE, BYTESIZE, PARITY, STOPBITS, TIMEOUT)

                device = mm.Instrument(
                    ser, self.slaveAddress, mm.MODE_RTU, close_port_after_each_call=False, debug=False
                )

                # Try to write the position 100
                device.write_registers(1000, [0, 100, 0])

                # Try to read the position request eco
                registers = device.read_registers(2000, 3, 4)
                posRequestEchoReg3 = registers[1] & 0b0000000011111111

                # Check if position request eco reflect the requested position
                if posRequestEchoReg3 != 100:
                    raise Exception("Not a gripper")
                portName = port.device
                del device

                ser.close()  # Close the port
            except Exception as e:
                print(f"Error connecting to gripper on port {port.device}: {e}")
                pass  # Skip if port cannot be opened

        if portName is not None:
            print(f"Found gripper on port {portName}")
        else:
            print("No gripper found")
        return portName

    def _buildRegisterDic(self) -> None:
        """Build a dictionnary with comment to explain each register variable.

        Dictionnary key are variable names. Dictionnary value are dictionnary\
        with comments about each statut of the variable (key=variable value,\
        value=comment)
        """
        ######################################################################
        # input register variable
        self.registerDic.update(
            {"gOBJ": {}, "gSTA": {}, "gGTO": {}, "gACT": {}, "kFLT": {}, "gFLT": {}, "gPR": {}, "gPO": {}, "gCU": {}}
        )

        # gOBJ
        gOBJdic = self.registerDic["gOBJ"]

        gOBJdic[0] = "Fingers are in motion towards requested position. No\
            object detected."
        gOBJdic[1] = "Fingers have stopped due to a contact while opening before\
            requested position. Object detected opening."
        gOBJdic[2] = "Fingers have stopped due to a contact while closing before\
            requested position. Object detected closing."
        gOBJdic[3] = "Fingers are at requested position. No object detected or\
            object has been loss / dropped."

        # gSTA
        gSTAdic = self.registerDic["gSTA"]

        gSTAdic[0] = "Gripper is in reset ( or automatic release ) state. See\
            Fault Status if Gripper is activated."
        gSTAdic[1] = "Activation in progress."
        gSTAdic[3] = "Activation is completed."

        # gGTO
        gGTOdic = self.registerDic["gGTO"]

        gGTOdic[0] = "Stopped (or performing activation / automatic release)."
        gGTOdic[1] = "Go to Position Request."
        gGTOdic[2] = "Unknown status"
        gGTOdic[3] = "Unknown status"

        # gACT
        gACTdic = self.registerDic["gACT"]

        gACTdic[0] = "Gripper reset."
        gACTdic[1] = "Gripper activation."

        # kFLT
        kFLTdic = self.registerDic["kFLT"]
        i = 0
        while i < 256:
            kFLTdic[i] = i
            i += 1

        # See your optional Controller Manual (input registers & status).

        # gFLT
        gFLTdic = self.registerDic["gFLT"]
        i = 0
        while i < 256:
            gFLTdic[i] = i
            i += 1
        gFLTdic[0] = "No fault (LED is blue)"
        gFLTdic[5] = "Priority faults (LED is blue). Action delayed, activation\
            (reactivation) must be completed prior to perfmoring the action."
        gFLTdic[7] = "Priority faults (LED is blue). The activation bit must be\
            set prior to action."
        gFLTdic[8] = "Minor faults (LED continuous red). Maximum operating\
            temperature exceeded, wait for cool-down."
        gFLTdic[9] = "Minor faults (LED continuous red). No communication during\
            at least 1 second."
        gFLTdic[10] = "Major faults (LED blinking red/blue) - Reset is required\
            (rising edge on activation bit rACT needed). Under minimum\
            operating voltage."
        gFLTdic[11] = "Major faults (LED blinking red/blue) - Reset is required\
            (rising edge on activation bit rACT needed). Automatic release in\
            progress."
        gFLTdic[12] = "Major faults (LED blinking red/blue) - Reset is required\
            (rising edge on activation bit rACT needed). Internal fault;\
            contact support@robotiq.com."
        gFLTdic[13] = "Major faults (LED blinking red/blue) - Reset is required\
            (rising edge on activation bit rACT needed). Activation fault,\
            verify that no interference or other error occurred."
        gFLTdic[14] = "Major faults (LED blinking red/blue) - Reset is required\
            (rising edge on activation bit rACT needed). Overcurrent triggered."
        gFLTdic[15] = "Major faults (LED blinking red/blue) - Reset is required\
            (rising edge on activation bit rACT needed). Automatic release\
            completed."

        # gPR
        gPRdic = self.registerDic["gPR"]

        i = 0
        while i < 256:
            gPRdic[i] = "Echo of the requested position for the Gripper:\
                {}/255".format(i)
            i += 1

        # gPO
        gPOdic = self.registerDic["gPO"]
        i = 0
        while i < 256:
            gPOdic[i] = "Actual position of the Gripper obtained via the encoders:\
                {}/255".format(i)
            i += 1

        # gCU
        gCUdic = self.registerDic["gCU"]
        i = 0
        while i < 256:
            current = i * 10
            gCUdic[i] = "The current is read instantaneously from the motor\
                drive, approximate current: {} mA".format(current)
            i += 1

        ######################################################################
        # output register variable
        self.registerDic.update({"rARD": {}, "rATR": {}, "rGTO": {}, "rACT": {}, "rPR": {}, "rFR": {}, "rSP": {}})

        ######################################################################

    def readAll(self) -> None:
        """Retrieve gripper output register information and save it in the\
            parameter dictionary.

        The dictionary keys are as follows:

        - gOBJ: Object detection status. This built-in feature provides\
            information on possible object pick-up. Ignore if gGTO == 0.
        - gSTA: Gripper status. Returns the current status and motion of the\
            gripper fingers.
        - gGTO: Action status. Echo of the rGTO bit (go-to bit).
        - gACT: Activation status. Echo of the rACT bit (activation bit).
        - kFLT: See your optional controller manual for input registers and\
            status.
        - gFLT: Fault status. Returns general error messages useful for\
            troubleshooting. A fault LED (red) is present on the gripper\
            chassis. The LED can be blue, red, or both, and can be solid\
            or blinking.
        - gPR: Echo of the requested position for the gripper. Value between\
            0x00 and 0xFF.
        - gPO: Actual position of the gripper obtained via the encoders.\
            Value between 0x00 and 0xFF.
        - gCU: The current is read instantaneously from the motor drive. Value\
            between 0x00 and 0xFF. Approximate current equivalent is 10 times\
            the value read in mA.
        """
        # Clear parameter dictionnary data
        self.paramDic = {}

        # Read 3 16bits registers starting from register 2000
        registers = self.read_registers(2000, 3)

        #########################################
        # Register 2000
        # First Byte: gripperStatus
        # Second Byte: RESERVED

        # First Byte: gripperStatus
        gripperStatusReg0 = (registers[0] >> 8) & 0b11111111  # xxxxxxxx00000000
        #########################################
        # Object detection
        self.paramDic["gOBJ"] = (gripperStatusReg0 >> 6) & 0b11  # xx000000
        # Gripper status
        self.paramDic["gSTA"] = (gripperStatusReg0 >> 4) & 0b11  # 00xx0000
        # Action status. echo of rGTO (go to bit)
        self.paramDic["gGTO"] = (gripperStatusReg0 >> 3) & 0b1  # 0000x000
        # Activation status
        self.paramDic["gACT"] = gripperStatusReg0 & 0b00000001  # 0000000x

        #########################################
        # Register 2001
        # First Byte: Fault status
        # Second Byte: Pos request echo

        # First Byte: fault status
        faultStatusReg2 = (registers[1] >> 8) & 0b11111111  # xxxxxxxx00000000
        #########################################
        # Universal controler
        self.paramDic["kFLT"] = (faultStatusReg2 >> 4) & 0b1111  # xxxx0000
        # Fault
        self.paramDic["gFLT"] = faultStatusReg2 & 0b00001111  # 0000xxxx

        #########################################
        # Second Byte: Pos request echo
        posRequestEchoReg3 = registers[1] & 0b11111111  # 00000000xxxxxxxx
        #########################################
        # Echo of request position
        self.paramDic["gPR"] = posRequestEchoReg3

        #########################################
        # Register 2002
        # First Byte: Position
        # Second Byte: Current

        # First Byte: Position
        positionReg4 = (registers[2] >> 8) & 0b11111111  # xxxxxxxx00000000

        #########################################
        # Actual position of the gripper
        self.paramDic["gPO"] = positionReg4

        #########################################
        # Second Byte: Current
        currentReg5 = registers[2] & 0b0000000011111111  # 00000000xxxxxxxx
        #########################################
        # Current
        self.paramDic["gCU"] = currentReg5

    def reset(self) -> None:
        """Reset the gripper (clear previous activation if any)"""
        # Reset the gripper
        self.write_registers(1000, [0, 0, 0])

    def activate(self) -> None:
        """If not already activated, activate the gripper.

        .. warning::
            When you execute this function the gripper is going to fully open\
            and close. During this operation the gripper must be able to freely\
            move. Do not place object inside the gripper.
        """
        # Turn the variable which indicate that the gripper is processing
        # an action to True
        self.processing = True

        # Activate the gripper
        # rACT=1 Activate Gripper (must stay on after activation routine is
        # completed).
        self.write_registers(1000, [0b0000000100000000, 0, 0])

        # Waiting for activation to complete
        activationStartTime = time.time()
        activationCompleted = False
        activationTime = 0

        while (not activationCompleted) and activationTime < self.timeOut:
            activationTime = time.time() - activationStartTime

            self.readAll()
            gSTA = self.paramDic["gSTA"]

            if gSTA == 3:
                activationCompleted = True
                print("Activation completed. Activation time : ", activationTime)
        if activationTime > self.timeOut:
            raise Exception("Activation did not complete without timeout.")

        self.processing = False

    def resetActivate(self) -> None:
        """Reset the gripper (clear previous activation if any) and activat\
        the gripper. During this operation the gripper will open and close.
        """
        # Reset the gripper
        self.reset()
        # Activate the gripper
        self.activate()

    def goTo(self, position: int, speed: int = 255, force: int = 255, non_blocking: bool = False):
        """Go to the position with determined speed and force.

        Args:
            - position (int): Position of the gripper. Integer between 0 and 255.\
            0 being the open position and 255 being the close position.
            - speed (int): Gripper speed between 0 and 255
            - force (int): Gripper force between 0 and 255

        Returns:
            - objectDetected (bool): True if object detected
            - position (int): End position of the gripper in bits
        """
        # Check if the grippre is activated
        if not self.isActivated:
            raise Exception(
                "Gripper must be activated before requesting\
                             an action."
            )

        # Check input value
        if position > 255:
            raise Exception("Position value cannot exceed 255")
        elif position < 0:
            raise Exception("Position value cannot be under 0")

        self.processing = True

        # rARD(5) rATR(4) rGTO(3) rACT(0)
        # gACT=1 (Gripper activation.) and gGTO=1 (Go to Position Request.)
        self.write_registers(1000, [0b0000100100000000, position, speed * 0b100000000 + force])
        if non_blocking:
            # read the current status then return
            now = time.time()
            self.readAll()
            gOBJ = self.paramDic["gOBJ"]
            objectDetected = False
            if gOBJ in [1, 2]:
                # Fingers have stopped due to a contact
                objectDetected = True

            elif gOBJ == 3:
                # Fingers are at requested position.
                objectDetected = False
            position = self.paramDic["gPO"]
            return position, objectDetected

        # Waiting for activation to complete
        motionStartTime = time.time()
        motionCompleted = False
        motionTime = 0
        objectDetected = False

        while (not objectDetected) and (not motionCompleted) and (motionTime < self.timeOut):
            motionTime = time.time() - motionStartTime
            self.readAll()
            # Object detection status, is a built-in feature that provides
            # information on possible object pick-up. Ignore if gGTO == 0.
            gOBJ = self.paramDic["gOBJ"]

            if gOBJ in [1, 2]:
                # Fingers have stopped due to a contact
                objectDetected = True

            elif gOBJ == 3:
                # Fingers are at requested position.
                objectDetected = False
                motionCompleted = True

        if motionTime > self.timeOut:
            raise Exception(
                "Gripper never reach its requested position and\
                            no object have been detected"
            )

        position = self.paramDic["gPO"]

        return position, objectDetected

    def close(self, speed=255, force=255) -> None:
        """Close the gripper.

        Args:
            - speed (int, optional): Gripper speed between 0 and 255.\
            Default is 255.
            - force (int, optional): Gripper force between 0 and 255.\
            Default is 255.
        """
        self.goTo(255, speed, force)

    def open(self, speed=255, force=255) -> None:
        """Open the gripper

        Args:
            - speed (int, optional): Gripper speed between 0 and 255.\
            Default is 255.
            - force (int, optional): Gripper force between 0 and 255.\
            Default is 255.
        """
        self.goTo(0, force, speed)

    def go_to_normalized_value(self, position, speed=255, force=255) -> None:
        """Go to the requested opening expressed in normalized value

        Args:
            - position (float): Gripper opening in normalized value.
            - speed (int, optional): Gripper speed between 0 and 255.\
            Default is 255.
            - force (int, optional): Gripper force between 0 and 255.\
            Default is 255.

        .. note::
            Calibration is needed to use this function.\n
            Execute the function calibrate at least 1 time before using this function.
        """
        if not self.isCalibrated:
            raise Exception("The gripper must be calibrated before been requested to go to a position in mm")

        if position > self.close_value and position < self.open_value:
            raise Exception(
                f"The position {position} is out of the calibrated range {self.close_value} - {self.open_value}"
            )

        position = int(self._normalized_value_to_bit(position))
        self.goTo(position, speed, force)

    def getPosition(self):
        """Return the position of the gripper in bits

        Returns:
            - int: Position of the gripper in bits.
        """
        self.readAll()

        position = self.paramDic["gPO"]

        return position

    def _normalized_value_to_bit(self, position):
        """Convert a normalized value gripper opening in bit opening.

        .. note::
            Calibration is needed to use this function.\n
            Execute the function calibrate at least 1 time before using this function.
        """
        bit = (position - self._bCoef) / self._aCoef_with_dead_zone  # to make it close tighter
        bit = max(min(bit, 255), 0)

        return int(bit)

    def _bit_to_normalized_value(self, bit):
        """Convert a bit gripper opening in normalized value.

        Returns:
            float: Gripper position converted in normalized value.

        .. note::
            Calibration is needed to use this function.\n
            Execute the function calibrate at least 1 time before using this function.
        """
        position = self._aCoef * bit + self._bCoef

        if self.open_value is not None and self.close_value is not None:
            return max(min(position, self.open_value), self.close_value)
        else:
            return position

    def get_pos_normalized_value(self):
        """Return the position of the gripper in mm.

        Returns:
            float: Current gripper position in mm

        .. note::
            Calibration is needed to use this function.\n
            Execute the function calibrate at least 1 time before using this function.
        """
        position = self.getPosition()

        positionmm = self._bit_to_normalized_value(position)
        return positionmm

    def calibrate(self, close_value, open_value, bit_dead_zone=2) -> None:
        """Calibrate the gripper for normalized value positionning.

        Once the calibration is done it is possible to control the gripper in\
        normalized value.

        Args:
            - close_value (float): Normalized value when the gripper is\
            fully closed.
            - open_value (float): Normalized value when the gripper is\
            fully open.
        """
        self.close_value = close_value
        self.open_value = open_value

        self.open()
        # get open bit
        self.openbit = self.getPosition()
        obit = self.openbit

        self.close()
        # get close bit
        self.closebit = self.getPosition()
        cbit = self.closebit

        self._aCoef = (close_value - open_value) / (cbit - obit)
        self._aCoef_with_dead_zone = (close_value - open_value) / (cbit - obit + bit_dead_zone)
        self._bCoef = open_value

    def printInfo(self) -> None:
        """Print gripper register info in the python terminal"""
        self.readAll()
        for key, value in self.paramDic.items():
            print("{} : {}".format(key, value))
            print(self.registerDic[key][value])

    def isActivated(self):
        """Tells if the gripper is activated

        Returns:
            bool: True if the gripper is activated. False otherwise.
        """

        self.readAll()
        is_activated = self.paramDic["gSTA"] == 3

        return is_activated

    def isCalibrated(self):
        """Return if the gripper is qualibrated

        Returns:
            bool: True if the gripper is calibrated. False otherwise.
        """
        is_calibrated = False
        if (self.open_value is None) or (self.close_value is None):
            is_calibrated = False
        else:
            is_calibrated = True

        return is_calibrated



@dataclass
class RobotiqGripperCommand:
    joint_pos: int  # 0-255
    speed: int
    force: int


class RobotiqGripperRobot(Robot):
    """Robotiq gripper that follows the Robot Protocol.

    Maps gripper position to a 0-1 range where:
    - 1 = fully open
    - 0 = fully closed
    """

    def __init__(
        self,
        port: str = "auto",
        default_speed: int = 25,
        default_force: int = 25,
        debug: bool = False,
        device_name: Optional[str] = None,
        robotiq_min_frequency: Optional[float] = 80.0,
        pinned_cpu: int | None = None,
    ) -> None:
        """Initialize the RobotiqGripperRobot.

        Args:
            port: Serial port name or "auto" for automatic detection
            robotiq_min_frequency: If not None, sets the minimum required frequency in Hz (default: 80.0)
        """
        if pinned_cpu is not None:
            set_realtime_and_pin(pinned_cpu)
        self._joint_state_saver = None
        self._debug = debug
        try:
            self._gripper = RobotiqGripper(portname=port)
        except Exception as e:
            if device_name is None:
                logger.error(f"Failed to find the gripper in port: {port}.")
            else:
                logger.error(f"Failed to find the gripper: {device_name}, gripper port :{port}.")
            raise e

        # Activate the gripper if not already activated
        if not self._gripper.isActivated():
            self._gripper.resetActivate()

        # The gripper pos range is 0-255, but it is not reachable at the end. So we need to calibrate it at first.
        # We use 0-1 to calibrate the gripper.
        if not self._gripper.isCalibrated():
            self._gripper.calibrate(close_value=0, open_value=1)

        self._default_speed = default_speed
        self._default_force = default_force
        self._device_name = device_name or "robotiq_gripper"
        self._port = port
        self._robotiq_min_frequency = robotiq_min_frequency

        self._pos_bit, self._objectDetected = self._gripper.getPosition(), False
        # command uses 0-1 normalized command
        self.command = RobotiqGripperCommand(
            joint_pos=self._pos_bit, speed=self._default_speed, force=self._default_force
        )

        self._lock = threading.Lock()

        self._stop_thread = False
        self._command_thread = threading.Thread(target=self._command_execution_loop, daemon=True)
        self._command_thread.start()

    def _command_execution_loop(self) -> None:
        """Thread that processes gripper commands."""
        with RateRecorder(
            name="robotiq gripper loop"
        ) as rate_recorder:
            while not self._stop_thread:
                # Initialize t_start at the beginning to ensure it's always defined
                with self._lock:
                    current_position, object_detected = self._gripper.goTo(
                        self.command.joint_pos, self.command.speed, self.command.force, non_blocking=True
                    )
                    self._pos_bit = current_position
                    self._objectDetected = object_detected
                try:
                    rate_recorder.track()
                except RuntimeError as e:
                    logger.error(
                        f"Robotiq gripper frequency is too low. {self._robotiq_min_frequency} Hz is required. Solution: 1. Run: xdof/scripts/ftdi_latency_config.sh. or 2. set robotiq_min_frequency=None"
                    )
                    # raise e
                time.sleep(0.001)
                if self._joint_state_saver is not None:
                    self._joint_state_saver.add(
                        time.time(),
                        pos=self._gripper._bit_to_normalized_value(self._pos_bit),
                        vel=None,
                        eff=None,
                        ee_pos=None,
                    )

    def close(self) -> None:
        """Properly shutdown the command thread."""
        self._stop_thread = True
        if self._command_thread.is_alive():
            self._command_thread.join(timeout=1.0)

    def num_dofs(self) -> int:
        """Return number of degrees of freedom (always 1 for the gripper)."""
        return 1

    def get_joint_pos(self) -> np.ndarray:
        """Get current gripper position in normalized 0-1 range.

        Returns:
            np.ndarray: Position in range [0, 1] where 1 is open and 0 is closed
        """
        with self._lock:
            return np.array([self._gripper._bit_to_normalized_value(self._pos_bit)])

    def command_joint_pos(self, joint_pos: np.ndarray, speed: int | None = None, force: int | None = None) -> None:
        """Command the gripper to a position in the 0-1 range.

        Args:
            joint_pos: Normalized position in range [0, 1] where:
                       1 = fully open, 0 = fully closed
            speed: Speed of the gripper motion (0-255)
            force: Force applied by the gripper (0-255)
        """
        joint_pos = np.clip(joint_pos, 0, 1)
        # Convert from 0-1 range to 0-255 range (inverted since 255 is closed)
        gripper_pos_bits = self._gripper._normalized_value_to_bit(joint_pos)
        command_speed = speed or self._default_speed
        command_force = force or self._default_force

        with self._lock:
            self.command = RobotiqGripperCommand(joint_pos=gripper_pos_bits, speed=command_speed, force=command_force)

    def get_observations(self) -> Dict[str, Any]:  # Changed return type to Dict[str, Any]
        """Get current observations from the gripper.

        Returns:
            Dict with joint_pos, object_detected, speed, and force
        """
        with self._lock:
            return {
                "joint_pos": np.array([self._gripper._bit_to_normalized_value(self._pos_bit)]),
                "speed": np.array([self.command.speed]),
                "force": np.array([self.command.force]),
                "object_detected": np.array([self._objectDetected]),
            }


if __name__ == "__main__":
    import time

    gripper = RobotiqGripperRobot(port="/dev/robotiq", debug=True)
    # Test the frequency of goTo function
    iterations = 10
    start_time = time.time()
    rate = Rate(30)
    print("start")
    input("Press Enter to continue")
    for i in range(iterations):
        start_iter = time.time()
        cmd = np.array([i / iterations])
        now = time.time()
        gripper.command_joint_pos(cmd, speed=255)
        print(f"cmd time: {time.time() - now}")
        now = time.time()
        obs = gripper.get_observations()
        print(f"obs time: {time.time() - now}")
        rate.sleep()
        end_iter = time.time()
        print(f"curent cmd: {cmd}, obs: {obs}")
        print(f"Iteration {i + 1}: {1 / (end_iter - start_iter):.2f} Hz")
    for i in range(iterations):
        obs = gripper.get_observations()
        print(f"obs {obs}")
        rate.sleep()
    time.sleep(1)
    # close the gripper
    for i in range(iterations):
        gripper.command_joint_pos(np.array([1 - i / iterations]), speed=255)
        rate.sleep()
    total_time = time.time() - start_time
    time.sleep(15)