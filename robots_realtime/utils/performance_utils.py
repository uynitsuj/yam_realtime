import logging
import os

logger = logging.getLogger(__name__)


def set_realtime_and_pin(pinned_cpu: int):
    # Set CPU affinity to pin to specific core
    os.sched_setaffinity(0, {pinned_cpu})
    logger.info(f"Pinned robot process to CPU core {pinned_cpu}")

    # Set real-time scheduling policy (SCHED_RR) with high priority
    # We use SCHED_RR instead of SCHED_FIFO because otherwise the saving operation
    # will block the motor control loop, causing the robot arm to die.
    param = os.sched_param(90)
    os.sched_setscheduler(0, os.SCHED_RR, param)
    logger.info("Set SCHED_RR scheduling with priority 90")