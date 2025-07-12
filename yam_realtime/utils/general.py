import subprocess
from enum import Enum
from typing import Any, List

import numpy as np
from omegaconf import ListConfig
from packaging import version


class AutoNameEnum(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name


def get_git_commit():
    # First try git command
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    except Exception as e:
        # Fallback to reading from file
        try:
            with open("git_version", "r") as f:
                commit_hash = f.read().strip()
                if commit_hash:
                    return commit_hash
        except Exception as e:
            print(f"Error reading git_version file: {e}")
            return "unknown"


def is_git_dirty():
    # First try git command
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"]).strip()
        return bool(status)
    except Exception as e:
        # Fallback to reading from file
        try:
            with open("git_dirty", "r") as f:
                dirty_count = int(f.read().strip())
                return dirty_count > 0
        except Exception as e:
            print(f"Error reading git_dirty file: {e}")
            return "unknown"


# Helper function to convert ndarray to list
def convert_to_list(obj):
    if isinstance(obj, ListConfig):
        return [convert_to_list(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_list(i) for i in obj]
    else:
        return obj


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_with_full_shift(x, w=1, b_x=0, b_y=0):
    return sigmoid(w * (x - b_x)) + b_y


def compare_versions(version1: str, version2: str) -> int:
    """
    Compare two version strings.

    :param version1: First version string.
    :param version2: Second version string.
    :return: -1 if version1 < version2, 1 if version1 > version2, 0 if they are equal.
    """
    v1 = version.parse(version1)
    v2 = version.parse(version2)

    if v1 < v2:
        return -1
    elif v1 > v2:
        return 1
    else:
        return 0


def is_nested_subset(small, big, path=""):
    """_summary_

    Args:
        small (_type_): action spec of robot
        big (_type_): action spec of agent
        path (str, optional): _description_. Defaults to "".

    check if small is a subset of big recursively

    """
    if isinstance(small, dict):
        if not isinstance(big, dict):
            return False, f"Expected dict at '{path}', got {type(big).__name__}"
        for key in small:
            if key not in big:
                return False, f"Missing key '{path + key}' in agent action_spec"
            ok, msg = is_nested_subset(small[key], big[key], path + key + ".")
            if not ok:
                return False, msg
        return True, ""
    else:
        # Compare leaf values, e.g., shapes of arrays
        if hasattr(small, "shape") and hasattr(big, "shape"):
            if small.shape != big.shape:
                return False, f"Shape mismatch at '{path}': env {small.shape}, agent {big.shape}"
        # Optional: Add type checking or tolerance checks here
        return True, ""


class LowPassFilter:
    def __init__(self, alpha: float, order: int = 1):
        assert 0 <= alpha <= 1
        assert order >= 1
        self.alpha = alpha
        self.order = order
        self.states: List[Any] = [None] * order

    def reset(self) -> None:
        self.states = [None] * self.order

    def filter(self, x):
        if isinstance(x, dict):
            result = {}
            for key, value in x.items():
                if isinstance(self.states[0], dict) and key in self.states[0]:
                    # Use existing state structure
                    current = value
                    for i in range(self.order):
                        if self.states[i] is None:
                            self.states[i] = {k: v for k, v in x.items()}
                        elif key not in self.states[i]:
                            self.states[i][key] = current
                        else:
                            self.states[i][key] = self.alpha * current + (1 - self.alpha) * self.states[i][key]
                        current = self.states[i][key]
                    result[key] = current
                else:
                    # Initialize state structure if needed
                    if self.states[0] is None:
                        self.states = [{} for _ in range(self.order)]

                    current = value
                    for i in range(self.order):
                        if key not in self.states[i]:
                            self.states[i][key] = current
                        else:
                            self.states[i][key] = self.alpha * current + (1 - self.alpha) * self.states[i][key]
                        current = self.states[i][key]
                    result[key] = current
            return result
        else:
            # Original implementation for non-dict inputs
            current = x
            for i in range(self.order):
                if self.states[i] is None:
                    self.states[i] = current
                else:
                    self.states[i] = self.alpha * current + (1 - self.alpha) * self.states[i]
                current = self.states[i]
            return current
