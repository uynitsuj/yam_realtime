# How to Permanently Assign a Fixed Video Device Path (/dev/videoX) on Linux

On Linux, video device names like /dev/video0, /dev/video1 are assigned dynamically. When you reboot or plug/unplug a camera, these numbers may change — which can break your code if it assumes a fixed device ID.

This guide explains how to assign a persistent, symbolic name to a specific camera (e.g., a ZED 2i), and how to access it from code (e.g., OpenCV).

---

🔧 Step 1: Identify Your Camera

**1. Find the Device Path:**

First, list all available video devices and find the one corresponding to your camera. You can use the `v4l2-ctl` tool (install it if needed, e.g., `sudo apt install v4l-utils` on Debian/Ubuntu).

    v4l2-ctl --list-devices

This command will output a list of devices and their corresponding `/dev/videoX` paths. For example:

    ZED 2i Camera (usb-0000:03:00.0-2):
            /dev/video0
            /dev/video1

    Integrated Camera (usb-0000:00:14.0-1):
            /dev/video2
            /dev/video3

Identify the name of your camera (e.g., "ZED 2i Camera") and **usually the first one is the main video stream**.

**2. Get Device Attributes:**

Run `udevadm` using the device path you found:

    # Replace /dev/video0 with the correct path if different
    udevadm info --attribute-walk --name=/dev/video0 | grep -E 'idVendor|idProduct|serial'

Sample output:

    ATTRS{idVendor}=="2b03"
    ATTRS{idProduct}=="f880"
    ATTRS{serial}=="OV0001"

Take note of these values — `idVendor`, `idProduct`, and `serial` uniquely identify your camera. If `serial` is missing or not unique, you might need to find other unique attributes from the full `udevadm info` output to use in your rule.

---

📝 Step 2: Create a Udev Rule

Create a custom udev rule file:

    sudo nano /etc/udev/rules.d/99-usb-video.rules

Add the following line (replace idVendor, idProduct, and serial as needed):

    SUBSYSTEM=="video4linux", ATTR{name}=="ZED 2i: ZED 2i", SUBSYSTEMS=="usb", ATTR{index}=="0", ATTRS{serial}=="OV0001", SYMLINK+="zed2i", MODE="0666", OPTIONS+="last_rule"

This will create a persistent symbolic link at /dev/video-zed2i.

If using an OAK camera on Linux, you will also need a udev rule to allow the USB device to be accessed by non-root users. The following rule has to be set to allow access to the USB device:

    ```
    echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
    sudo udevadm control --reload-rules && sudo udevadm trigger
    ```

---

🔄 Step 3: Reload Udev Rules

Run the following to reload udev and apply the rule:

    sudo udevadm control --reload-rules
    sudo udevadm trigger

You should now see the symbolic link:

    ls -l /dev/video-zed2i
    # Example output: /dev/video-zed2i -> video0

---

✅ Step 4: Use the Stable Device in Code

Now in your code (e.g., Python + OpenCV), use the persistent path instead of a number:

    import cv2

    cap = cv2.VideoCapture("/dev/video-zed2i")

This will always open the correct camera, regardless of whether it was assigned /dev/video0, /dev/video1, etc.
