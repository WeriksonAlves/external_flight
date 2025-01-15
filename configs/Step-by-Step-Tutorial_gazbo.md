# **Setting Up Gazebo with Bebop2 in Docker on Ubuntu 24.04**

This guide walks you through setting up **Gazebo with Bebop2** for simulation using Docker on Ubuntu 24.04. We’ll enable graphical interface support to visualize the simulation.

---

## **1. Install Docker**

1. Update your system and install Docker:
   ```bash
   sudo apt update
   sudo apt install -y docker.io
   sudo usermod -aG docker $USER
   ```

2. Log out and log back in to apply the `docker` group permissions.

3. Verify Docker installation:
   ```bash
   docker --version
   ```

---

## **2. Pull the ROS Noetic Docker Image**

1. Download the official ROS Noetic Docker image:
   ```bash
   docker pull ros:noetic
   ```

2. Confirm the image has been downloaded:
   ```bash
   docker images
   ```

---

## **3. Create a ROS Workspace**

1. Create a directory for your ROS workspace:
   ```bash
   mkdir -p ~/gazebo_bebop_ws/src
   ```

2. Navigate to the workspace directory:
   ```bash
   cd ~/gazebo_bebop_ws
   ```

---

## **4. Enable GUI Access**

1. Allow Docker containers to access your graphical display:
   ```bash
   xhost +local:docker
   ```

   > To revoke this access later, use:
   ```bash
   xhost -local:docker
   ```

2. Ensure the `DISPLAY` environment variable is set:
   ```bash
   echo $DISPLAY
   ```

   - If it is empty, set it manually:
     ```bash
     export DISPLAY=:1
     ```

---

## **5. Launch the Docker Container**

Start the ROS Noetic Docker container with GUI and workspace access:
```bash
docker run -it \
--network host \
--device /dev/video0:/dev/video0 \
--name gazebo_bebop_builder \
-v ~/gazebo_bebop_ws:/root/gazebo_bebop_ws \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-e DISPLAY=$DISPLAY \
ros:noetic bash
```

---

## **6. Install Required Dependencies**

1. Inside the container, install the necessary tools and libraries:
   ```bash
   apt update
   apt install -y \
   build-essential \
   git \
   python3-rosdep \
   python3-catkin-tools \
   libusb-dev \
   libspnav-dev \
   libbluetooth-dev \
   libcwiid-dev \
   libgoogle-glog-dev \
   ros-noetic-mavros \
   ros-noetic-gazebo-ros-pkgs \
   ros-noetic-gazebo-ros-control \
   ros-noetic-xacro \
   ros-noetic-robot-state-publisher \
   ros-noetic-joint-state-publisher \
   ros-noetic-joy \
   ros-noetic-eigen-conversions \
   x11-apps \
   mesa-utils
   ```

2. Test graphical interface support:
   ```bash
   xclock
   ```

   > A clock window should appear on your host. If not, check the `DISPLAY` variable and `xhost` permissions.

3. Verify GPU and 3D rendering support:
   ```bash
   glxinfo | grep "OpenGL"
   ```

   > Ensure the output includes information about your OpenGL driver and version.

---

## **7. Initialize the ROS Workspace**

1. Inside the container, initialize the workspace:
   ```bash
   cd /root/gazebo_bebop_ws
   catkin init
   ```

2. Clone the necessary repositories:
   ```bash
    cd src
    git clone https://github.com/ethz-asl/mav_comm
    git clone -b noetic https://github.com/simonernst/iROS_drone
    git clone https://github.com/ros-drivers/joystick_drivers
   ```

3. Install dependencies:
   ```bash
    cd /root/gazebo_bebop_ws
    rosdep install --from-paths src --ignore-src -r -y
   ```

---

## **8. Build the Workspace**

1. Build the workspace inside the container:
   ```bash
    cd /root/gazebo_bebop_ws
    catkin build
   ```

---

## **9. Save the Container**

1. Exit the container:
   ```bash
   exit
   ```

2. Save the container as a new Docker image:
   ```bash
   docker commit gazebo_bebop_builder ros:gazebo_bebop
   ```

---

## **10. Run the Saved Image**

1. Start a container using the saved image:
   ```bash
   docker run -it \
   --rm \
   --network host \
   --device /dev/video0:/dev/video0 \
   --name gazebo_bebop_runner \
   --gpus all \
   -v ~/gazebo_bebop_ws:/root/gazebo_bebop_ws \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   -e DISPLAY=$DISPLAY \
   ros:gazebo_bebop bash
   ```

2. Source the workspace:
   ```bash
   source /root/gazebo_bebop_ws/devel/setup.bash
   ```

---

## **11. Launch Gazebo Simulation**

1. Start the Gazebo simulation:
   ```bash
   roslaunch rotors_gazebo mav_velocity_control_with_fake_driver.launch
   ```

2. Open another terminal and check available topics:
   ```bash
   docker exec -it gazebo_bebop_run bash
   source /root/gazebo_bebop_ws/devel/setup.bash
   rostopic list
   ```

   > You should see topics like `/bebop/image_raw` and `/gazebo/model_states`.

---

## **12. Troubleshooting**

- **Workspace Not Mounted**:  
   Ensure `~/gazebo_bebop_ws` exists on your host and is mounted in the container.  

- **Permission Issues**:  
   Grant appropriate permissions:
   ```bash
   chmod -R 777 ~/gazebo_bebop_ws
   ```

- **Missing Dependencies**:  
   Install missing dependencies using:
   ```bash
   rosdep install --from-paths src --ignore-src -r -y
   ```

