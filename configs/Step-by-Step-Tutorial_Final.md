### **Step-by-Step Tutorial: Installing `bebop_autonomy` with ROS Noetic Using Docker and Setting Up Python 3.9**

This tutorial provides a comprehensive guide for setting up **bebop_autonomy** on **Ubuntu 24.04** using **Docker**. It includes instructions for configuring **Python 3.9**, cloning the `external_flight` repository (`join` branch), installing all necessary dependencies, and enabling the graphical interface for real-time visualization in the `main.py` script.

---

### **Prerequisites**
- Ubuntu 24.04 installed on your system.
- Docker installed (steps included below).
- Bebop drone available for testing.

---

### **Step 1: Install Docker**

Install Docker on your system:

```bash
sudo apt update
sudo apt install -y docker.io
sudo usermod -aG docker $USER
```

> **Note:** After adding your user to the `docker` group, log out and log back in to apply the changes.

---

### **Step 2: Pull the ROS Noetic Docker Image**

Download the official ROS Noetic Docker image:

```bash
docker pull ros:noetic
```

Verify the download:

```bash
docker image ls
```

---

### **Step 3: Set Up the ROS Workspace**

Create a directory on your host machine to store the ROS workspace:

```bash
mkdir -p ~/external_flight_ws/src
```

---

### **Step 4: Configure GUI Permissions**

Allow the Docker container to access the host's X11 display server:

1. Permit access to the graphical server:

   ```bash
   xhost +local:docker
   ```

   > **Note:** This command grants access to any Docker container running locally. You can revoke access later using `xhost -local:docker`.

2. Verify the `DISPLAY` environment variable:

   ```bash
   echo $DISPLAY
   ```

   It should return a value like `:1`. If not set, configure it manually:

   ```bash
   export DISPLAY=:1
   ```

---

### **Step 5: Launch the Docker Container**

Run the ROS Noetic container, mounting the workspace directory:

```bash
docker run -it --network host --device /dev/video0:/dev/video0 --name external_flight_builder -v ~/external_flight_ws:/root/external_flight_ws -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY ros:noetic bash
```

> **Explanation of the new flags:**
> - `--network host`: flag allows ROS to communicate with the Bebop drone over Wi-Fi.
> - `-v /tmp/.X11-unix:/tmp/.X11-unix`: Mounts the X11 server socket for GUI communication.
> - `-e DISPLAY=$DISPLAY`: Passes the host's `DISPLAY` environment variable to the container.
> - `--device /dev/video0:/dev/video0`: Grants access to the camera device (if applicable).

---

### **Step 6: Install Dependencies in the Container**

Inside the Docker container, install the required dependencies:

```bash
apt-get update
apt-get install -y \
  build-essential \
  git \
  python3 \
  python3-pip \
  python3-catkin-tools \
  libavahi-client-dev \
  ros-noetic-joy \
  ros-noetic-joy-teleop \
  ros-noetic-teleop-twist-joy \
  ros-noetic-camera-info-manager \
  ros-noetic-tf2* \
  ros-noetic-tf2-ros \
  ros-noetic-roslint \
  ros-noetic-cv-bridge \
  libgl1 \
  libavcodec-dev \
  libavformat-dev \
  libavutil-dev \
  libswscale-dev \
  ros-noetic-xacro \
  ros-noetic-robot-state-publisher \
  x11-apps
```

---

### **Step 7: Verify GUI Access**

To confirm the graphical interface works, run the following command inside the container:

```bash
xclock
```

> **Expected Result:** The clock interface should appear on your host machine. If not, review the `DISPLAY` and `xhost` settings.

---

### **Step 8: Configure the ROS Workspace**

1. Initialize and build the workspace:

   ```bash
    cd /root/external_flight_ws/src
    catkin_init_workspace
    cd ..
    catkin_make
    source devel/setup.bash
```

2. Install ROS dependencies:

   ```bash
    sudo rosdep init
    rosdep update
    rosdep install --from-paths src --ignore-src --rosdistro=noetic -y
    ```

3. Rebuild the workspace:

   ```bash
    cd /root/external_flight_ws
    rm -rf build devel
    catkin_make
    ```

---

### **Step 9: Clone the `external_flight` Repository (In the host)**

In the host, clone the repository and navigate to the `join` branch:

```bash
cd ~/external_flight_ws/src
mkdir -p env_external_flight
cd env_external_flight
git clone -b join https://github.com/WeriksonAlves/external_flight.git
```

On the host, copy the `install.sh` script into the container:

```bash
docker cp ~/external_flight_ws/src/env_external_flight/external_flight/myLibs/rospy_uav/configs/install.sh external_flight_builder:/root/install.sh
```

---

### **Step 10: Run the Installation Script and Export ROS Environment Variables**

1. Make the installation script executable and run it:

   ```bash
    cd /root
    chmod +x install.sh
    ./install.sh
    ```

2. Update the library path:

   ```bash
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/external_flight_ws/devel/lib/parrot_arsdk
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/root/external_flight_ws/devel/lib/parrot_arsdk" >> ~/.bashrc
    source ~/.bashrc
    ```

3. Configure the workspace environment:
    ```bash
    cd /root/external_flight_ws
    source devel/setup.bash
    ```

4. Configure the `ROS_IP` variables to connect to the network correctly:
    ```bash
    export ROS_IP=192.168.0.105
    echo "source /root/external_flight_ws/devel/setup.bash" >> ~/.bashrc
    echo "export ROS_IP=192.168.0.105" >> ~/.bashrc
    ```

**Note:** Replace `192.168.0.105` with the correct IP of your system, if necessary.

---

### **Step 11: Install Python 3.9 in the Container**

Install Python 3.9, pip, and necessary tools:

```bash
apt update
apt install -y software-properties-common
```

```bash
add-apt-repository -y ppa:deadsnakes/ppa
apt update
apt install -y python3.9 python3.9-distutils python3.9-dev python3.9-venv curl
```

```bash
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9
```

Verify the installation:

```bash
python3.9 --version
pip3.9 --version
```

---

### **Step 12: Set Up Python Environment**

Create and activate a virtual environment:

```bash
cd /root/external_flight_ws/src/env_external_flight
python3.9 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip3.9 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3.9 install -r external_flight/requirements.txt
```

---

### **Step 13: Save the Container as a Docker Image**

Save the current container state as a new Docker image:

```bash
exit
```

```bash
docker commit external_flight_builder ros:external_flight
```

---

### **Step 14: Verify Drone Connection**

Ensure the Bebop drone is connected to your machine:

```bash
ping 192.168.0.202
```

---

### **Step 15: Start the New Docker Image**

Launch a new container using the saved Docker image and ensure that the configurations have persisted:

```bash
docker run -it --rm   --network host   --device /dev/video0:/dev/video0   --name external_flight_runner   -v ~/external_flight_ws:/root/external_flight_ws   -v /tmp/.X11-unix:/tmp/.X11-unix   -e DISPLAY=$DISPLAY   ros:external_flight bash
```

```bash
export ROS_IP=192.168.0.105
echo "source /root/external_flight_ws/devel/setup.bash" >> ~/.bashrc
echo "export ROS_IP=192.168.0.105" >> ~/.bashrc
source ~/.bashrc
echo $ROS_IP
apt-get update && apt-get install -y iputils-ping
```

---

### **Step 16: Start the ROS Master Node**

Open another terminal and start `roscore`:

```bash
docker exec -it external_flight_runner bash
```

```bash
roscore
```

---

### **Step 17: Launch the Bebop Driver**

In a separate terminal, start the Bebop driver node:

```bash
docker exec -it external_flight_runner bash
```

```bash
cd /root/external_flight_ws
source devel/setup.bash
roslaunch bebop_driver bebop_node.launch
```

---

### **Step 18: Verify ROS Topics**

Check the ROS topics to confirm communication with the drone:

```bash
docker exec -it external_flight_runner bash
```

```bash
rostopic list
rostopic echo /bebop/image_raw
```

---

### **Step 19: Run `main.py`**


1. Export the `DISPLAY` variable inside the container:

```bash
export DISPLAY=:0
```

2. Start the graphical server if required by your setup:

```bash
Xvfb :1 -screen 0 1280x1024x24 &
export DISPLAY=:1
```

3. Execute the main program:

```bash
cd /root/external_flight_ws/src/env_external_flight
source /root/external_flight_ws/devel/setup.bash
source venv/bin/activate
python3.9 external_flight/main.py
```

The graphical interface should appear on your host machine, allowing you to view the processed images in real time.

---

### **If Errors Occur**

Review the logs and ensure all dependencies are correctly installed. Share the logs if you need further assistance!













-----------------------------------------------------------------------------------------------


It looks like FFmpeg is not installed on your system. You can install it using the following steps:

### Install FFmpeg:

1. **On Ubuntu 24.04** (or similar distributions), use the following command to install FFmpeg:

   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ```

2. **Verify Installation**:

   After installation, check if FFmpeg is correctly installed by running:

   ```bash
   ffmpeg -version
   ```

Once installed, try re-encoding the video with FFmpeg as mentioned earlier:

```bash
ffmpeg -i external_flight/video_01.mp4 -c:v libx264 -c:a aac output.mp4
```

Let me know if you need further assistance!






