# Gesture Recognition System (GRS)

**Python interface for Gesture Recognition System**

`grs` is a Python-based project designed to interface a modular tracking and classification system using open-source frameworks. It supports gesture recognition for command applications in autonomous robotics. This library is ideal for developers, researchers, and enthusiasts interested in gesture recognition systems in both indoor and outdoor environments.

---

## **Overview of the Gesture Recognition System**

The Gesture Recognition System (GRS) is a software solution for analyzing and interpreting human gestures captured through camera feeds. It supports real-time classification and gesture dataset management, making it applicable in various fields, such as:

- Human-computer interaction
- Autonomous robot control via gestures
- Sign language recognition
- Motion analysis

**Key Features:**

1. **Real-Time Gesture Recognition:**
   - Classify gestures from live camera streams.

2. **Dataset Management:**
   - Create and manage gesture datasets for training machine learning models.

3. **Validation Mode:**
   - Validate the classification accuracy of trained models.

---

## **Getting Started**

### **Installation Instructions**

To set up the system, install the required Python libraries and dependencies. It is recommended to use a virtual environment for clean installation. 

1. **Clone the repository:**
   ```bash
   mkdir env_grs
   cd env_grs/
   git clone https://github.com/WeriksonAlves/grs.git
   ```

2. **Set up a virtual environment and install dependencies:**
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate
   pip3.9 install -r grs/requirements.txt
   ```

3. **Verify Pytorch Installation:**
   Ensure you have the correct version of [Pytorch](https://pytorch.org/get-started/locally/) installed based on your system and hardware (e.g., GPU or CPU).

---

### **Usage Instructions**

The Gesture Recognition System supports three operation modes:

1. **Dataset Mode (D):**
   - Create gesture datasets to use as references for classification.

2. **Real-Time Mode (RT):**
   - Perform live gesture recognition using a connected camera.

3. **Validation Mode (V):**
   - Validate classifier performance using test datasets.

#### **Running the System**

1. **Initialize the Operating Mode:**
   Define the mode (e.g., Dataset, Real-Time, or Validate) using the `initialize_modes` function.
   
   Example:
   ```python
   operation_mode = initialize_modes(3)  # Real-Time mode
   ```

2. **Initialize the Gesture Recognition System:**
   Create the system with the desired configuration:
   ```python
   gesture_system = create_gesture_recognition_system(
       camera=4,  # or "http://192.168.209.199:81/stream" for ESPCam
       operation_mode=operation_mode,
       sps=None  # Optional: Servo Position System
   )
   ```

3. **Run the System:**
   Call the `run()` method to start gesture recognition.
   ```python
   try:
       gesture_system.run()
   finally:
       gesture_system.stop()
   ```

---

### **Modes and Configuration**

#### **Dataset Mode**
- Specify the classes for gestures and the filename to save the dataset.

#### **Real-Time Mode**
- Load a pre-existing dataset and classify gestures live from the camera feed.

#### **Validation Mode**
- Specify a test dataset and validate the model's performance.

Example configuration code:
```python
# Dataset Mode
dataset_mode = initialize_modes(1)

# Real-Time Mode
real_time_mode = initialize_modes(3)

# Validation Mode
validate_mode = initialize_modes(2)
```

---

### **Example Code**

Below is a complete example to initialize and run the system in Real-Time Mode:

```python
from gesture_recognition_system import initialize_modes, create_gesture_recognition_system

# Initialize Real-Time Mode
operation_mode = initialize_modes(3)

# Create Gesture Recognition System
gesture_system = create_gesture_recognition_system(
    camera=4,  # RealSense Camera
    operation_mode=operation_mode,
    sps=None  # Servo Position System (optional)
)

# Run the System
try:
    gesture_system.run()
finally:
    gesture_system.stop()
```

---

### **Dataset Example**

The system includes a sample dataset in the "datasets" folder. Use this dataset to test the Real-Time or Validation mode. Launch the system via the `main.py` script. When using this example, ensure:

- Gestures are initiated by bringing the fingertips of the tracked hand together until their distance is less than `0.025`.
- The system displays the action's stages (`S0` for awaiting trigger, `S1` for storing gesture information).

---

## **Documentation**

Detailed documentation on the system's architecture, modules, and configuration options is available in the `docs` folder. Key sections include:

- Camera setup and supported models
- Modes of operation
- Gesture classification pipeline
- ROS integration and logging

---

## **Examples**

The following image demonstrates the gestures included in the sample dataset:

![Gestures Overview](docs/images/gesture_class.png)

Run the example as follows:
```bash
python main.py
```

*Note:* Ensure the preview window indicates the correct gesture stage (e.g., `S0`, `S1`) during operation.

---

## **Updates**

Version 3.0 introduces the following changes:
- Refactored code to comply with PEP8 and clean code principles.
- Improved logging using `rospy`.
- Added modular design for easier extension.

---

## **Contributing**

We welcome contributions from the community! Please follow these steps to contribute:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with detailed descriptions of your changes.

Feel free to open issues or suggest improvements.

---

## **Contact Information**

For questions, suggestions, or feedback, please reach out:

- **Name**: Werikson Alves
- **Email**: werikson.alves@ufv.br
- **GitHub**: [Your GitHub Profile](https://github.com/WeriksonAlves)
