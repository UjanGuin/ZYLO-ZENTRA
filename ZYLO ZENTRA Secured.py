import cv2
import numpy as np 
import os
import time
import math
from picamera2 import Picamera2
import pyttsx3
import speech_recognition as sr
from quart import Quart, render_template, request, jsonify, abort
import RPi.GPIO as GPIO
from threading import Thread, Lock
import face_recognition
from pyzbar.pyzbar import decode, ZBarSymbol
import board
import digitalio
import adafruit_st7789
import displayio
from adafruit_display_text import label
from adafruit_bitmap_font import bitmap_font
from adafruit_display_shapes.rect import Rect
import smbus
from heapq import heappop, heappush
import asyncio  
import av
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
import socket
import logging
import Adafruit_BMP.BMP280 as BMP280
import VL53L0X
from PIL import ImageFont
from openvino.runtime import Core
from openvino.runtime import AsyncInferQueue
from simple_pid import PID
from collections import deque
import signal
import sys
from mpu6050 import mpu6050
from pykalman import KalmanFilter
import hmac
import hashlib
selected_color_range = {"lower": None, "upper": None}
# Global state variables
current_mode = None
target_object = ""

authorized_clients = set()
AUTH_PASSWORD = "Zylo@Cobot2025"
SECRET_KEY = b"SuperSecretKey123!"

kalman_speed_x = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
kalman_speed_y = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
kalman_speed_z = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
kalman_object = KalmanFilter(initial_state_mean=np.zeros(2), n_dim_obs=2)

# Factory servos (GPIO 18 and 19)
servo_factory1 = GPIO.PWM(19, 50)  # ID 1 → GPIO 19
servo_factory2 = GPIO.PWM(18, 50)  # ID 2 → GPIO 18
servo_factory1.start(2.5)  # Start at 0°
servo_factory2.start(2.5)
GPIO.setup(19, GPIO.OUT)
GPIO.setup(18, GPIO.OUT)


GYRO_Z_OFFSET = 0.0

results_lock = Lock()

try:
    servo_pan = GPIO.PWM(16, 50)
    servo_tilt = GPIO.PWM(13, 50)
    servo_pan.start(7.5)
    servo_tilt.start(7.5)
except Exception as e:
    logging.error(f"Servo initialization error: {e}")
# Add with other global variables
current_servo1_angle = 90  # Tracks first servo (GPIO 19)
current_servo2_angle = 90

imu = mpu6050(0x68)  # Default I2C address for MPU6050
GPIO.setmode(GPIO.BCM)
video_track = None
yolo_enabled = False

JSN_TRIG_PIN = 23
JSN_ECHO_PIN = 24
GPIO.setup(JSN_TRIG_PIN, GPIO.OUT)
GPIO.setup(JSN_ECHO_PIN, GPIO.IN)
GPIO.output(JSN_TRIG_PIN, GPIO.LOW)

tracker_lock = Lock()
results_queue = deque(maxlen=1)  # Shared queue to hold inference results

def inference_callback(request, userdata):
    output = request.get_output_tensor(output_layer.index).data
    with results_lock:
        results_queue.append(output)

def calibrate_gyro(samples=500):
    global GYRO_Z_OFFSET     
    print("Calibrating gyro... Keep IMU stationary!")
    offsets = np.zeros(3)
    for _ in range(samples):
        _, _, _, gx, gy, gz = read_mpu6050()
        offsets += [gx, gy, gz]
        time.sleep(0.01)
    GYRO_Z_OFFSET = offsets[2] / samples  # Store as global
    print(f"Gyro Z offset: {GYRO_Z_OFFSET:.1f} LSB")

calibrate_gyro()  

class SensorFusionSystem:
    def __init__(self):
        self.vl = VL53L0X_Sensor()
        self.jsn = JSN_SR04T_Sensor(JSN_TRIG_PIN, JSN_ECHO_PIN)
        self.last_valid = float('inf')
        self.crossover = 0.8  # Sensor crossover point in meters

    def get_distance(self):
        d_vl = self.vl.read_distance()
        d_jsn = self.jsn.read_distance()
        
        # Raw sensor fusion logic
        if d_vl <= self.crossover:
            f_vl = d_vl  # Trust VL53L0X at close range
        else:
            f_vl = float('inf')
            
        if d_jsn >= self.crossover - 0.1:
            f_jsn = d_jsn  # Trust JSN-SR04T at longer range
        else:
            f_jsn = float('inf')
            
        # Fuse valid measurements
        if f_vl != float('inf') and f_jsn != float('inf'):
            weight = min(1.0, max(0.0, (self.crossover + 0.2 - f_jsn) / 0.4))
            fused = weight * f_vl + (1 - weight) * f_jsn
        elif f_vl != float('inf'):
            fused = f_vl
        elif f_jsn != float('inf'):
            fused = f_jsn
        else:
            fused = self.last_valid * 1.1
            fused = float('inf') if fused > 5 else fused
            
        if fused != float('inf'):
            self.last_valid = fused
            
            # Convert to world coordinates using current yaw
            with ekf_lock:
                # Get current orientation from EKF (radians)
                yaw = ekf.state[8]  
                # Project lidar measurement to world frame
                x = fused * math.cos(yaw)
                y = fused * math.sin(yaw)
                ekf.update({'lidar': (x, y, 0)})  # ✅ Correct world coordinates
        
        logging.info(f"Fusion: VL={d_vl:.3f}m JSN={d_jsn:.3f}m → {fused:.3f}m")
        return fused
# ---------------------------
# Kalman Filter Classes
# ---------------------------
class CobotEKF:
    def __init__(self):
        # State: [x, y, z, vx, vy, vz, roll, pitch, yaw]
        self.state = np.zeros(9)
        self.covariance = np.eye(9) * 0.1
        
        # Sensor noise variances (diagonal entries)
        self.R_lidar = [0.1, 0.1, 0.1]       # x, y, z
        self.R_imu = [0.05, 0.05, 0.01]      # roll, pitch, yaw
        self.R_odom = [0.5, 0.5, 0.5]        # vx, vy, vz
        
        # Process noise Q (unchanged)
        self.Q = np.diag([
            0.01, 0.01, 0.01,     # Position
            0.1, 0.1, 0.1,        # Velocity
            0.01, 0.01, 0.005     # Orientation (roll,pitch,yaw)
        ])
        
    def predict(self, dt, gyro_z_rads=None):
        # 1. State transition matrix (F) for linear motion
        F = np.eye(9)  # 9x9 for 9 state variables
        F[0:3, 3:6] = np.eye(3) * dt  # Position-velocity coupling

        # 2. Predict new state: linear part
        self.state = F @ self.state

        # 3. Integrate yaw using gyro_z if provided
        if gyro_z_rads is not None:
            self.state[8] += gyro_z_rads * dt  # Yaw = Yaw + ωz * dt

            # Optional: wrap yaw angle to [-π, π] to avoid overflow
            self.state[8] = (self.state[8] + np.pi) % (2 * np.pi) - np.pi

        # 4. Predict new covariance
        self.covariance = F @ self.covariance @ F.T + self.Q

    def update(self, measurements):
        H = np.zeros((0, 9))  # Measurement Jacobian
        z = np.zeros(0)        # Measurement vector
        R_diag = []            # Diagonal entries of R
        
        # Lidar updates position (x,y,z)
        if 'lidar' in measurements:
            lidar_x, lidar_y, lidar_z = measurements['lidar']
            H = np.vstack([H, np.array([
                [1,0,0,0,0,0,0,0,0],  # x
                [0,1,0,0,0,0,0,0,0],   # y
                [0,0,1,0,0,0,0,0,0]    # z
            ])])
            z = np.concatenate([z, [lidar_x, lidar_y, lidar_z]])
            R_diag.extend(self.R_lidar)  # Add lidar variances
        
        # IMU updates orientation (roll,pitch,yaw)
        if 'imu' in measurements:
            roll, pitch, yaw = measurements['imu']
            H = np.vstack([H, np.array([
                [0,0,0,0,0,0,1,0,0],  # roll
                [0,0,0,0,0,0,0,1,0],   # pitch
                [0,0,0,0,0,0,0,0,1]    # yaw
            ])])
            z = np.concatenate([z, [roll, pitch, yaw]])
            R_diag.extend(self.R_imu)  # Add IMU variances
        
        # Odometry updates velocity (vx, vy, vz)
        if 'odom' in measurements:
            vx, vy, vz = measurements['odom']
            H = np.vstack([H, np.array([
                [0,0,0,1,0,0,0,0,0],  # vx
                [0,0,0,0,1,0,0,0,0],   # vy
                [0,0,0,0,0,1,0,0,0]    # vz
            ])])
            z = np.concatenate([z, [vx, vy, vz]])
            R_diag.extend(self.R_odom)  # Add odometry variances
        
        # Build R dynamically
        R = np.diag(R_diag) if R_diag else np.zeros((0, 0))
        
        # Kalman update equations
        if H.shape[0] > 0:
            y = z - H @ self.state
            S = H @ self.covariance @ H.T + R  # ✅ Correct dimension
            K = self.covariance @ H.T @ np.linalg.inv(S)
            self.state += K @ y
            self.covariance = (np.eye(9) - K @ H) @ self.covariance
# ---------------------------
# Global Kalman Filter Instances
# ---------------------------
ekf = CobotEKF()  # Single EKF instance
ekf_lock = Lock()  # For thread safety

initial_yaw = math.radians(0)  # Convert to radians
ekf.state[8] = initial_yaw

# Set higher initial covariance for yaw if uncertain
ekf.covariance[8,8] = 0.5  #

# --------------------------------------------
# Logging and Locks
# --------------------------------------------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
state_lock = Lock()
camera_lock = Lock()

# --------------------------------------------
# Global Variables for Control
# --------------------------------------------
pcs = set()  # Active WebRTC connections
data_channel = None

speed = 50               # Motor speed (0-100)
current_motor_speed = 0  # Simulated motor speed for ramping
current_mode = None
target_object = ""
cobot_active = False
current_angle = 0.0      # Yaw angle (degrees) from MPU6050
qr_mapping = {}

bmp280 = BMP280.BMP280(busnum=1, address=0x76)

# --------------------------------------------
# Global PID Controllers & Reset Helper
# --------------------------------------------
class AutoTuner:
    def __init__(self, output_range=(-80, 80)):  # Constrained output range
        self.output_range = output_range
        self.oscillations = deque(maxlen=100)
        self.last_output = 0
        self.last_crossing = None
        self.peak_times = deque(maxlen=5)
        self.hysteresis = 0.1
        
    def relay_control(self, process_value, setpoint):
        """Safety-constrained relay control"""
        if process_value < setpoint - self.hysteresis:
            output = self.output_range[1]
        elif process_value > setpoint + self.hysteresis:
            output = self.output_range[0]
        else:
            output = self.last_output
            
        # Ramp output changes for smoother transitions
        output = np.clip(output, 
                        self.last_output - 10, 
                        self.last_output + 10)
        self.last_output = output
        return output
    
    def analyze_oscillations(self, process_value, setpoint):
        """Analyzes stability with multiple safety checks"""
        self.oscillations.append(process_value)
        
        # Detect zero crossings safely
        if len(self.oscillations) > 1:
            prev_val = self.oscillations[-2]
            curr_val = self.oscillations[-1]
            
            if (prev_val < setpoint <= curr_val) or (prev_val > setpoint >= curr_val):
                now = time.time()
                if self.last_crossing:
                    self.peak_times.append(now - self.last_crossing)
                self.last_crossing = now
        
        # Only calculate parameters if we have clean oscillation data
        if len(self.peak_times) >= 3 and len(self.oscillations) > 10:
            stable_periods = np.abs(np.diff(list(self.peak_times)))
            if np.max(stable_periods) > 2*np.min(stable_periods):
                return None  # Reject unstable oscillations
                
            amplitude = (max(self.oscillations) - min(self.oscillations)) / 2
            Pu = np.median(self.peak_times)  # More robust than mean
            Ku = (4 * self.hysteresis) / (np.pi * amplitude)
            
            # Conservative Ziegler-Nichols with safety limits
            Kp = min(0.4 * Ku, 5.0)  # Max Kp = 5.0
            Ki = min(0.8 * Ku / Pu, 0.9)  # Max Ki = 0.9
            Kd = min(0.15 * Ku * Pu, 2.0)  # Max Kd = 2.0
            
            return Kp, Ki, Kd
        
        return None

class HybridPIDController:
    def __init__(self, initial_Kp=0.5, initial_Ki=0.01, initial_Kd=0.05):
        self.pid = PID(initial_Kp, initial_Ki, initial_Kd, setpoint=0)
        self.pid.output_limits = (-100, 100)
        self.pid.sample_time = 0.01
        self.auto_tuner = AutoTuner()
        self.last_tune_time = time.time()
        self.error_history = deque(maxlen=20)
        self.output_history = deque(maxlen=20)
        self.adaptation_active = False
        self.stable_params = (initial_Kp, initial_Ki, initial_Kd)
        self.last_stable_time = time.time()
        self.adaptation_lock = Lock()  # ✅ Thread safety lock added

        # Safety thresholds
        self.max_error = 5.0
        self.max_oscillation = 2.0
        self.max_output = 93  # 80% max motor speed

    def update(self, error, dt):
        """Main update with safety checks"""
        error = np.clip(error, -self.max_error, self.max_error)
        self.error_history.append(error)

        # Safety check 1: Divergence detection
        if len(self.error_history) > 3:
            error_rate = abs(self.error_history[-1] - self.error_history[-3]) / dt
            if error_rate > 15:
                self._emergency_reset()
                return 0

        # Safety check 2: Excessive oscillation
        if len(self.error_history) > 10:
            oscillation = max(self.error_history) - min(self.error_history)
            if oscillation > self.max_oscillation:
                self._trigger_safe_tune()

        # Normal PID update with output limiting
        output = self.pid(error)
        output = np.clip(output, -self.max_output, self.max_output)
        self.output_history.append(output)

        # Adaptive tuning if enabled
        if self.adaptation_active:
            self._adaptive_adjustment()

        return output

    def _emergency_reset(self):
        logging.error("PID DIVERGENCE DETECTED! Resetting to stable params.")
        self.pid.tunings = self.stable_params
        self.error_history.clear()
        self.output_history.clear()

    def _trigger_safe_tune(self):
        """Safe auto-tuning with thread safety"""
        with self.adaptation_lock:  # ✅ Lock protects this section
            if time.time() - self.last_tune_time < 15.0:
                return

            logging.warning("Stability compromised - initiating safe auto-tune")

            prev_active = self.adaptation_active
            self.adaptation_active = False

            if self._auto_tune():
                self.last_tune_time = time.time()
                if time.time() - self.last_stable_time > 30.0:
                    self.stable_params = self.pid.tunings
                    self.last_stable_time = time.time()

            self.adaptation_active = prev_active

    def _auto_tune(self):
        """Protected auto-tuning for PID (rotation or distance) with safety constraints"""
        try:
            # Determine setpoint and feedback provider
            if hasattr(self, 'is_distance_pid') and self.is_distance_pid:
                setpoint = 0.5  # e.g., desired center distance in meters
                pv_callback = lambda: slam.get_center_distance()

            elif hasattr(self, 'is_rotation_pid') and self.is_rotation_pid:
                # Use EKF yaw in degrees
                with ekf_lock:
                    initial_yaw = math.degrees(ekf.state[8])
                setpoint = (initial_yaw + 30) % 360
                pv_callback = lambda: math.degrees(ekf.state[8]) % 360

            else:
                logging.warning("PID type not identified; skipping auto-tune")
                return False

            # Tuning loop
            start_time = time.time()
            while time.time() - start_time < 10.0:
                pv = pv_callback()
                error = setpoint - pv

                # Normalize angle error to [-180, 180] for rotation tuning
                if hasattr(self, 'is_rotation_pid') and self.is_rotation_pid:
                    error = (error + 180) % 360 - 180

                output = self.auto_tuner.relay_control(pv, setpoint)
                direction = None

                # Send constrained command based on PID type
                if hasattr(self, 'is_distance_pid') and self.is_distance_pid:
                    direction = "forward" if output > 0 else "backward"

                elif hasattr(self, 'is_rotation_pid') and self.is_rotation_pid:
                    direction = "right" if output > 0 else "left"

                if direction:
                    constrained_motor_control(direction, abs(output))

                # Check for convergence
                params = self.auto_tuner.analyze_oscillations(pv, setpoint)
                if params:
                    self.pid.tunings = params
                    manual_control("stop")
                    logging.info(f"PID auto-tuning complete. Tunings: {params}")
                    return True

                time.sleep(0.01)

        except Exception as e:
            logging.error(f"Auto-tuning failed: {str(e)}")

        manual_control("stop")
        return False

    def _adaptive_adjustment(self):
        """Safe, gradual parameter adjustment"""
        if len(self.error_history) < 10:
            return

        avg_error = np.mean(np.abs(list(self.error_history)[-10:]))
        oscillation = max(self.error_history) - min(self.error_history)

        if oscillation > 1.5:
            self.pid.Kp *= 0.95
            self.pid.Kd *= 1.05
            self.pid.Ki *= 0.95

        elif avg_error > 1.0:
            self.pid.Kp = min(self.pid.Kp * 1.05, 2.0)
            self.pid.Ki = min(self.pid.Ki * 1.02, 0.5)

pid_distance = HybridPIDController(1.0, 0.03, 0.1)
pid_rotation = HybridPIDController(1.2, 0.04, 0.12)
pid_lateral  = HybridPIDController(0.9, 0.025, 0.08)
pid_distance.is_distance_pid = True  # ✅ Mark as distance PID
pid_rotation.is_rotation_pid = True  # ✅ Mark as rotation PID

def constrained_motor_control(command, speed_value):
    """Motor control with safety limits"""
    MAX_SAFE_SPEED = 80  # 80% max duty cycle
    MAX_ACCEL_STEP = 10  # Max 10% change per call
    
    # Clamp speed and acceleration
    speed_value = np.clip(speed_value, 0, MAX_SAFE_SPEED)
    
    # Smooth acceleration
    if hasattr(constrained_motor_control, 'last_speed'):
        speed_value = np.clip(speed_value,
                            constrained_motor_control.last_speed - MAX_ACCEL_STEP,
                            constrained_motor_control.last_speed + MAX_ACCEL_STEP)
    
    constrained_motor_control.last_speed = speed_value
    manual_control(command, speed_value=speed_value)

def reset_pid():
    pid_distance.pid.reset()
    pid_rotation.pid.reset()
    pid_lateral.pid.reset()
    logging.info("All PID controllers reset")
# --------------------------------------------
# MPU6050: Initialization and Yaw Update (Replaced with Kalman Filter)
# --------------------------------------------

bus = smbus.SMBus(1)
MPU6500_ADDR = 0x68  # Same as MPU6050

try:
    bus.write_byte_data(MPU6500_ADDR, 0x6B, 0x00)
    time.sleep(0.1)  # Small delay for safety
    bus.write_byte_data(MPU6500_ADDR, 0x6A, 0x00)
    bus.write_byte_data(MPU6500_ADDR, 0x1B, 0x00)
    bus.write_byte_data(MPU6500_ADDR, 0x1C, 0x00)
    logging.info("MPU6500 initialization completed successfully.")

except Exception as e:
    logging.error(f"MPU6500 initialization error: {e}")


def read_word_2c(adr):
    try:
        high = bus.read_byte_data(MPU6050_ADDR, adr)
        low = bus.read_byte_data(MPU6050_ADDR, adr+1)
        val = (high << 8) + low
        return -((65535 - val) + 1) if val >= 0x8000 else val
    except Exception as e:
        logging.error(f"MPU6050 I2C error: {e}")
        return 0  # Default safe value

def read_mpu6050():
    try:
        # Read all 6 DOF (accel x,y,z + gyro x,y,z)
        accel_x = read_word_2c(0x3B) / 16384.0
        accel_y = read_word_2c(0x3D) / 16384.0 
        accel_z = read_word_2c(0x3F) / 16384.0
        gyro_x = read_word_2c(0x43) / 131.0
        gyro_y = read_word_2c(0x45) / 131.0
        gyro_z = read_word_2c(0x47) / 131.0

        gyro_z -= GYRO_Z_OFFSET
        
        return (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
    except Exception as e:
        logging.error(f"MPU6050 read error: {e}")
        return (0, 0, 0, 0, 0, 0)  # Safe default

def update_current_angle():
    global current_angle
    prev_time = time.time()
    
    while True:
        now = time.time()
        dt = now - prev_time
        prev_time = now

        try:
            # Read IMU data (including raw gyro Z for yaw rate)
            accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = read_mpu6050()
            
            # Calculate orientation from accelerometer (roll/pitch)
            roll = math.atan2(accel_y, accel_z)
            pitch = math.atan2(-accel_x, math.sqrt(accel_y**2 + accel_z**2))
            
            # Convert gyro Z to rad/s (MPU6050 default: 131 LSB/°/s → 250 deg/s range)
            gyro_z_rads = math.radians(gyro_z)  # Now in rad/s
            
            # Update EKF - critical changes:
            with ekf_lock:
                ekf.predict(dt, gyro_z_rads=gyro_z_rads)
                ekf.update({
                    'imu': (roll, pitch, gyro_z_rads),  # Note: Passing yaw RATE, not absolute yaw
                    'odom': (gyro_x, gyro_y, gyro_z)    # Gyros as angular velocity inputs
                })
                
                # Get fused yaw estimate from state
                current_angle = math.degrees(ekf.state[8])  # Convert back to degrees
                
        except Exception as e:
            logging.error(f"Yaw update error: {e}")
            time.sleep(0.01)


# --------------------------------------------
# Full SLAM Class Integrating Lidar, MPU6050, and Camera
# --------------------------------------------
class SLAM:
    def __init__(self, grid_size=100, cell_size=0.1):
        self.position = (0.0, 0.0)  # (x, y) in meters
        self.theta = 0.0            # Yaw angle (degrees)
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.map = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        self.origin = (grid_size // 2, grid_size // 2)
        self.landmarks = {'p1': (1.0, 2.0), 'p2': (3.0, 1.5)}
        self.starting_pose = None

    def update_position(self):
        dt = 0.1  # Time step
        with state_lock:
            self.theta = current_angle
        v = (current_motor_speed / 100.0) * 0.1
        dx = v * math.cos(math.radians(self.theta)) * dt
        dy = v * math.sin(math.radians(self.theta)) * dt
        self.position = (self.position[0] + dx, self.position[1] + dy)
        return self.position

    def update_map_with_lidar(self):
        if sensor_fusion is None:
            return self.map
        center_distance = get_center_distance()
        if center_distance > 0.3:
            return self.map
        scan_angles = range(-45, 46, 5)
        with state_lock:
            base_theta = current_angle
        for offset in scan_angles:
            self._scan_sector(base_theta, offset)
        return self.map

    def _scan_sector(self, base_theta, offset):
        try:
            angle = 90 + offset
            lidar_servo.ChangeDutyCycle(safe_servo_duty(angle))
            time.sleep(0.05)
            distance_mm = sensor_fusion.get_distance()
            if distance_mm <= 0:
                return
            d = distance_mm / 1000.0
            total_angle = math.radians(base_theta + offset)
            obs_x = self.position[0] + d * math.cos(total_angle)
            obs_y = self.position[1] + d * math.sin(total_angle)
            grid_x = int(self.origin[0] + (obs_x / self.cell_size))
            grid_y = int(self.origin[1] - (obs_y / self.cell_size))
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                self.map[grid_y][grid_x] = 1
                self._clear_neighbors(grid_x, grid_y)
        except Exception as e:
            logging.error(f"Scan error at {offset}°: {e}")
                   
    def get_center_distance(self):
        try:
            lidar_servo.ChangeDutyCycle(safe_servo_duty(90))
            time.sleep(0.1)
            return sensor_fusion.get_distance()
        except Exception as e:
            logging.error(f"Error in get_center_distance: {e}")
            return float('inf')

    def update_map(self):
        self.update_map_with_lidar()
        return self.map

    def detect_obstacle(self):
        return get_center_distance() < 0.3

    def update_with_camera(self, qr_data):
        if qr_data in self.landmarks:
            self.position = self.landmarks[qr_data]
        return self.position

    def get_qr_position(self, qr_data):
        try:
            num = int(qr_data[1:])
            coord = ((num * 5) % 10, (num * 5) % 10)
            return coord
        except Exception as e:
            logging.error(f"QR position parse error: {e}")
            return self.position

    def get_grid_coords(self, world_coords):
        x, y = world_coords
        grid_x = int(self.origin[0] + (x / self.cell_size))
        grid_y = int(self.origin[1] - (y / self.cell_size))
        return (grid_x, grid_y)

    def get_world_coords(self, grid_coords):
        grid_x, grid_y = grid_coords
        x = (grid_x - self.origin[0]) * self.cell_size
        y = (self.origin[1] - grid_y) * self.cell_size
        return (x, y)

slam = SLAM()

last_imu_time = time.time()
# --------------------------------------------
# YOLO Model and Font Loading
# --------------------------------------------
def init_ov_model():
    core = Core()
    core.set_property("CPU", {"CPU_THREADS_NUM": 4, "CPU_BIND_THREADS": "YES"})
    model = core.read_model("yolov11n.xml", "yolov11n.bin")  # Now loading FP16 IR model
    compiled_model = core.compile_model(model, "AUTO")
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    N, C, H, W = input_layer.shape
    return compiled_model, input_layer, output_layer, (W, H)
yolo_model, input_layer, output_layer, (model_w, model_h) = init_ov_model()

try:
    # Load the built-in default font (removes dependency on external fonts)
    font = ImageFont.load_default()
    logging.info("Default font loaded successfully.")
except Exception as e:
    logging.error(f"Error loading default font: {e}")
    font = None

yolo_classes = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
    19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
    24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase",
    29: "frisbee", 30: "skis", 31: "snowboard", 32: "sports ball",
    33: "kite", 34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "TV",
    63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster",
    71: "sink", 72: "refrigerator", 73: "book", 74: "clock", 75: "vase",
    76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"
}

def yolo_detection_loop(video_track):
    global yolo_enabled
    YOLO_CONFIDENCE_THRESHOLD = 0.5

    infer_queue = AsyncInferQueue(yolo_model, 2)
    infer_queue.set_callback(inference_callback)

    RES_HIGH = (model_w, model_h)
    RES_MED = (224, 224)
    RES_LOW = (160, 160)
    CLOSE_DISTANCE = 1.0
    LOST_THRESHOLD = 5
    RES_CHANGE_COOLDOWN = 30
    last_res_change = time.time()
    target_lost_counter = 0
    tracker = None

    def letterbox_resize(frame, target_size):
        h, w = frame.shape[:2]
        scale = min(target_size[1]/h, target_size[0]/w)
        new_h, new_w = int(h*scale), int(w*scale)
        resized = cv2.resize(frame, (new_w, new_h))
        pad_w = (target_size[0] - new_w) // 2
        pad_h = (target_size[1] - new_h) // 2
        return resized, scale, (pad_w, pad_h)

    while True:
        if not yolo_enabled:
            tracker = None
            time.sleep(0.1)
            continue

        with camera_lock:
            frame = video_track.get_latest_frame()

        if frame is None:
            time.sleep(0.01)
            continue

        current_res = RES_HIGH
        if sensor_fusion and (time.time() - last_res_change > RES_CHANGE_COOLDOWN):
            distance = sensor_fusion.get_distance()
            if distance < CLOSE_DISTANCE:
                current_res = RES_MED
                last_res_change = time.time()
            elif target_lost_counter >= LOST_THRESHOLD:
                current_res = RES_LOW
                last_res_change = time.time()

        resized, scale, (pad_w, pad_h) = letterbox_resize(frame, current_res)
        input_data = np.expand_dims(resized.transpose(2, 0, 1), 0).astype(np.float32) / 255.0

        infer_queue.start_async({input_layer.any_name: input_data})
        infer_queue.wait_all()

        with results_lock:
            if results_queue:
                results = results_queue[-1]
                detections = np.squeeze(results, 0).T
                target_detected = False

                for obj in detections:
                    confidence = obj[4]
                    if confidence < YOLO_CONFIDENCE_THRESHOLD:
                        continue

                    xmin = int((obj[0] - pad_w)/scale)
                    ymin = int((obj[1] - pad_h)/scale)
                    xmax = int((obj[2] - pad_w)/scale)
                    ymax = int((obj[3] - pad_h)/scale)

                    xmin = max(0, min(xmin, frame.shape[1]))
                    ymin = max(0, min(ymin, frame.shape[0]))
                    xmax = max(0, min(xmax, frame.shape[1]))
                    ymax = max(0, min(ymax, frame.shape[0]))

                    class_id = int(obj[5])
                    label = yolo_classes.get(class_id, "unknown")

                    # Always draw all detections
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
                    cv2.putText(frame, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    # Special handling for person tracking with color
                    if target_object and label.lower() == target_object.lower():
                        person_crop = frame[ymin:ymax, xmin:xmax]
                        if selected_color_range["lower"] is not None and person_crop.size > 0:
                            hsv_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
                            lower_hsv = cv2.cvtColor(np.uint8([[selected_color_range["lower"]]]), cv2.COLOR_RGB2HSV)[0][0]
                            upper_hsv = cv2.cvtColor(np.uint8([[selected_color_range["upper"]]]), cv2.COLOR_RGB2HSV)[0][0]
                            mask = cv2.inRange(hsv_crop, lower_hsv, upper_hsv)
                            coverage = np.sum(mask > 0) / mask.size

                            if coverage > 0.15:
                                target_detected = True
                                bbox = (xmin, ymin, xmax - xmin, ymax - ymin)

                                with tracker_lock:
                                    if tracker is None:
                                        tracker = cv2.TrackerKCF_create()
                                        tracker.init(frame, bbox)
                                    else:
                                        ok, _ = tracker.update(frame)
                                        if not ok:
                                            tracker = None

                                if current_mode == "follow_object":
                                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                                    cv2.putText(frame, f"{label}", (xmin, ymin - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Tracker fallback if no fresh detection
            if not target_detected and tracker:
                ok, bbox = tracker.update(frame)
                if ok:
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    if current_mode == "follow_object":
                        # Tracker still active
                        target_detected = True
                else:
                    tracker = None
                    target_lost_counter = LOST_THRESHOLD

            # Lost target tracking counter
            target_lost_counter = 0 if target_detected else target_lost_counter + 1

        time.sleep(0.01)

try:
    sensor_fusion = SensorFusionSystem()
    logging.info("Sensor Fusion initialized globally.")
except Exception as e:
    logging.error(f"Sensor Fusion failed: {e}")
    sensor_fusion = None

class VL53L0X_Sensor:
    def __init__(self):
        try:
            self.sensor = VL53L0X.VL53L0X()
            self.sensor.start_ranging(VL53L0X.Vl53l0xAccuracyMode.BEST)
        except Exception as e:
            logging.error(f"VL53L0X init failed: {e}")
            self.sensor = None

    def read_distance(self):
        if not self.sensor:
            return float('inf')
        try:
            distance_mm = self.sensor.get_distance()
            if 0 < distance_mm <= 8190:
                return distance_mm / 1000.0  # meters
            else:
                return float('inf')
        except Exception as e:
            logging.error(f"VL53L0X read error: {e}")
            return float('inf')

class JSN_SR04T_Sensor:
    def __init__(self, trig_pin, echo_pin):
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.speed_of_sound = 343.2  # m/s

    def read_distance(self):
        # Send trigger pulse
        GPIO.output(self.trig_pin, True)
        time.sleep(0.00001)  # 10μs pulse
        GPIO.output(self.trig_pin, False)

        timeout_start = time.time()
        
        # Wait for echo to go HIGH (start of pulse)
        while GPIO.input(self.echo_pin) == 0:
            if time.time() - timeout_start > 0.1:  # 100ms timeout
                return float('inf')
        pulse_start = time.time()

        # Wait for echo to go LOW (end of pulse)
        while GPIO.input(self.echo_pin) == 1:
            if time.time() - timeout_start > 0.1:  # 100ms timeout
                return float('inf')
        pulse_end = time.time()

        # Calculate distance
        duration = pulse_end - pulse_start
        distance = (duration * self.speed_of_sound) / 2.0  # Divide by 2 for round-trip

        # Validate within sensor range (20cm to 4.5m)
        return distance if 0.2 <= distance <= 4.5 else float('inf')

# --------------------------------------------
# Lidar Sensor VL53L0X) Initialization and Setup
# --------------------------------------------
LIDAR_SERVO_PIN = 17
GPIO.setup(LIDAR_SERVO_PIN, GPIO.OUT)
def safe_servo_duty(angle):
    angle = max(0, min(180, angle))
    return 2.5 + (angle / 180.0) * 10
lidar_servo = GPIO.PWM(LIDAR_SERVO_PIN, 50)
lidar_servo.start(safe_servo_duty(90))  # Start at 90° (center)

# --------------------------------------------
# WebRTC: Custom Video Track from Camera
# --------------------------------------------
class CameraVideoTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.last_frame = None
        self.lock = Lock()
        
    async def recv(self):
        pts, time_base = await self.next_timestamp()
        loop = asyncio.get_event_loop()
        
        # Get frame using executor (maintains your async pattern)
        with camera_lock:
            frame = await loop.run_in_executor(
                None, 
                lambda: camera.capture_array("lores")  # Use lores stream
            )
        
        # Store original frame (now 640x480 from lores stream)
        with self.lock:
            self.last_frame = frame.copy()
            
        # No need to resize since we're using lores stream
        av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        av_frame.pts = pts
        av_frame.time_base = time_base
        return av_frame

    def get_latest_frame(self):
        with self.lock:
            return self.last_frame.copy() if self.last_frame is not None else None


# --------------------------------------------
# Hardware Setup: Motors, Servos, Display, Camera
# --------------------------------------------
PWMA, AIN1, BIN1 = 12, 20, 21
PWMB, AIN2, BIN2 = 14, 22, 26
for pin in [PWMA, AIN1, BIN1, PWMB, AIN2, BIN2]:
    GPIO.setup(pin, GPIO.OUT)
pwmA = GPIO.PWM(PWMA, 1000)
pwmB = GPIO.PWM(PWMB, 1000)
pwmA.start(0)
pwmB.start(0)

SD_CARD_PATH = "/mnt/sdcard/captured_images"
if not os.path.exists(SD_CARD_PATH):
    os.makedirs(SD_CARD_PATH)
app = Quart(__name__)

try:
    camera = Picamera2()
    config = camera.create_still_configuration(
        main={"size": (3280, 2464)},  # High res for photos
        lores={"size": (640, 480)},    # Low res for video
        display="lores"               # Display the video stream
    )
    camera.configure(config)
    camera.start()
    video_track = CameraVideoTrack()  # Initialize after camera starts
except Exception as e:
    logging.error(f"Camera initialization error: {e}")
    camera = None
    video_track = None
    
try:
    displayio.release_displays()
    spi = board.SPI()
    tft_cs = digitalio.DigitalInOut(board.CE0)
    tft_dc = digitalio.DigitalInOut(board.D25)
    display_bus = displayio.FourWire(spi, command=tft_dc, chip_select=tft_cs)
    display = adafruit_st7789.ST7789(display_bus, width=128, height=128, rowstart=80)
    splash = displayio.Group()
    display.show(splash)
    background = Rect(0, 0, 128, 128, fill=0x000000)
    splash.append(background)
except Exception as e:
    logging.error(f"TFT display initialization error: {e}")
recognizer = sr.Recognizer()
engine = pyttsx3.init()

def get_fused_tilt(imu):
    global last_imu_time

    # 1. Read raw sensor data
    accel = imu.get_accel_data()
    gyro = imu.get_gyro_data()

    accel_x = accel['x']
    accel_y = accel['y']
    accel_z = accel['z']

    gyro_x = gyro['x']
    gyro_y = gyro['y']
    gyro_z = gyro['z']

    # 2. Calculate accelerometer angles
    acc_roll = math.atan2(accel_y, accel_z) * 180.0 / math.pi
    acc_pitch = math.atan2(-accel_x, math.sqrt(accel_y**2 + accel_z**2)) * 180.0 / math.pi

    # 3. Time delta
    current_time = time.time()
    dt = current_time - last_imu_time
    last_imu_time = current_time

    # 4. Fuse data using Kalman filters
    with ekf_lock:
        ekf.update({'imu': (math.radians(acc_roll), math.radians(acc_pitch))})
        fused_roll = math.degrees(ekf.state[6])  # Roll is state[6]
        fused_pitch = math.degrees(ekf.state[7])  # Pitch is state[7]

    return fused_roll, fused_pitch

# --------------------------------------------
# A* Path Planning Using Euclidean Heuristic
# --------------------------------------------
def a_star(grid, start, goal):
    def heuristic(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    while open_set:
        _, current = heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        for dx, dy in movements:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]):
                if grid[neighbor[1]][neighbor[0]] == 1:
                    continue
                tentative_g = g_score[current] + math.sqrt(dx**2 + dy**2)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))
    return []

# --------------------------------------------
# Utility Functions for Display and Motor Control
# --------------------------------------------
def display_text(text):
    while len(splash) > 1:
        splash.pop()
    if font:
        txt = label.Label(font, text=text, color=0xFFFFFF, x=10, y=64)
        splash.append(txt)
    else:
        logging.info(text)

def reset_current_angle():
    global current_angle
    with state_lock:
        current_angle = 0.0
    logging.info("Current angle reset to 0.0")

def ramp_motor_speed(command, target_speed):
    global current_motor_speed
    step = 5
    delay = 0.05
    start_time = time.time()
    while abs(current_motor_speed - target_speed) > 0.1:
        if time.time() - start_time > 5:
            logging.error("Timeout in ramp_motor_speed")
            break
        with state_lock:
            if current_motor_speed < target_speed:
                current_motor_speed = min(current_motor_speed + step, target_speed)
            else:
                current_motor_speed = max(current_motor_speed - step, target_speed)
        manual_control(command, speed_value=current_motor_speed)
        time.sleep(delay)

def manual_control(command, speed_value=None):
        global speed, current_motor_speed
        duty = speed_value if speed_value is not None else speed
        if command == "forward":
            GPIO.output(AIN1, GPIO.HIGH)
            GPIO.output(BIN1, GPIO.LOW)
            GPIO.output(AIN2, GPIO.HIGH)
            GPIO.output(BIN2, GPIO.LOW)
            pwmA.ChangeDutyCycle(duty)
            pwmB.ChangeDutyCycle(duty)
        elif command == "backward":
            GPIO.output(AIN1, GPIO.LOW)
            GPIO.output(BIN1, GPIO.HIGH)
            GPIO.output(AIN2, GPIO.LOW)
            GPIO.output(BIN2, GPIO.HIGH)
            pwmA.ChangeDutyCycle(duty)
            pwmB.ChangeDutyCycle(duty)
        elif command == "left":
            GPIO.output(AIN1, GPIO.LOW)
            GPIO.output(BIN1, GPIO.HIGH)
            GPIO.output(AIN2, GPIO.HIGH)
            GPIO.output(BIN2, GPIO.LOW)
            pwmA.ChangeDutyCycle(duty)
            pwmB.ChangeDutyCycle(duty)
        elif command == "right":
            GPIO.output(AIN1, GPIO.HIGH)
            GPIO.output(BIN1, GPIO.LOW)
            GPIO.output(AIN2, GPIO.LOW)
            GPIO.output(BIN2, GPIO.HIGH)
            pwmA.ChangeDutyCycle(duty)
            pwmB.ChangeDutyCycle(duty)
        elif command == "stop":
            with state_lock:
                current_motor_speed = 0
            pwmA.ChangeDutyCycle(0)
            pwmB.ChangeDutyCycle(0)
        elif command.startswith("servo:pan:"):
            try:
                angle = int(command.split(":")[-1])
                angle = max(0, min(180, angle))
                duty_cycle = 2.5 + (angle / 180.0) * 10
                servo_pan.ChangeDutyCycle(duty_cycle)
                time.sleep(0.2)
            except Exception as e:
                logging.error(f"Error setting pan angle: {e}")
        elif command.startswith("servo:tilt:"):
            try:
                angle = int(command.split(":")[-1])
                angle = max(0, min(180, angle))
                duty_cycle = 2.5 + (angle / 180.0) * 10
                servo_tilt.ChangeDutyCycle(duty_cycle)
                time.sleep(0.2)
            except Exception as e:
                logging.error(f"Error setting tilt angle: {e}")
        elif command.startswith("servo:lidar:"):
            try:
                angle = int(command.split(":")[-1])
                angle = max(0, min(180, angle))
                duty_cycle = safe_servo_duty(angle)
                lidar_servo.ChangeDutyCycle(duty_cycle)
                time.sleep(0.2)
                lidar_servo.ChangeDutyCycle(0)
            except Exception as e:
                logging.error(f"Error setting lidar servo angle: {e}")
        if command in ["forward", "backward", "left", "right", "stop"]:
            slam.update_position()

def pid_update_with_filter(pid_controller, raw_error, dt):
    with ekf_lock:
        filtered_error = ekf.state[3]
    return pid_controller.update(filtered_error, dt)

def rotate_cobot(degrees):
    """
    Rotates the cobot by a specific number of degrees using EKF yaw (state[8]) for feedback.
    Assumes EKF state[8] holds yaw in radians.
    """
    tolerance = 2  # degrees
    min_speed = 20
    max_speed = 80

    # Calculate target yaw (in degrees)
    with ekf_lock:
        current_yaw_deg = math.degrees(ekf.state[8])
    target_yaw_deg = current_yaw_deg + degrees

    # Normalize to [-180, 180]
    def normalize(angle):
        return (angle + 180) % 360 - 180

    target_yaw_deg = normalize(target_yaw_deg)

    # PID Preparation
    pid_rotation.adaptation_active = True
    reset_pid()
    prev_time = time.time()

    while True:
        now = time.time()
        dt = now - prev_time
        prev_time = now

        # Read current yaw from EKF
        with ekf_lock:
            current_yaw_deg = math.degrees(ekf.state[8])

        # Calculate error (normalize to [-180, 180])
        error = normalize(target_yaw_deg - current_yaw_deg)

        # Exit if within tolerance
        if abs(error) < tolerance:
            pid_rotation.adaptation_active = False
            constrained_motor_control("stop", 0)
            break

        # PID update
        output = pid_update_with_filter(pid_rotation, error, dt)
        speed = np.clip(abs(output), min_speed, max_speed)
        direction = "right" if output > 0 else "left"

        # Apply motor control
        constrained_motor_control(direction, speed)

        # Loop frequency management
        elapsed = time.time() - now
        if elapsed < 0.01:
            time.sleep(0.01 - elapsed)

def add_safety_margin(grid_map, margin=1):
    """Expand obstacles with safety margin"""
    safe_map = [row.copy() for row in grid_map]
    height, width = len(grid_map), len(grid_map[0])
    
    for y in range(height):
        for x in range(width):
            if grid_map[y][x] == 1:  # Obstacle
                for dy in range(-margin, margin+1):
                    for dx in range(-margin, margin+1):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < height and 0 <= nx < width:
                            safe_map[ny][nx] = 1
    return safe_map

def get_center_distance():
    try:
        lidar_servo.ChangeDutyCycle(safe_servo_duty(90))
        time.sleep(0.1)
        return sensor_fusion.get_distance()
    except Exception as e:
        logging.error(f"Error in get_center_distance: {e}")
        return float('inf')
    
def return_to_start():
    try:
        if not hasattr(slam, 'starting_pose'):
            slam.starting_pose = slam.update_position()
            
        logging.info("Initiating safe return to start...")
        
        # Enable adaptive control
        pid_rotation.adaptation_active = True
        pid_distance.adaptation_active = True
        
        # Get path with safety margin
        safe_map = add_safety_margin(slam.map, margin=2)  # Larger margin for return
        
        start_grid = slam.get_grid_coords(slam.position)
        goal_grid = slam.get_grid_coords(slam.starting_pose)
        
        path = a_star(safe_map, start_grid, goal_grid)
        
        if not path:
            raise RuntimeError("No safe return path found")
            
        # Execute path with PID control
        for step in path:
            target = slam.get_world_coords(step)
            
            # PID-controlled movement
            while True:
                dx = target[0] - slam.position[0]
                dy = target[1] - slam.position[1]
                distance = np.hypot(dx, dy)
                
                if distance < 0.1:  # 10cm tolerance
                    break
                    
                # Distance PID
                speed = np.clip(pid_distance.update(distance, 0.1), 20, 60)
                
                # Angle PID
                target_angle = math.atan2(dy, dx)
                with state_lock:
                    angle_error = target_angle - math.radians(current_angle)
                if abs(angle_error) > 0.2:  # ~11 degrees
                    rotate_cobot(math.degrees(angle_error))
                    continue
                    
                constrained_motor_control("forward", speed)
                time.sleep(0.1)
                
        # Final orientation adjustment
        rotate_cobot(-current_angle)  # Face original orientation
        
    except Exception as e:
        logging.error(f"Return failed: {e}")
        constrained_motor_control("stop", 0)
        raise
    finally:
        pid_rotation.adaptation_active = False
        pid_distance.adaptation_active = False
        
# --------------------------------------------
# Face Recognition and Distance Measurement
# --------------------------------------------
try:
    known_image = face_recognition.load_image_file("my_face.jpg")
    known_encodings = face_recognition.face_encodings(known_image)
    if not known_encodings:
        raise ValueError("No face found in my_face.jpg")
    known_encoding = known_encodings[0]
except Exception as e:
    logging.critical(f"Face recognition setup failed: {e}")
    sys.exit(1)

# Function to check if authorized face is in frame
def face_recognization():
    try:
        with camera_lock:
            frame = video_track.get_latest_frame()
        if frame is None:
            return False

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        for face_encoding in face_encodings:
            match = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=0.5)
            if match[0]:
                logging.info("✅ Authorized face recognized! Unlocking cobot.")
                return True
    except Exception as e:
        logging.error(f"Face recognition error: {e}")
    return False

# Function to wait up to 30 seconds for face unlock
def wait_for_face_unlock(timeout=30):
    logging.info("🔒 Waiting for face authentication...")
    start = time.time()
    while time.time() - start < timeout:
        if face_recognization():
            return True
        time.sleep(1)
    return False

def measure_distance():
    try:
        if sensor_fusion is None:
            logging.error("Sensor fusion system is not available!")
            return None
        distance = sensor_fusion.get_distance()
        if distance == float('inf'):
            logging.warning("No valid fused distance reading available.")
            return None
        logging.info(f"Fused distance measured: {distance:.3f} m")
        if data_channel is not None:
            data_channel.send(f"distance:{distance:.3f}")
        return distance
    except Exception as e:
        logging.error(f"Error in measure_distance: {e}")
        return None

# --------------------------------------------
# Image and QR Code Utilities
# --------------------------------------------

def detect_qr_code():
    try:
        with camera_lock:
            frame = video_track.get_latest_frame()
            if frame is None:
                return None
                
        # Optimization 1: Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Optimization 2: Only scan for QR codes (ignore other barcode types)
        decoded = decode(gray, symbols=[ZBarSymbol.QRCODE])
        
        return decoded[0].data.decode("utf-8") if decoded else None
        
    except Exception as e:
        logging.error(f"QR detection error: {e}")
        return None

def scan_qr():
    qr = detect_qr_code()
    if qr:
        logging.info(f"QR Code detected: {qr}")
        slam.update_with_camera(qr)
    else:
        logging.info("No QR Code detected.")
    return qr

def capture_image(high_res=False):
    try:
        if high_res:
            filename = f"image_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            path = os.path.join(SD_CARD_PATH, filename)
            with camera_lock:
                camera.switch_mode_and_capture_file(
                    camera.create_still_configuration(),
                    path
                )
            return path

        else:
            # Capture from video stream (faster but lower quality)
            with camera_lock:
                frame = video_track.get_latest_frame()
            if frame is not None:
                path = os.path.join(SD_CARD_PATH, f"image_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(path, frame)
                return path
        return None
    except Exception as e:
        logging.error(f"Image capture error: {e}")
        return None

def detect_objects(frame, target_object):
    # Preprocess the frame: resize and normalize as per model input
    input_data = np.expand_dims(frame.transpose(2, 0, 1), 0).astype(np.float32) / 255.0
    results = yolo_model([input_data])[output_layer]
    detections = []
    for obj in results[0][0]:
        if obj[4] < 0.5:
            continue
        x1, y1, x2, y2 = map(int, obj[0:4] * [frame.shape[1], frame.shape[0]] * 2)
        class_id = int(obj[5])
        label_text = yolo_classes.get(class_id, "unknown")
        if target_object.lower() in label_text.lower():
            detections.append((label_text, (x1, y1, x2, y2)))
    return detections

# --------------------------------------------
# Motor Control and Obstacle Avoidance (Simulation)
# --------------------------------------------
class MotorControl:
    def move_to(self, step):
        logging.info(f"Simulating move to grid cell: {step}")
        manual_control("forward", speed_value=speed)
        time.sleep(0.3)
        manual_control("stop")

motor = MotorControl()

def stop_cobot():
    try:
        manual_control("stop")
        global current_mode
        current_mode = None
        if camera:
            camera.stop_preview()
        GPIO.cleanup()
    except Exception as e:
        logging.error(f"Error stopping cobot: {e}")

def get_side_distance(direction):
    try:
        if sensor_fusion is None:
            logging.error("Lidar sensor not available for side distance measurement.")
            return float('inf')
        if direction == 'left':
            angle = 90 + 30
        elif direction == 'right':
            angle = 90 - 30
        else:
            angle = 90
        lidar_servo.ChangeDutyCycle(safe_servo_duty(angle))
        time.sleep(0.1)
        distance_mm = sensor_fusion.get_distance()
        if distance_mm <= 0:
            return float('inf')
        d = distance_mm / 10.0
        return d
    except Exception as e:
        logging.error(f"Error getting side distance: {e}")
    return float('inf')

def execute_command(command: str):
    try:
        # -- Mode Selection --
        if command.startswith("mode:"):
            parts = command.split(":")
            mode = parts[1]

            global current_mode
            current_mode = mode  # ✅ Required for follow_mode()

            if mode == "factory_mode":
                Thread(target=factory_mode, daemon=True).start()

            elif mode == "follow_me":
                Thread(target=follow_mode, args=("follow_me",), daemon=True).start()

            elif mode == "manual_control":
                manual_control("stop")
                logging.info("Manual control ready")

            elif mode == "follow_object" and len(parts) >= 3:
                global target_object
                target_object = parts[2]
                Thread(target=follow_mode, args=("follow_object",), daemon=True).start()

        # -- Core Commands --
        elif command == "returnToStart":
            Thread(target=return_to_start, daemon=True).start()

        elif command == "scanQR":
            qr_data = scan_qr()
            if data_channel:
                data_channel.send(f"qrScanResult:{qr_data or 'No QR Found'}")

        elif command == "captureImage":
            path = capture_image()
            if data_channel and path:
                data_channel.send(f"imageSaved:{path}")

        # -- Motor Control --
        elif command.startswith("motorSpeed:"):
            parts = command.split(":")
            if len(parts) == 3:
                direction, speed = parts[1], int(parts[2])
                ramp_motor_speed(direction, speed)
            else:
                logging.warning("⚠️ Malformed motorSpeed command.")

        # -- Servo Control --
        elif command.startswith("servo:pan:"):
            angle = max(0, min(180, int(command.split(":")[2])))
            duty = 2.5 + (angle / 180.0) * 10
            servo_pan.ChangeDutyCycle(duty)

        elif command.startswith("servo:tilt:"):
            angle = max(0, min(180, int(command.split(":")[2])))
            duty = 2.5 + (angle / 180.0) * 10
            servo_tilt.ChangeDutyCycle(duty)

        elif command.startswith("servo:lidar:"):
            angle = max(0, min(180, int(command.split(":")[2])))
            duty = safe_servo_duty(angle)
            lidar_servo.ChangeDutyCycle(duty)
            time.sleep(0.2)
            lidar_servo.ChangeDutyCycle(0)

        # -- Color Control --
        elif command.startswith("updateColor:"):
            _, lower_hex, upper_hex = command.split(":")
            lower_rgb = tuple(int(lower_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            upper_rgb = tuple(int(upper_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            lower_hsv = cv2.cvtColor(np.uint8([[lower_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
            upper_hsv = cv2.cvtColor(np.uint8([[upper_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
            selected_color_range["lower"] = lower_hsv
            selected_color_range["upper"] = upper_hsv

        # -- Emergency Stop --
        elif command == "emergencyStop":
            manual_control("stop")
            reset_pid()
            GPIO.cleanup()
            sys.exit(1)

        else:
            logging.warning(f"Unhandled command: {command}")

    except Exception as e:
        logging.error(f"Command execution failed: {e}")
        try:
            if data_channel and data_channel.readyState == "open":
                data_channel.send(f"error:{str(e)}")
        except Exception as dc_err:
            logging.error(f"❌ Failed to notify frontend: {dc_err}")


def obstacle_avoidance_control():
    try:
        if not slam.detect_obstacle():
            return
            
        logging.info("Obstacle detected. Initiating avoidance maneuver...")
        
        # Enable distance PID adaptation during avoidance
        pid_distance.adaptation_active = True
        
        # Get precise distance measurements
        lidar_servo.ChangeDutyCycle(safe_servo_duty(90))  # Center lidar
        time.sleep(0.1)
        center_d = get_center_distance()
        
        left_d = get_side_distance('left')
        right_d = get_side_distance('right')
        
        logging.info(f"Distances - Center: {center_d:.2f}m, Left: {left_d:.2f}m, Right: {right_d:.2f}m")
        
        # Case 1: Narrow passage (both sides have space)
        if left_d > 0.5 and right_d > 0.5:
            logging.info("Narrow passage detected. Calculating optimal path...")
            
            # Use PID-controlled centering
            error = left_d - right_d
            correction = pid_lateral.update(error, 0.1)  # dt=0.1
            
            # Constrained motor control
            steer_speed = np.clip(abs(correction), 30, 60)
            if correction > 0:
                constrained_motor_control("left", steer_speed)
            else:
                constrained_motor_control("right", steer_speed)
                
            time.sleep(0.5)
            
        # Case 2: More space on left
        elif left_d > right_d:
            logging.info(f"Turning left (space: {left_d:.2f}m)...")
            turn_amount = min(45, 30*(left_d - right_d))  # Dynamic turn angle
            rotate_cobot(turn_amount)
            
        # Case 3: More space on right
        else:
            logging.info(f"Turning right (space: {right_d:.2f}m)...")
            turn_amount = min(45, 30*(right_d - left_d))  # Dynamic turn angle
            rotate_cobot(-turn_amount)
            
        # Final approach with PID-controlled distance
        if center_d < 1.0:  # Only if still close to obstacle
            logging.info("Final approach with distance PID control")
            while center_d < 0.8:  # Target 0.8m clearance
                error = 0.8 - center_d
                output = pid_distance.update(error, 0.1)
                speed = np.clip(output, 20, 40)
                constrained_motor_control("forward", speed)
                
                center_d = get_center_distance()
                if center_d < 0.3:  # Emergency stop
                    constrained_motor_control("stop", 0)
                    raise RuntimeError("Obstacle too close!")
                    
                time.sleep(0.1)
                
        # Resume SLAM navigation
        slam_avoid_obstacle()
        
    except Exception as e:
        logging.error(f"Avoidance error: {e}")
        constrained_motor_control("stop", 0)
        raise
    finally:
        pid_distance.adaptation_active = False
        lidar_servo.ChangeDutyCycle(safe_servo_duty(90))  # Reset lidar position

def slam_avoid_obstacle():
    try:
        while slam.detect_obstacle():
            logging.info("Recalculating path with safety checks...")
            
            # Enable adaptation
            pid_distance.adaptation_active = True
            
            # Get current and goal positions with safety margin
            current_pos = slam.get_grid_coords(slam.position)
            goal_pos = slam.get_grid_coords(slam.starting_pose)
            
            # Add 1-cell safety margin to obstacles
            safe_map = add_safety_margin(slam.map, margin=1)
            
            path = a_star(safe_map, current_pos, goal_pos)
            
            if not path:
                logging.error("No safe path found! Performing emergency stop.")
                constrained_motor_control("stop", 0)
                break
                
            for step in path:
                if slam.detect_obstacle():  # Recheck at each step
                    constrained_motor_control("stop", 0)
                    break
                    
                # PID-controlled movement to next grid cell
                target = slam.get_world_coords(step)
                dx = target[0] - slam.position[0]
                dy = target[1] - slam.position[1]
                
                # Use PID for precise movement
                while np.hypot(dx, dy) > 0.1:  # 10cm tolerance
                    error = np.hypot(dx, dy)
                    output = pid_distance.update(error, 0.1)
                    speed = np.clip(output, 20, 50)  # Constrained speed
                    
                    # Direction handling
                    angle = math.atan2(dy, dx)
                    if abs(angle - current_angle) > 0.2:  # ~11 degrees
                        rotate_cobot(math.degrees(angle - current_angle))
                    
                    constrained_motor_control("forward", speed)
                    
                    # Update position
                    slam.update_position()
                    dx = target[0] - slam.position[0]
                    dy = target[1] - slam.position[1]
                    
                    time.sleep(0.1)
                    
    finally:
        pid_distance.adaptation_active = False

def verify_signature(command, signature_hex):
    mac = hmac.new(SECRET_KEY, msg=command.encode(), digestmod=hashlib.sha256)
    expected_signature = mac.hexdigest()
    return hmac.compare_digest(expected_signature, signature_hex)

# --------------------------------------------
# WebRTC Endpoints and Control
# --------------------------------------------
@app.route("/")
async def index():
    try:
        return await render_template("index.html")
    except Exception as e:
        logging.error(f"Error rendering index page: {e}")
        return "Error loading page", 500

@app.route("/updateQRMapping", methods=["POST"])
async def update_qr_mapping():
    global qr_mapping
    try:
        data = await request.get_json()
        for mapping in data.get("mappings", []):
            p_val = mapping.get("p")
            a_val = mapping.get("a")
            if p_val and a_val:
                qr_mapping[p_val] = a_val
        return jsonify({"status": "success", "qr_mapping": qr_mapping})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/auth", methods=["POST"])
async def authenticate():
    data = await request.get_json()
    if not data or data.get("password") != AUTH_PASSWORD:
        return "FAIL", 401
    authorized_clients.add(request.remote_addr)
    return "OK", 200

@app.route("/offer", methods=["POST"])
async def offer():
    if request.remote_addr not in authorized_clients:
    abort(401)

    try:
        
        params = await request.get_json()
        offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        pc = RTCPeerConnection()
        pcs.add(pc)

        @pc.on("iceconnectionstatechange")
        async def on_ice_state_change():
            logging.info(f"ICE state: {pc.iceConnectionState}")
            if pc.iceConnectionState in ["failed", "closed"]:
                await pc.close()
                pcs.discard(pc)

        @pc.on("datachannel")
        def on_datachannel(channel):
            global data_channel
            data_channel = channel
            @channel.on("message")
            def on_message(message):
                handle_control(message, channel)

        pc.addTrack(video_track)

        await pc.setRemoteDescription(offer_desc)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
    except Exception as e:
        logging.error(f"WebRTC failed: {e}")
        return jsonify({"error": str(e)}), 500

    # Dynamic WebRTC Optimization
    video_sender = pc.getSenders()[0]
    parameters = video_sender.getParameters()
    
    if parameters.encodings:
        # Base configuration for 25 FPS
        base_config = {
            "maxBitrate": 2_000_000,  # 2 Mbps for high-quality 25 FPS
            "maxFramerate": 25,
            "scaleResolutionDownBy": 1.0  # Full resolution
        }

        # Adjust for multiple clients
        if len(pcs) > 1:
            base_config.update({
                "maxBitrate": 800_000,  # 800kbps per client
                "scaleResolutionDownBy": 1.5  # Slightly reduced resolution
            })

        # Apply settings to all encodings
        for encoding in parameters.encodings:
            encoding.update(base_config)
        
        try:
            video_sender.setParameters(parameters)
            logging.info(f"WebRTC configured: {base_config}")
        except Exception as e:
            logging.error(f"WebRTC parameter error: {e}")

    return jsonify({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

@app.websocket("/ws")
async def websocket_handler():
    ws = await request.accept()
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("datachannel")
    def on_datachannel(channel):
        logging.info(f"✅ DataChannel created: {channel.label}")

        @channel.on("message")
        def on_message(message):
            try:
                payload = json.loads(message)
                command = payload.get("command")
                signature = payload.get("signature")

                if not verify_signature(command, signature):
                    logging.warning("⚠️ Invalid command signature from WebRTC")
                    return

                logging.info(f"Executing verified command: {command}")
                execute_command(command)  # This function must be defined by you
            except Exception as e:
                logging.error(f"❌ Command handling error: {e}")

@app.route("/sensor")
async def sensor_data():
    try:
        temperature = bmp280.read_temperature()
        pressure = bmp280.read_pressure()
        return jsonify({"temperature": temperature, "pressure": pressure})
    except Exception as e:
        logging.error(f"BMP280 sensor error: {e}")
        return jsonify({"temperature": None, "pressure": None})

@app.route("/speed")
async def get_speed():
    return jsonify(speed_estimator.get_speed())

# --------------------------------------------
# MPU6050 Speed Estimation with Kalman Filtering
# --------------------------------------------
ACCEL_SCALE = 16384.0
GRAVITY = 9.81
COMPLEMENTARY_ALPHA = 0.95

class MPU6050_SpeedEstimator:
    def __init__(self):
        self.speed_x = 0.0
        self.speed_y = 0.0
        self.speed_z = 0.0
        self.prev_time = time.time()
        self.offsets = self.calibrate_sensor()
    
    def calibrate_sensor(self, samples=100):
        print("[INFO] Calibrating MPU6050... Keep the sensor still.")
        offset_x, offset_y, offset_z = 0, 0, 0
        for _ in range(samples):
            ax, ay, az = self.read_acceleration(raw=True)
            offset_x += ax
            offset_y += ay
            offset_z += az
            time.sleep(0.005)
        return (offset_x / samples, offset_y / samples, offset_z / samples)

    def read_word_2c(self, addr):
        high = bus.read_byte_data(MPU6050_ADDR, addr)
        low = bus.read_byte_data(MPU6050_ADDR, addr+1)
        val = (high << 8) + low
        return -((65535 - val) + 1) if val >= 0x8000 else val

    def read_acceleration(self, raw=False):
        ax_raw = self.read_word_2c(0x3B)
        ay_raw = self.read_word_2c(0x3D)
        az_raw = self.read_word_2c(0x3F)
        if raw:
            return ax_raw, ay_raw, az_raw
        ax = ((ax_raw - self.offsets[0]) / ACCEL_SCALE) * GRAVITY
        ay = ((ay_raw - self.offsets[1]) / ACCEL_SCALE) * GRAVITY
        az = ((az_raw - self.offsets[2]) / ACCEL_SCALE) * GRAVITY
        return ax, ay, az
    
    def update_speed(self):
        # Initialize previous acceleration values with an initial reading
        prev_ax, prev_ay, prev_az = self.read_acceleration()
        while True:
            current_time = time.time()
            dt = current_time - self.prev_time
            self.prev_time = current_time
            ax, ay, az = self.read_acceleration()
            ax = COMPLEMENTARY_ALPHA * ax + (1 - COMPLEMENTARY_ALPHA) * prev_ax
            ay = COMPLEMENTARY_ALPHA * ay + (1 - COMPLEMENTARY_ALPHA) * prev_ay
            az = COMPLEMENTARY_ALPHA * az + (1 - COMPLEMENTARY_ALPHA) * prev_az
            az -= GRAVITY
            raw_speed_x = self.speed_x + ax * dt
            raw_speed_y = self.speed_y + ay * dt
            raw_speed_z = self.speed_z + az * dt
            self.speed_x = kalman_speed_x.update(raw_speed_x)
            self.speed_y = kalman_speed_y.update(raw_speed_y)
            self.speed_z = kalman_speed_z.update(raw_speed_z)
            prev_ax, prev_ay, prev_az = ax, ay, az
            time.sleep(0.01)

    def get_speed(self):
        with state_lock:
            speed_magnitude = math.sqrt(self.speed_x**2 + self.speed_y**2 + self.speed_z**2)
            return {
                "speed_x": self.speed_x,
                "speed_y": self.speed_y,
                "speed_z": self.speed_z,
                "speed_magnitude": speed_magnitude
            }

speed_estimator = MPU6050_SpeedEstimator()
Thread(target=speed_estimator.update_speed, daemon=True).start()

# --------------------------------------------
# Command Handling: Data Channel and Watch Control
# --------------------------------------------
def handle_control(message, channel=None):
    logging.info(f"Received message: {message}")

    try:
        # ✅ New: Kotlin-style motorSpeed command
        if message.startswith("motorSpeed:"):
            parts = message.split(":")
            if len(parts) == 3:
                _, direction, speed_str = parts
                try:
                    speed_val = int(speed_str)
                    manual_control(direction, speed_val)
                    if channel:
                        channel.send(f"✅ Speed set to {speed_val} in direction {direction}")
                except ValueError:
                    logging.warning(f"Invalid speed value: {speed_str}")
                    if channel:
                        channel.send("❌ Invalid speed value.")
            else:
                if channel:
                    channel.send("❌ Invalid motorSpeed format. Use motorSpeed:forward:60")
            return

        # ✅ Regular commands like "stop", "left", etc.
        elif message in ["forward", "backward", "left", "right", "stop"]:
            manual_control(message)
            if channel:
                channel.send(f"✅ Executed: {message}")
            return

        # 👀 You can add more command types here if needed (e.g., servo, scan_qr)

        else:
            logging.warning(f"Unknown command: {message}")
            if channel:
                channel.send("❌ Unknown command.")

    except Exception as e:
        logging.error(f"Error in handle_control: {e}")
        if channel:
            channel.send(f"❌ Error: {str(e)}")


def update_object_position(measured_x, measured_y):
    z = np.array([[measured_x], [measured_y]])
    kalman_object.predict()
    filtered_state = kalman_object.update(z)
    return filtered_state[0, 0], filtered_state[1, 0]

def follow_object(position):
    filtered_x, filtered_y = update_object_position(position[0], position[1])
    center_x, center_y = 320, 240
    error_x = filtered_x - center_x
    error_y = filtered_y - center_y
    dt = 0.1
    correction = pid_lateral.update(error_x, dt)
    steer_speed = max(min(abs(correction), 100), 20)
    if error_x > 20:
        manual_control("right", speed_value=steer_speed)
    elif error_x < -20:
        manual_control("left", speed_value=steer_speed)
    else:
        if error_y > 20:
            manual_control("forward", speed_value=speed)
        elif error_y < -20:
            manual_control("backward", speed_value=speed)
        else:
            manual_control("stop")

def search_for_target():
    slam.update_position()
    slam.update_map()
    for _ in range(12):
        try:
            with camera_lock:
                frame = video_track.get_latest_frame() 
            servo_tilt.ChangeDutyCycle(2.5 + (60 / 180.0) * 10)
            time.sleep(0.2)
            if detect_qr_code() or detect_objects(frame, target_object):
                return True
            with camera_lock:
                frame = video_track.get_latest_frame() 
            servo_tilt.ChangeDutyCycle(2.5 + (120 / 180.0) * 10)
            time.sleep(0.2)
            if detect_qr_code() or detect_objects(frame, target_object):
                return True
            servo_tilt.ChangeDutyCycle(2.5 + (90 / 180.0) * 10)
            rotate_cobot(30)
        except Exception as e:
            logging.error(f"Error in search_for_target: {e}")
    return False

def follow_mode(mode):
    global current_mode, target_object, yolo_enabled

    if mode == "follow_object":
        yolo_enabled = True  # ✅ Enable YOLO

    while current_mode == mode:
        if mode == "follow_me":
            yolo_enabled = False  # ⛔ Ensure YOLO is off
            qr_data = detect_qr_code()
            slam.update_position()
            slam.update_map()
            if qr_data and qr_data.startswith("p"):
                goal = slam.get_qr_position(qr_data)
                start_grid = (
                    int(slam.position[0] / slam.cell_size) + slam.origin[0],
                    slam.origin[1] - int(slam.position[1] / slam.cell_size)
                )
                goal_grid = (
                    int(goal[0] / slam.cell_size) + slam.origin[0],
                    slam.origin[1] - int(goal[1] / slam.cell_size)
                )
                path = a_star(slam.map, start_grid, goal_grid)
                for step in path:
                    motor.move_to(step)
                    if slam.detect_obstacle():
                        slam_avoid_obstacle()
                        break
            else:
                logging.info("QR code not recognized for follow_me mode.")
                if search_for_target():
                    continue
                else:
                    ramp_motor_speed("stop", 0)

        elif mode == "follow_object":
            with camera_lock:
                frame = video_track.get_latest_frame()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected = detect_objects(frame, target_object)
            slam.update_position()
            slam.update_map()
            if detected:
                _, bbox = detected[0]
                x1, y1, x2, y2 = bbox
                center_obj = ((x1 + x2) / 2, (y1 + y2) / 2)
                follow_object(center_obj)
            else:
                if search_for_target():
                    continue
                else:
                    ramp_motor_speed("stop", 0)

        time.sleep(0.1)

    yolo_enabled = False  # 🔚 Disable YOLO when exiting follow mode

def adjust_tilt_and_follow(position):
    center_x, center_y = 320, 240
    object_x, object_y = int(position[0]), int(position[1])
    error_x = object_x - center_x
    error_y = object_y - center_y
    dt = 0.1
    correction_x = pid_lateral.update(error_x, dt)
    correction_y = pid_distance.update(error_y, dt)
    new_tilt = 90 - correction_y  
    new_tilt = max(0, min(180, new_tilt))
    servo_tilt.ChangeDutyCycle(safe_servo_duty(new_tilt))
    if abs(error_x) > 20:
        steer_speed = max(min(abs(correction_x), 100), 20)
        if error_x > 0:
            manual_control("right", speed_value=steer_speed)
        else:
            manual_control("left", speed_value=steer_speed)
    else:
        if error_y > 20:
            manual_control("forward", speed_value=speed)
        elif error_y < -20:
            manual_control("backward", speed_value=speed)
        else:
            manual_control("stop")

def handle_watch_client(client_socket):
    global speed
    with client_socket:
        while True:
            data = client_socket.recv(1024)
            if not data:
                break
            command = data.decode().strip()
            logging.info(f"Received gesture command: {command}")
            if command == "knock":
                ramp_motor_speed("forward", 60)
                speed = 60
            elif command == "wrist_up":
                speed = min(speed + 10, 100)
                ramp_motor_speed("forward", speed)
            elif command == "wrist_down":
                speed = max(speed - 10, 0)
                ramp_motor_speed("forward", speed)
            elif command.startswith("tilt_left"):
                try:
                    parts = command.split(":")
                    tilt_val = float(parts[1]) if len(parts) > 1 else 10
                    turning_speed = min(speed, tilt_val * 10)
                    manual_control("left", speed_value=turning_speed)
                except Exception as e:
                    logging.error(f"Error processing tilt_left: {e}")
            elif command.startswith("tilt_right"):
                try:
                    parts = command.split(":")
                    tilt_val = float(parts[1]) if len(parts) > 1 else 10
                    turning_speed = min(speed, tilt_val * 10)
                    manual_control("right", speed_value=turning_speed)
                except Exception as e:
                    logging.error(f"Error processing tilt_right: {e}")
            elif command == "pinch":
                manual_control("stop")
            try:
                lidar_servo.ChangeDutyCycle(safe_servo_duty(90))
                time.sleep(0.1)
                distance_mm = sensor_fusion.get_distance()
                if distance_mm > 0 and (distance_mm / 10.0) < 5:
                    manual_control("stop")
            except Exception as e:
                logging.error(f"Error checking lidar sensor: {e}")

def galaxy_watch_control_server():
    host = ''
    port = 6000
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    logging.info(f"Galaxy Watch Server listening on port {port}")
    while True:
        client_socket, addr = server_socket.accept()
        logging.info(f"Connected with Galaxy Watch: {addr}")
        Thread(target=handle_watch_client, args=(client_socket,), daemon=True).start()

def get_bird_eye_view(frame):
    """
    Applies perspective correction to transform the captured frame into a top-down view.
    These source points should be calibrated for your specific setup.
    """
    height, width = frame.shape[:2]
    # Example source points (calibrate these for your camera)
    src = np.float32([[width * 0.1, height * 0.3],
                      [width * 0.9, height * 0.3],
                      [width * 0.9, height * 0.9],
                      [width * 0.1, height * 0.9]])
    dst = np.float32([[0, 0],
                      [width, 0],
                      [width, height],
                      [0, height]])
    M = cv2.getPerspectiveTransform(src, dst)
    corrected = cv2.warpPerspective(frame, M, (width, height))
    return corrected

def dynamic_row_threshold(decoded_objects, frame_height):
    """
    Calculates a dynamic threshold based on the median y-coordinate of detected QR codes.
    This helps to adapt to variations in QR placement rather than using a fixed half-height.
    """
    if len(decoded_objects) > 1:
        centers = []
        for qr in decoded_objects:
            y = qr.rect.top
            h = qr.rect.height
            centers.append(y + h / 2)
        threshold = np.median(centers)
        return threshold
    else:
        return frame_height / 2

def detect_qr_in_row(desired_row):
    """
    Captures a frame, applies perspective correction, decodes QR codes,
    and then dynamically determines the row threshold.
    Returns the data from the first QR code that matches the desired row ("upper" or "lower").
    """
    while True:
        with camera_lock:
            frame = video_track.get_latest_frame() 
        # Apply perspective correction for a top-down view
        corrected_frame = get_bird_eye_view(frame)
        decoded_objects = decode(corrected_frame)
        if decoded_objects:
            threshold = dynamic_row_threshold(decoded_objects, corrected_frame.shape[0])
            for qr in decoded_objects:
                y = qr.rect.top
                h = qr.rect.height
                qr_center_y = y + h / 2
                if desired_row == "upper" and qr_center_y < threshold:
                    return qr.data.decode("utf-8")
                elif desired_row == "lower" and qr_center_y >= threshold:
                    return qr.data.decode("utf-8")
        time.sleep(0.1)

def center_qr():
    """
    Centers the QR code within the camera frame.
    This routine remains unchanged, as it primarily manages lateral adjustments.
    """
    while True:
        with camera_lock:
            frame = video_track.get_latest_frame() 
        decoded_objects = decode(frame)
        if decoded_objects:
            qr = decoded_objects[0]
            x = qr.rect.left
            w = qr.rect.width
            qr_center_x = x + w / 2
            frame_center_x = frame.shape[1] / 2
            lateral_error = qr_center_x - frame_center_x
            if abs(lateral_error) < 10:
                reset_current_angle()  # Reset angle after centering QR code
                break
            elif lateral_error > 0:
                manual_control("right", speed_value=20)
            else:
                manual_control("left", speed_value=20)
        time.sleep(0.01)
    manual_control("stop")

def manage_distance(target_distance, timeout=30):
    start_time = time.time()
    pid_distance_local = PID(1.0, 0.1, 0.05, max_integral=50)  # Anti-windup
    pid_lateral_local = PID(1.0, 0.0, 0.1, max_integral=20)
    tolerance = 0.003  # ±3mm tolerance
    lateral_threshold = 10  # pixels

    # Reset PID states
    pid_distance_local.integral = 0
    pid_distance_local.last_error = 0
    pid_lateral_local.integral = 0
    pid_lateral_local.last_error = 0

    prev_time = time.time()

    while True:
        # Timeout check
        if time.time() - start_time > timeout:
            logging.error("Distance adjustment timeout!")
            manual_control("stop")
            break

        # Timing calculations
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt <= 0:
            dt = 0.01  # Prevent division by zero

        # Get measurements
        current_distance = sensor_fusion.get_distance()
        
        if current_distance == -1 or current_distance == float('inf'):
            logging.error("All distance sensors failed! Stopping.")
            manual_control("stop")
            return False

        lateral_error = 0

        # Get lateral error from QR code
        with camera_lock:
            frame = camera.capture_array() if camera else None
            if frame is not None:
                decoded = decode(frame)
                if decoded:
                    qr = decoded[0]
                    qr_center = qr.rect.left + qr.rect.width / 2
                    lateral_error = qr_center - (frame.shape[1] / 2)

        # Target achievement check (0.1cm tolerance)
        distance_ok = abs(target_distance - current_distance) <= tolerance
        lateral_ok = abs(lateral_error) < lateral_threshold

        if distance_ok and lateral_ok:
            logging.info(f"Target {target_distance}cm achieved ±{tolerance}cm")
            break

        # Lateral correction
        if not lateral_ok:
            output = pid_lateral_local.update(lateral_error, dt)
            speed = max(min(abs(output), 40), 20)  # 20-40% speed range
            if lateral_error > 0:
                manual_control("right", speed_value=speed)
            else:
                manual_control("left", speed_value=speed)

        # Distance correction
        else:
            error = target_distance - current_distance
            output = pid_distance_local.update(error, dt)
            speed = max(min(abs(output), 60), 20)  # 20-60% speed range
            if error > 0:
                manual_control("forward", speed_value=speed)
            else:
                manual_control("backward", speed_value=speed)

        time.sleep(0.01)

    manual_control("stop")
    return True

def control_factory_servo(servo_id, angle):
    """Controls factory servos using persistent PWM objects"""
    global servo_factory1, servo_factory2

    angle = max(0, min(180, angle))
    duty = 2.5 + (angle / 180.0) * 10

    if servo_id == 1:
        servo_factory1.ChangeDutyCycle(duty)
    elif servo_id == 2:
        servo_factory2.ChangeDutyCycle(duty)
    else:
        logging.error(f"Invalid servo ID: {servo_id}")
        return

    time.sleep(abs(angle) * 0.01)  # ~10ms per degree
    logging.info(f"Factory Servo {servo_id} moved to {angle}°")

def factory_mode():
    try:
        def map_qr_to_a(qr_data):
            global qr_mapping
            if qr_mapping and qr_data in qr_mapping:
                return qr_mapping[qr_data]
            if qr_data in ["p1", "p4"]: return "a1"
            if qr_data in ["p2", "p5"]: return "a2"
            if qr_data in ["p3", "p6"]: return "a3"
            return None

        def handle_upper_row():
            count = 0
            while count < 3:
                qr_data = detect_qr_in_row("upper")
                if not qr_data:
                    logging.warning("QR not found in upper row; retrying…")
                    search_for_target()
                    continue

                center_qr()
                if not manage_distance(0.05):
                    logging.error("Distance @5cm failed. Aborting factory_mode.")
                    return False

                control_factory_servo(1, 140)
                if not manage_distance(0.055):
                    logging.error("Distance @5.5cm failed. Aborting factory_mode.")
                    return False

                control_factory_servo(2, 90)
                control_factory_servo(1, 160)
                rotate_cobot(180)

                _ = map_qr_to_a(qr_data)
                search_for_target()
                center_qr()
                if not manage_distance(0.05):
                    logging.error("Return to 5cm failed. Aborting factory_mode.")
                    return False

                control_factory_servo(1, 170)
                if not manage_distance(0.045):
                    logging.error("Distance @4.5cm failed. Aborting factory_mode.")
                    return False

                control_factory_servo(2, 0)
                control_factory_servo(1, 160)
                rotate_cobot(180)

                count += 1

            return True

        def handle_lower_row():
            count = 0
            while count < 3:
                qr_data = detect_qr_in_row("lower")
                if not qr_data:
                    logging.warning("QR not found in lower row; retrying…")
                    search_for_target()
                    continue

                center_qr()
                if not manage_distance(0.05):
                    logging.error("Distance @5cm failed. Aborting factory_mode.")
                    return False

                control_factory_servo(1, 170)
                if not manage_distance(0.055):
                    logging.error("Distance @5.5cm failed. Aborting factory_mode.")
                    return False

                control_factory_servo(2, 90)
                control_factory_servo(1, 160)
                rotate_cobot(180)

                _ = map_qr_to_a(qr_data)
                search_for_target()
                center_qr()
                if not manage_distance(0.05):
                    logging.error("Return to 5cm failed. Aborting factory_mode.")
                    return False

                control_factory_servo(1, 170)
                if not manage_distance(0.045):
                    logging.error("Distance @4.5cm failed. Aborting factory_mode.")
                    return False

                control_factory_servo(2, 0)
                control_factory_servo(1, 160)
                rotate_cobot(180)

                count += 1

            return True

        # Run upper row first, then lower
        if not handle_upper_row():
            return
        handle_lower_row()

    except Exception as e:
        logging.error(f"[factory_mode] Unexpected error: {e}")

    finally:
        # Servos are stopped inside set_servo_angle() after each move
        pass  # No need to stop servos again here

def shutdown_handler(signum, frame):
    logging.info("Shutting down cobot...")

    try:
        manual_control("stop")

        # Stop all PWM instances safely
        try:
            pwmA.stop()
            pwmB.stop()
            servo_pan.stop()
            servo_tilt.stop()
            lidar_servo.stop()
        except Exception as e:
            logging.error(f"PWM stop error: {e}")

        # Stop camera if running
        if 'camera' in globals() and camera:
            camera.stop_preview()
        servo_factory1.stop()
        servo_factory2.stop()
        GPIO.cleanup()
        logging.info("GPIO cleanup complete.")

    except Exception as e:
        logging.error(f"Shutdown error: {e}")

    sys.exit(0)

# Attach shutdown signal hooks
signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

# --------------------------------------------
# Main Loop and Server Initialization
# --------------------------------------------
def main_loop():
    global cobot_active
    face_check_active = True

    while True:
        try:
            if face_check_active and face_recognization():
                cobot_active = True
                face_check_active = False
                logging.info("Face recognized, cobot activated")

            if cobot_active:
                # Enable adaptive PID during operation
                pid_distance.adaptation_active = True

                # Add IMU tilt fusion here 👇
                fused_roll, fused_pitch = get_fused_tilt(imu)
                logging.info(f"Fused Tilt - Roll: {fused_roll:.2f}°, Pitch: {fused_pitch:.2f}°")

                # Optional: use pitch/roll in navigation logic
                # Example: Stop if pitch is too steep
                if abs(fused_pitch) > 20:
                    logging.warning("Pitch angle exceeded safe limit! Stopping.")
                    manual_control("stop")

                # Fused obstacle detection using get_center_distance()
                if sensor_fusion and get_center_distance() < 0.3:
                    obstacle_avoidance_control()

            # SLAM operations continue even when cobot is idle
            slam.update_position()
            slam.update_map()

            time.sleep(0.1)

        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            pid_distance.adaptation_active = False
            pid_rotation.adaptation_active = False
            pid_lateral.adaptation_active = False

if __name__ == "__main__":
    # 1. Initialize hardware interfaces first
    try:
        camera = Picamera2()
        # ... other hardware init
    except Exception as e:
        logging.critical(f"Hardware init failed: {e}")
        sys.exit(1)

    # 2. Initialize EKF and synchronization primitives
    ekf_lock = Lock()
    state_lock = Lock()
    ekf = CobotEKF()

    # 3. Start sensor threads
    Thread(target=update_current_angle, daemon=True).start()
    
    # 4. Security check
    if not wait_for_face_unlock(30):
        stop_cobot()
        sys.exit(1)

    # 5. Start main functionality
    Thread(target=main_loop, daemon=True).start()
    Thread(target=yolo_detection_loop, args=(video_track,), daemon=True).start()

    # 6. Start web interface last
    app.run(host="0.0.0.0", port=5000)
