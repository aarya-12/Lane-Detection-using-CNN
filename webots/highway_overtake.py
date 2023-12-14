# Copyright 1996-2023 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vehicle_driver controller."""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from vehicle import Driver

lane_detection_model = load_model('D:/Projects/ADAS/full_CNN_modell.h5')
sensorsNames = [
    "front",
    "front right 0",
    "front right 1",
    "front right 2",
    "front left 0",
    "front left 1",
    "front left 2",
    "rear",
    "rear left",
    "rear right",
    "right",
    "left"]
sensors = {}

lanePositions = [10.6, 6.875, 3.2]
currentLane = 1
overtakingSide = None
maxSpeed = 80
safeOvertake = False


def apply_PID(position, targetPosition):
    p_coefficient = 0.05
    i_coefficient = 0.000015
    d_coefficient = 25
    diff = position - targetPosition
    if apply_PID.previousDiff is None:
        apply_PID.previousDiff = diff
    # anti-windup mechanism
    if diff > 0 and apply_PID.previousDiff < 0:
        apply_PID.integral = 0
    if diff < 0 and apply_PID.previousDiff > 0:
        apply_PID.integral = 0
    apply_PID.integral += diff
    # compute angle
    angle = p_coefficient * diff + i_coefficient * apply_PID.integral + d_coefficient * (diff - apply_PID.previousDiff)
    apply_PID.previousDiff = diff
    return angle


apply_PID.integral = 0
apply_PID.previousDiff = None


def get_filtered_speed(speed):
    get_filtered_speed.previousSpeeds.append(speed)
    if len(get_filtered_speed.previousSpeeds) > 100:  
        get_filtered_speed.previousSpeeds.pop(0)
    return sum(get_filtered_speed.previousSpeeds) / float(len(get_filtered_speed.previousSpeeds))


def is_vehicle_on_side(side):
    for i in range(3):
        name = "front " + side + " " + str(i)
        if sensors[name].getValue() > 0.8 * sensors[name].getMaxValue():
            return True
    return False


def reduce_speed_if_vehicle_on_side(speed, side):
    minRatio = 1
    for i in range(3):
        name = "front " + overtakingSide + " " + str(i)
        ratio = sensors[name].getValue() / sensors[name].getMaxValue()
        if ratio < minRatio:
            minRatio = ratio
    return minRatio * speed


get_filtered_speed.previousSpeeds = []
driver = Driver()
for name in sensorsNames:
    sensors[name] = driver.getDevice("distance sensor " + name)
    sensors[name].enable(10)

gps = driver.getDevice("gps")
gps.enable(10)

camera = driver.getDevice("camera")
camera.enable(10)
camera.recognitionEnable(50)

def apply_lane_detection(image):
    small_img = cv2.resize(image[:, :, :3], (160, 80))
    small_img = np.array(small_img)
    small_img = small_img[None, :, :, :]

    prediction = lane_detection_model.predict(small_img)[0] * 255

    prediction = prediction.astype(np.uint8)

    return prediction

while driver.step() != -1:
    frontDistance = sensors["front"].getValue()
    frontRange = sensors["front"].getMaxValue()
    speed = maxSpeed * frontDistance / frontRange
    if sensors["front right 0"].getValue() < 8.0 or sensors["front left 0"].getValue() < 8.0:
        speed = min(0.5 * maxSpeed, speed)
    if overtakingSide is not None:
        if overtakingSide == 'right' and sensors["left"].getValue() < 0.8 * sensors["left"].getMaxValue():
            overtakingSide = None
            currentLane -= 1
        elif overtakingSide == 'left' and sensors["right"].getValue() < 0.8 * sensors["right"].getMaxValue():
            overtakingSide = None
            currentLane += 1
        else:  
            speed2 = reduce_speed_if_vehicle_on_side(speed, overtakingSide)
            if speed2 < speed:
                speed = speed2
    speed = get_filtered_speed(speed)
    driver.setCruisingSpeed(speed)
    speedDiff = driver.getCurrentSpeed() - speed
    if speedDiff > 0:
        driver.setBrakeIntensity(min(speedDiff / speed, 1))
    else:
        driver.setBrakeIntensity(0)
    if frontDistance < 0.8 * frontRange and overtakingSide is None:
        if (is_vehicle_on_side("left") and
                (not safeOvertake or sensors["rear left"].getValue() > 0.8 * sensors["rear left"].getMaxValue()) and
                sensors["left"].getValue() > 0.8 * sensors["left"].getMaxValue() and
                currentLane < 2):
            currentLane += 1
            overtakingSide = 'right'
        elif (is_vehicle_on_side("right") and
                (not safeOvertake or sensors["rear right"].getValue() > 0.8 * sensors["rear right"].getMaxValue()) and
                sensors["right"].getValue() > 0.8 * sensors["right"].getMaxValue() and
                currentLane > 0):
            currentLane -= 1
            overtakingSide = 'left'
    position = gps.getValues()[1]
    angle = max(min(apply_PID(position, lanePositions[currentLane]), 0.5), -0.5)
    driver.setSteeringAngle(-angle)
    if abs(position - lanePositions[currentLane]) < 1.5:  
        overtakingSide = None
    image = camera.getImage()
    image = np.frombuffer(image, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    lane_detection_result = apply_lane_detection(image)

    cv2.imshow("Lane Detection Result", lane_detection_result)
    cv2.waitKey(1)
