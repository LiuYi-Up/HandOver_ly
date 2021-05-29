from X_arm import X_arm
import keyboard
import time


X_arm = X_arm()
X_arm.Open_port('/dev/ttyUSB2')
while True:
    if keyboard.is_pressed('k'):
        X_arm.setJointAngle(7, 30, 0.5)
        # time.sleep(5)
    
    if keyboard.is_pressed('t'):
        X_arm.setJointAngle(3, 15, 0.5)
        # time.sleep(5)

    if keyboard.is_pressed('g'):
        X_arm.setJointAngle(7, 76, 0.5)
        # time.sleep(5)
    
    if keyboard.is_pressed('r'):
        X_arm.setJointAngle(3, 0, 0.5)
        # time.sleep(5)