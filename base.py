from robot_hat import PWM, Pin
from robot_hat.utils import reset_mcu
import keyboard as k
import time

reset_mcu()
time.sleep(0.2)

m1_pwm, m1_dir = PWM("P13"), Pin("D4")   # D4 = GPIO23
m2_pwm, m2_dir = PWM("P12"), Pin("D5")   # D5 = GPIO24

def set_motor(pwm, dir_pin, d, v):
    """d: 1=正転, -1=逆転, 0=停止"""
    if d == 0:
        pwm.pulse_width_percent(0)
        return
    dir_pin.high() if d == 1 else dir_pin.low()
    pwm.pulse_width_percent(v)

def drive(d1, d2, v):
    set_motor(m1_pwm, m1_dir, d1, v)
    set_motor(m2_pwm, m2_dir, d2, v)

v = 50
try:
    while True:
        time.sleep(0.1)
        if k.is_pressed("q"):
            v = min(v + 10, 100)
        if k.is_pressed("e"):
            v = max(v - 10, 0)

        if k.is_pressed("w"):
            drive(1, 1, v)
        elif k.is_pressed("s"):
            drive(-1, -1, v)
        elif k.is_pressed("a"):
            drive(-1, 1, v)
        elif k.is_pressed("d"):
            drive(1, -1, v)
        else:
            drive(0, 0, v)

        if k.is_pressed("f"):
            break
finally:
    drive(0, 0, 0)