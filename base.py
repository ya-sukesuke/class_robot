import RPi.GPIO as G
import keyboard as k
import time

# 左モータ
L_IN1, L_IN2, L_PWM = 23, 24, 13
# 右モータ
R_IN1, R_IN2, R_PWM = 5, 6, 12

G.setmode(G.BCM)
G.setup([L_IN1, L_IN2, L_PWM, R_IN1, R_IN2, R_PWM], G.OUT)  # PWMピンもsetup

pl = G.PWM(L_PWM, 1000)
pr = G.PWM(R_PWM, 1000)
v = 75
pl.start(v)
pr.start(v)

def motor(in1, in2, d):
    """d: 1=正転, -1=逆転, 0=停止"""
    G.output(in1, G.HIGH if d == 1 else G.LOW)
    G.output(in2, G.HIGH if d == -1 else G.LOW)

def drive(left, right):
    motor(L_IN1, L_IN2, left)
    motor(R_IN1, R_IN2, right)

try:
    while True:
        time.sleep(0.1)

        if k.is_pressed("q"):
            v = min(v + 10, 100)
        if k.is_pressed("e"):
            v = max(v - 10, 0)
        pl.ChangeDutyCycle(v)
        pr.ChangeDutyCycle(v)

        if k.is_pressed("w"):
            drive(1, 1)      # 前進
        elif k.is_pressed("s"):
            drive(-1, -1)    # 後退
        elif k.is_pressed("a"):
            drive(-1, 1)     # 左旋回
        elif k.is_pressed("d"):
            drive(1, -1)     # 右旋回
        else:
            drive(0, 0)      # 何も押してなければ停止

        if k.is_pressed("f"):
            break
finally:
    pl.stop()
    pr.stop()
    G.cleanup()