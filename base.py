import RPi.GPIO as G
import keyboard as k
import time
 
G.setmode(G.BCM)
G.setup([23, 24], G.OUT)
v = 75
p = G.PWM(18, 1000)
p.start(v)
 
while True:
    time.sleep(0.1)
    if k.is_pressed("q"):
        v += 10
        if v > 100:
            v = 100
        if v < 0:
            v = 0
        p.ChangeDutyCycle(v)
    if k.is_pressed("e"):
        v -= 10
        if v > 100:
            v = 100
        if v < 0:
            v = 0
        p.ChangeDutyCycle(v)
    
    if k.is_pressed("w"):
        G.output(23, G.HIGH)
        G.output(24, G.LOW)
    if k.is_pressed("s"):
        G.output(23, G.HIGH)
        G.output(24, G.LOW)
    if k.is_pressed("d"):
        G.output(23, G.LOW)
        G.output(24, G.HIGH)
    if k.is_pressed("a"):
        G.output(23, G.LOW)
        G.output(24, G.HIGH)
    if k.is_pressed("f"):
        break
 
p.stop()
G.cleanup()
