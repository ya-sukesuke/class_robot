from robot_hat import Servo
from robot_hat.utils import reset_mcu
import time

# ===== 初期化 =====
reset_mcu()
time.sleep(0.2)

servo = Servo("P0")   # サーボをつないだピン
servo.angle(-90)      # スタート位置

if __name__ == "__main__":
    print("s + Enter で回転、Ctrl+C で終了")
    try:
        while True:
            if input("> ") == "s":
                servo.angle(-90)
                time.sleep(0.5)
                for a in range(-90, 91, 10):
                    servo.angle(a)
                    print(a)
                    time.sleep(0.1)
    except KeyboardInterrupt:
        print("終了")
