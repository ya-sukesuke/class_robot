from robot_hat import PWM, Pin
from robot_hat.utils import reset_mcu
import time
import sys

# ===== 初期化 =====
reset_mcu()
time.sleep(0.2)


line1 = Pin("D11")  # ラインセンサ1: GPIO17
line2 = Pin("D7")  # ラインセンサ4: GPIO4
line3 = Pin("D12")  # ラインセンサ2: GPIO27
line4 = Pin("D13")  # ラインセンサ3: GPIO22

if __name__ == "__main__":
    while True:
        print("入力確認")
        print(line1.value(), line2.value(), line3.value(), line4.value())
        time.sleep(0.1)
        print("Ctrl+C で終了")
