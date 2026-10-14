from robot_hat import PWM, Pin
from robot_hat.utils import reset_mcu
import time
import sys

# ===== 設定 =====
SPEED = 50        # 速度(0〜100)
MOVE_TIME = 2.0   # 各動作の時間[秒]（回転数はこれで調整）
PAUSE = 0.5       # 動作の間の停止時間[秒]


# 片側のモータが逆に回る場合は、その側を -1 にする
M1_SIGN = 1
M2_SIGN = 1

# ===== 初期化 =====
reset_mcu()
time.sleep(0.2)

m1_pwm, m1_dir = PWM("P13"), Pin("D4")   # モーター1: PWM13 / GPIO23
m2_pwm, m2_dir = PWM("P12"), Pin("D5")   # モーター2: PWM12 / GPIO24


def set_motor(pwm, dir_pin, d, speed):
    """d: 1=正転, -1=逆転, 0=停止"""
    if d == 0:
        pwm.pulse_width_percent(0)
        return
    if d == 1:
        dir_pin.high()
    else:
        dir_pin.low()
    pwm.pulse_width_percent(speed)


def drive(d1, d2, speed=None):
    # SPEED をデフォルト値に書くと定義時の値で固定され、
    # change_speed() の変更が反映されないため実行時に参照する
    if speed is None:
        speed = SPEED
    set_motor(m1_pwm, m1_dir, d1 * M1_SIGN, speed)
    set_motor(m2_pwm, m2_dir, d2 * M2_SIGN, speed)


def stop():
    drive(0, 0)



def run(action, d1, d2):
    print(action)
    try:
        drive(d1, d2)
        time.sleep(MOVE_TIME)
    finally:
        # Ctrl+C などで中断されてもモーターを通電したままにしない
        stop()
    time.sleep(PAUSE)

def change_speed():
    global SPEED
    new_speed = input("新しい速度を入力してください (0〜100): ")
    try:
        new_speed = int(new_speed)
        if 0 <= new_speed <= 100:
            SPEED = new_speed
            print(f"速度が {SPEED} に変更されました。")
        else:
            print("無効な速度です。0〜100の範囲で入力してください。")
    except ValueError:
        print("無効な入力です。整数を入力してください。")


# ===== メイン =====
if __name__ == "__main__":
    try:
        tmp = -1
        while tmp != 0:
            print("次の動作を選択してください\n")
            print("1: 前進")
            print("2: 後進")
            print("3: 右回転")
            print("4: 左回転")
            print("5: 速度変更")
            print("\n0: 終了")

            key = input("番号を入力してください: ")
            if key == "1":
                run("前進", 1, 1)
            elif key == "2":
                run("後進", -1, -1)
            elif key == "3":
                run("右回転", 1, -1)
            elif key == "4":
                run("左回転", -1, 1)
            elif key == "5":
                print("速度を変更します。")
                print(f"現在の速度: {SPEED}")
                print("新しい速度を入力してください。(0〜100)")
                change_speed()
            elif key == "0":
                print("終了します。")
                tmp = 0
                sys.exit()
            else:
                print("無効な入力です。")
    except KeyboardInterrupt:
        print("\n中断されました。")
    finally:
        # どの経路で抜けても必ずモーターを止める
        stop()
