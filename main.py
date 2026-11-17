import time
import linetrace        # 注意: linetrace と base はどちらも import 時に reset_mcu() する。
import base             #       モータ設定が消えないよう、base を必ず後に import する
import car_receiver

# ==== 設定 =====
SPEED = 50        # 速度(0〜100)
LOOP_DT = 0.05    # 制御の周期[秒]

def main():
    base.SPEED = SPEED
    car_receiver.start()                     # 座標の受信を裏で開始
    while True:
        car, objs = car_receiver.get_location()

        if car is None:                      # 車が見えない／通信が途切れた → 止める
            base.stop()
            time.sleep(LOOP_DT)
            continue

        targets = [o for o in objs.values() if o["crossed"]]
        if targets:
            t = targets[0]
            print(f"車({car['x']},{car['y']}) 向き{car['angle']}° → 目標 {t['id']}({t['color']}) ({t['x']},{t['y']})")
            # TODO: ここで目標へ向かう制御（base.drive(左, 右) など）
        else:
            base.stop()

        time.sleep(LOOP_DT)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断されました。")
    finally:
        base.stop()          # どの経路で抜けても必ずモーターを止める
