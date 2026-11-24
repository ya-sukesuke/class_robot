"""
main_haku.py - ロボット綱引き 自動制御「後引き」作戦（ラズパイで実行）

  ノートPC の board_vision.py が天井カメラで「車の位置・向き」と「ブロックの位置」を送る
  → car_receiver.py で受け取る → このプログラムが車を動かす

  作戦：相手が綱を引いてブロックが相手側へ動いたら、取りに行って引き返す
    （最初）  : 白線に沿って走り、横線に着いたらカメラの制御へ
    STANDBY  : 待機位置で、ラインの方を向いて待つ。各ブロックの位置を見張る
               → ブロックが「一番自陣寄りにあった位置」から PULLED_DIST 以上
                 相手側へ動いたら「相手に引かれた」と判断
    APPROACH : そのブロックの自陣側（真後ろ）に回り込み、ラインの方を向いて近づく
    GRAB     : つかむ → 後退して自陣側へ引く → はなして下がる → STANDBY に戻る
    2分（MATCH_TIME）たったら全停止

使い方
  1. ノートPC で board_vision.py を起動し、四隅の指定とブロックの登録をしておく
     （登録すると名前が固定され、ブロックの取り違えが起きにくい）
  2. ラズパイで  python3 main_haku.py
     最初は DRY_RUN = True で、モーターを回さずに表示だけ確認するのがおすすめ
"""
import math
import time

import base             # モーター（import 時に reset_mcu() される。サーボより先に import すること）
import car_receiver
from robot_hat import Pin, Servo
# ※ linetrace.py は使わない（ラインセンサの処理はこのファイルの中にある）

# ===================== 設定 =====================
DRY_RUN = False         # True: モーター・サーボを動かさず、判断の結果だけ表示する

# --- 盤面（board_vision.py と合わせる）---
BOARD_W, BOARD_H = 600, 400   # 補正後の盤面画像サイズ[px]
LINE_Y = 200                  # 中央のライン（board_vision.py の LINE_Y）
OUR_SIDE = "top"              # 自陣が補正後の盤面画像の 上(y < LINE_Y) なら "top"、下なら "bottom"
SAFE_MARGIN = 30              # 盤の端からこの距離[px]より外へは進まない（外周壁に当てない）
LINE_MARGIN = 10              # 車の先端はラインからこの距離[px]より相手側へ出さない（反則防止）

# --- 相手に引かれたかの判断 ---
PULLED_DIST = 25        # 一番自陣寄りにあった位置から、これ以上[px]相手側へ動いたら「引かれた」
BLOCK_LINE_MARGIN = 20  # ラインからこの距離[px]以内のブロックは、相手コートに入る恐れがあるので狙わない
COOLDOWN = 3.0          # 引き返した後、同じブロックをこの秒数は狙わない
SKIP_NAMES = []         # 狙わないブロックの名前（例: ["object3"]）

# --- 待機位置 ---
STANDBY_X = BOARD_W // 2                 # 待機する場所[px]（自陣の真ん中あたり）
STANDBY_DEPTH = 90                       # ラインから自陣側へこの距離[px]の所で待つ
STANDBY_TOL = 30                         # 待機位置からこの距離[px]以内なら「着いた」

# --- 走行 ---
FWD_SPEED = 40          # 前進の基本速度[%]
TURN_SPEED = 35         # その場旋回の速度[%]
KP = 0.8                # 向きのずれ[度] → 左右の速度差[%] の比例ゲイン
TURN_IN_PLACE_DEG = 30  # 向きのずれがこれより大きければ、その場で旋回してから進む
TURN_SIGN = 1           # 目標と逆向きに曲がっていく場合は -1 にする

# --- 回り込み・つかむ ---
STAGE_DIST = 110        # ブロックの自陣側、この距離[px]の点にまず回り込む
SIDE_TOL = 40           # ブロックとの横のずれがこれ以下なら「真後ろにいる」
GRAB_DIST = 60          # 車の中心（QRの中心）とブロックの距離[px]がこれ以下でつかむ
GRAB_ALIGN_DEG = 15     # つかむときに許す向きのずれ[度]
FRONT_OFFSET = 40       # 車の中心から先端（後端）までの距離[px]
SERVO_PIN = "P0"
SERVO_OPEN = -60        # 開いた角度[度]（servo_sweep.py で確かめた値に変える）
SERVO_CLOSE = 30        # 閉じた角度[度]
SERVO_WAIT = 0.6        # サーボが動き終わるまで待つ時間[秒]

# --- 引き返す ---
PULL_SPEED = 45         # 後退して引く速度[%]
PULL_DIST = 80          # つかんだ位置からこの距離[px]自陣側へ引いたら終わり
PULL_TIME = 3.0         # ブロックが見えなくても、この秒数で引くのをやめる
BACK_OFF_TIME = 0.5     # はなした後に下がる時間[秒]

# --- スタート直後のライントレース ---
USE_LINE_START = True   # True: 最初は白線に沿って走り、その後カメラの制御に切り替える
LINE_SPEED = 35         # ライントレースの速度[%]
LINE_STOP_AT_CROSS = True   # 横線（交差点）に着いたらカメラの制御に切り替える
LINE_IGNORE_TIME = 0.5  # スタート直後この秒数は横線の判定をしない
LINE_MAX_TIME = 5.0     # 横線が見つからなくても、この秒数でカメラの制御に切り替える

# --- ラインセンサ ---
LINE_PINS = [17, 27, 22, 4]   # GPIO番号を「左から順」に（D0～D3 = GPIO17, 4, 27, 22）
WHITE_VALUE = 0         # 白線の上でセンサが出す値（白=0・黒=1 と逆なら 1 にする）
LINE_WEIGHTS = [-3, -1, 1, 3]  # 各センサの位置（左が −、右が +）
LINE_KP = 12            # 線のずれ → 左右の速度差[%]
LINE_SEARCH_SPEED = 30  # 線を見失ったときに旋回して探す速度[%]
CROSS_COUNT = 3         # 白のセンサがこの個数以上なら「横線（交差点）」
MIN_SPEED = 25          # これより遅いとTTモーターが回らない[%]

# --- その他 ---
MATCH_TIME = 120        # 競技時間[秒]。これを過ぎたら止まる
LOST_TIME = 1.0         # 狙ったブロックがこの秒数見えなければ、待機に戻る
LOOP_DT = 0.05          # 制御の周期[秒]
# ================================================

OPP = 1 if OUR_SIDE == "top" else -1    # 相手側へ向かうときの y の増える向き
FACE_LINE_ANGLE = 90 * OPP              # ラインの方を向いたときの車の角度（下=+90, 上=-90）


# ---------------- モーター・サーボ ----------------
servo = None if DRY_RUN else Servo(SERVO_PIN)


def drive_lr(left, right):
    """左右のモーターを別々の速さで回す（+:前進, -:後退, 単位[%]）
    base.py の M1 = 左, M2 = 右（base.py の「右回転 = (1, -1)」より）"""
    if DRY_RUN:
        return
    for pwm, dir_pin, sign, v in ((base.m1_pwm, base.m1_dir, base.M1_SIGN, left),
                                  (base.m2_pwm, base.m2_dir, base.M2_SIGN, right)):
        v = max(-100, min(100, v))
        if abs(v) < 1:
            base.set_motor(pwm, dir_pin, 0, 0)
        else:
            d = 1 if v > 0 else -1
            base.set_motor(pwm, dir_pin, d * sign, max(abs(v), MIN_SPEED))


# ---------------- ラインセンサ ----------------
# active_state=True を付けないと「floating, but active_state is not defined」エラーになる
line_sensors = [Pin(p, Pin.IN, active_state=True) for p in LINE_PINS]
_last_side = 0          # 最後に線が見えた側（-1:左, +1:右）


def line_read():
    """[左, 左中, 右中, 右]  白=1, 黒=0"""
    return [1 if s.value() == WHITE_VALUE else 0 for s in line_sensors]


def line_is_cross(w):
    return sum(w) >= CROSS_COUNT


def line_speed(w, base_speed):
    """線のずれに比例して左右の速度を決める（線が右 → 右へ曲がる）"""
    global _last_side
    if sum(w) == 0:                       # 見失った → 最後に見えた側へその場旋回
        s = LINE_SEARCH_SPEED if _last_side >= 0 else -LINE_SEARCH_SPEED
        return s, -s
    pos = sum(k * v for k, v in zip(LINE_WEIGHTS, w)) / sum(w)
    if pos != 0:
        _last_side = 1 if pos > 0 else -1
    return base_speed + LINE_KP * pos, base_speed - LINE_KP * pos


def stop():
    if not DRY_RUN:
        base.stop()


def spin(err):
    """その場旋回（err + なら右へ）"""
    t = TURN_SPEED if err > 0 else -TURN_SPEED
    drive_lr(t, -t)


def gripper(close):
    if DRY_RUN:
        print("  [サーボ]", "閉じる" if close else "開く")
        return
    servo.angle(SERVO_CLOSE if close else SERVO_OPEN)
    time.sleep(SERVO_WAIT)


def line_start():
    """スタート直後：白線に沿って走り、横線に着いたら（または時間切れで）止まる"""
    print("ライントレースでスタート")
    t0 = time.time()
    while time.time() - t0 < LINE_MAX_TIME:
        w = line_read()
        # スタート地点の線を横線と間違えないよう、最初の LINE_IGNORE_TIME 秒は判定しない
        if LINE_STOP_AT_CROSS and time.time() - t0 > LINE_IGNORE_TIME and line_is_cross(w):
            print("横線に到着")
            break
        left, right = line_speed(w, LINE_SPEED)
        drive_lr(left, right)
        time.sleep(0.02)
    else:
        print("ライントレース時間切れ")
    stop()


# ---------------- 位置の計算 ----------------
def wrap180(a):
    """角度を -180〜180 に収める"""
    return (a + 180) % 360 - 180


def aim(car, x, y):
    """点(x, y) までの距離[px] と、向きのずれ[度]（+ = 右側, - = 左側）"""
    dx, dy = x - car["x"], y - car["y"]
    bearing = math.degrees(math.atan2(dy, dx))       # 画面座標（右=0, 下=+90）
    return math.hypot(dx, dy), wrap180(bearing - car["angle"]) * TURN_SIGN


def point_ahead(car, d):
    """車の中心から向いている方向へ d[px] 進んだ点（d が − なら後ろ）"""
    a = math.radians(car["angle"])
    return car["x"] + d * math.cos(a), car["y"] + d * math.sin(a)


def in_board(x, y):
    m = SAFE_MARGIN
    return m <= x <= BOARD_W - m and m <= y <= BOARD_H - m


def our_side(y, margin):
    """y がラインから margin[px] 以上 自陣側にあるか"""
    return (y - LINE_Y) * OPP <= -margin


def safe_ahead(car):
    """このまま前に進んでも、壁にも相手コートにも出ないか"""
    x, y = point_ahead(car, FRONT_OFFSET)
    return in_board(x, y) and our_side(y, LINE_MARGIN)


def clamp_point(x, y):
    """目標点を、盤の内側かつ自陣側に収める"""
    m = SAFE_MARGIN
    x = max(m, min(BOARD_W - m, x))
    y = max(m, min(BOARD_H - m, y))
    if not our_side(y, LINE_MARGIN + FRONT_OFFSET):
        y = LINE_Y - OPP * (LINE_MARGIN + FRONT_OFFSET)
    return x, y


def steer_to(car, x, y):
    """点(x, y) へ向かう。戻り値は (距離, ずれ, 動作の説明)"""
    dist, err = aim(car, x, y)
    if abs(err) > TURN_IN_PLACE_DEG:
        spin(err)
        return dist, err, "旋回"
    if not safe_ahead(car):
        stop()
        return dist, err, "前方が壁/ラインに近い"
    diff = KP * err                       # ずれに比例して左右の速さを変える
    drive_lr(FWD_SPEED + diff, FWD_SPEED - diff)
    return dist, err, "前進"


# ---------------- ブロックの見張り ----------------
class Watcher:
    """各ブロックの「一番自陣寄りにあった位置」を覚えておき、相手側へ動いた量を調べる"""

    def __init__(self):
        self.best = {}        # name -> 一番自陣寄りだった y
        self.cool = {}        # name -> この時刻までは狙わない

    def update(self, objs):
        for name, o in objs.items():
            b = self.best.get(name)
            if b is None or (o["y"] - b) * OPP < 0:
                self.best[name] = o["y"]

    def loss(self, o):
        """一番自陣寄りだった位置から、相手側へ動いた量[px]"""
        return (o["y"] - self.best.get(o["id"], o["y"])) * OPP

    def pulled(self, objs, now):
        """相手に引かれているブロック（引かれた量が多い順）"""
        res = [o for o in objs.values()
               if self.loss(o) >= PULLED_DIST
               and o["id"] not in SKIP_NAMES
               and now >= self.cool.get(o["id"], 0)
               and in_board(o["x"], o["y"])
               and our_side(o["y"], BLOCK_LINE_MARGIN)]
        return sorted(res, key=self.loss, reverse=True)


# ---------------- つかむ → 引き返す ----------------
def grab_and_pull(target, grab_y):
    print(f"{target} をつかむ")
    gripper(True)
    print("引き返す")
    t0 = time.time()
    while time.time() - t0 < PULL_TIME:
        c, objs = car_receiver.get_location()
        if c is not None and not in_board(*point_ahead(c, -FRONT_OFFSET)):
            print("  後ろが壁に近いので引くのをやめる")
            break
        o = objs.get(target)
        if o is not None and (grab_y - o["y"]) * OPP >= PULL_DIST:
            print(f"  {PULL_DIST}px 引き返した")
            break
        drive_lr(-PULL_SPEED, -PULL_SPEED)
        time.sleep(LOOP_DT)
    stop()
    print("はなす")
    gripper(False)
    drive_lr(-FWD_SPEED, -FWD_SPEED)
    time.sleep(BACK_OFF_TIME)
    stop()


# ---------------- メイン（状態遷移） ----------------
def main():
    car_receiver.start()
    gripper(False)                      # 最初は開いておく

    t_start = None                      # 2分を数え始める時刻
    if USE_LINE_START:
        t_start = time.time()           # ライントレースで動き出した時点からスタート
        line_start()

    print("カメラの制御に切り替え（データが来なければ board_vision.py を確認）")
    watch = Watcher()
    state, target, final = "STANDBY", None, False
    standby = clamp_point(STANDBY_X, LINE_Y - OPP * STANDBY_DEPTH)
    t_lost = time.time()
    last_print = 0

    while True:
        car, objs = car_receiver.get_location()
        now = time.time()

        if car is not None and t_start is None:
            t_start = now
            print("スタート")
        if t_start is not None and now - t_start > MATCH_TIME:
            stop()
            print("競技時間終了。停止します")
            return

        watch.update(objs)
        msg = ""

        # ---- 車が見えない／通信が切れた → その場で止まる ----
        if car is None:
            stop()
            msg = "車が見えません（QR・通信を確認）"

        # ---- 待機：引かれたブロックが出るまで待機位置で待つ ----
        elif state == "STANDBY":
            pulled = watch.pulled(objs, now)
            if pulled:
                target, final, t_lost = pulled[0]["id"], False, now
                state = "APPROACH"
                stop()
                msg = f"{target} が {watch.loss(pulled[0]):.0f}px 引かれた → 取り返しに行く"
            else:
                dist, err = aim(car, *standby)
                if dist > STANDBY_TOL:
                    _, _, act = steer_to(car, *standby)
                    msg = f"待機位置へ（残り {dist:.0f}px）{act}"
                else:
                    face = wrap180(FACE_LINE_ANGLE - car["angle"]) * TURN_SIGN
                    if abs(face) > 20:
                        spin(face)                  # ラインの方を向く
                        msg = "待機：向きを合わせる"
                    else:
                        stop()
                        msg = "待機中（相手が引くのを見張り中）"

        # ---- ブロックの真後ろに回り込んでから近づく ----
        elif state == "APPROACH":
            obj = objs.get(target)
            if obj is None or not our_side(obj["y"], 0):     # ラインを越えたら諦める
                stop()
                if obj is not None or now - t_lost > LOST_TIME:
                    state = "STANDBY"               # 見失った／相手コートに入った → 諦めて待機
                msg = f"{target} を見失い中"
            else:
                t_lost = now
                behind = (car["y"] - obj["y"]) * OPP     # − なら車がブロックより自陣側
                if not final and behind <= -STAGE_DIST * 0.6 and abs(car["x"] - obj["x"]) <= SIDE_TOL:
                    final = True                    # 真後ろに着いた → ブロックへ向かう
                if final and behind > 0:
                    final = False                   # ブロックを追い越した → 回り込み直し

                if not final:
                    sx, sy = clamp_point(obj["x"], obj["y"] - OPP * STAGE_DIST)
                    dist, err, act = steer_to(car, sx, sy)
                    msg = f"{target} の後ろへ回り込み（残り {dist:.0f}px）{act}"
                else:
                    dist, err = aim(car, obj["x"], obj["y"])
                    if dist <= GRAB_DIST and abs(err) <= GRAB_ALIGN_DEG:
                        stop()
                        state = "GRAB"
                    elif dist <= GRAB_DIST:
                        spin(err)
                        msg = "向きを合わせる"
                    else:
                        _, _, act = steer_to(car, obj["x"], obj["y"])
                        msg = act
                    msg = f"{target}: 距離 {dist:.0f}px ずれ {err:+.0f}° {msg}"

        # ---- つかむ → 引き返す → はなす ----
        elif state == "GRAB":
            obj = objs.get(target)
            grab_y = obj["y"] if obj else car["y"]
            grab_and_pull(target, grab_y)
            watch.cool[target] = time.time() + COOLDOWN
            state = "STANDBY"

        if msg and now - last_print > 0.5:
            print(f"[{state}] {msg}")
            last_print = now
        time.sleep(LOOP_DT)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断されました。")
    finally:
        stop()          # どの経路で抜けても必ずモーターを止める
