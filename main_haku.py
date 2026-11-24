"""
main_haku.py - ロボット綱引き 自動制御「後引き」作戦（ラズパイで実行）

  ノートPC の board_vision.py が天井カメラで「車の位置・向き」と「ブロックの位置」を送る
  → car_receiver.py で受け取る → このプログラムが車を動かす

  作戦：相手が綱を引いてブロックが相手側へ動いたら、取りに行って引き返す
    （最初）  line_start（USE_LINE_START = True のとき）
               1. 最初の物体（緑）へカメラで進む → car_pos_obj で止まる → サーボでつかむ
                  → 18cm 後退して引く（途中で止まったらOK）→ はなす → 中央の白線(HOME_LINE_Y)まで後退
               2. 白線の上で、相手に引かれた物体をカメラで見つける
               3. 90度回って、白線に沿ってライントレースで進み、x 座標を物体に合わせる
               4. 90度回って、ライントレース＋カメラで car_pos_obj まで進む → つかむ
                  → 18cm 引く → はなす → 中央の白線まで後退 → 2 に戻る
               ※ car_pos_obj = 物体の y − 腕の長さ（ARM_LEN_CM）
               （LINE_LOOP = True なら 2〜4 を競技終了までくり返す。False なら下のカメラ制御へ）
    STANDBY  : ラインの方を向いて待つ。各ブロックの位置を見張る
               → ブロックが「一番自陣寄りにあった位置」から PULLED_DIST 以上
                 相手側へ動いたら「相手に引かれた」と判断
    APPROACH : MOVE  … カメラで、ブロックの白線（レーン）の自陣側の点へ移動
               FACE  … その場でラインの方を向く（少しずつ回して行き過ぎを防ぐ）
               LINE  … ライントレースでブロックへ一直線に進む
    GRAB     : つかむ → 後退して自陣側へ引く → はなして下がる → その場で STANDBY
               （後退中は カメラの向き＋ラインセンサ で真っすぐ下がる）
    2分（MATCH_TIME）たったら全停止

  ※ ブロックは「ラインに垂直な白線（レーン）」の上にある前提。
    レーンの x 座標が分かっていれば LANE_XS に書く（空ならブロックの x を使う）

使い方
  python3 main_haku.py          自動制御
  python3 main_haku.py sensor   ラインセンサの値を表示するだけ（まずこれで確認！）
  python3 main_haku.py line     ライントレースだけを LINE_TEST_TIME 秒試す（ゲイン調整用）
  python3 main_haku.py back     後退だけを LINE_TEST_TIME 秒試す（カメラ必要）

  1. ノートPC で board_vision.py を起動し、四隅の指定とブロックの登録をしておく
  2. sensor モードで「白線の上 = ■」「左右の順番」が正しいか確かめる
  3. line モードでライントレースがフラフラしないか確かめる
  4. 最初は DRY_RUN = True で、モーターを回さずに表示だけ確認するのがおすすめ
"""
import math
import sys
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
LANE_XS = []                  # ブロックが乗っている白線（レーン）の x 座標[px]。例: [150, 300, 450]
                              # 空なら、ブロックの x をそのままレーンとして使う

# --- 相手に引かれたかの判断 ---
PULLED_DIST = 25        # 一番自陣寄りにあった位置から、これ以上[px]相手側へ動いたら「引かれた」
BLOCK_LINE_MARGIN = 20  # ラインからこの距離[px]以内のブロックは、相手コートに入る恐れがあるので狙わない
COOLDOWN = 3.0          # 引き返した後、同じブロックをこの秒数は狙わない
SKIP_NAMES = []         # 狙わないブロックの名前（例: ["object3"]）

# --- 最初の待機位置 ---
STANDBY_X = BOARD_W // 2                 # 最初に待機する場所[px]（自陣の真ん中あたり）
STANDBY_DEPTH = 90                       # ラインから自陣側へこの距離[px]の所で待つ
STANDBY_TOL = 30                         # 待機位置からこの距離[px]以内なら「着いた」

# --- カメラで走る（レーンの手前まで移動するとき）---
FWD_SPEED = 40          # 前進の基本速度[%]
TURN_SPEED = 35         # その場旋回の速度[%]
KP = 0.5                # 向きのずれ[度] → 左右の速度差[%]（大きいとフラフラする）
HEAD_DEADBAND = 4       # 向きのずれがこれ以下[度]なら補正しない（カメラの揺れで蛇行しないように）
TURN_IN_PLACE_DEG = 30  # 向きのずれがこれより大きければ、その場で旋回してから進む
SPIN_SLOW_DEG = 35      # ずれがこれ以下[度]なら、旋回を「少し回して止まる」をくり返す（行き過ぎ防止）
SPIN_PULSE = 0.06       # 少しずつ回すときの1回の回転時間[秒]
FACE_TOL = 6            # ラインの方を向いたとみなす向きのずれ[度]
TURN_SIGN = 1           # 目標と逆向きに曲がっていく場合は -1 にする

# --- 回り込み・つかむ ---
STAGE_DIST = 110        # ブロックの自陣側、この距離[px]のレーン上の点にまず移動する
STAGE_TOL = 15          # その点からこの距離[px]以内なら「着いた」
LANE_TOL = 40           # ライントレース中、車がレーンから横にこれ以上[px]ずれたら移動し直す
BOARD_W_CM = 180       # 盤面（補正後の画像の横 BOARD_W[px]）の実際の横幅[cm]。測って書く
PX_PER_CM = BOARD_W / BOARD_W_CM
ARM_LEN_CM = 15         # 車の中心（QRの中心）から腕（つかむ所）までの長さ[cm]。測って書く
ARM_LEN = ARM_LEN_CM * PX_PER_CM
#   car_pos_obj = 物体の y − 腕の長さ（自陣側へ）。車の中心の y がここに来たらつかむ
GRAB_TOL = 5            # car_pos_obj までの残りがこれ以下[px]で止まってつかむ
GRAB_SLOW_DIST = 30     # car_pos_obj までこの距離[px]以内になったら MIN_SPEED に落とす（行き過ぎ防止）
FRONT_OFFSET = 40       # 車の中心から先端（後端）までの距離[px]
SERVO_PIN = "P0"
SERVO_OPEN = -60        # 開いた角度[度]（servo_sweep.py で確かめた値に変える）
SERVO_CLOSE = 30        # 閉じた角度[度]
SERVO_WAIT = 0.6        # サーボが動き終わるまで待つ時間[秒]

# --- 引き返す（後退）---
PULL_SPEED = 40         # 後退して引く速度[%]
PULL_CM = 18            # つかんだ位置からこの距離[cm]後退して引いたら、はなす
PULL_TIME = 5.0         # 18cm に届かなくても、この秒数で引くのをやめる
STALL_TIME = 0.8        # 引いている途中、この秒数ほとんど動かなければ「止まった → OK」として、はなす
STALL_PX = 3            # この距離[px]以下しか動いていなければ「止まっている」とみなす
BACK_OFF_TIME = 0.6     # はなした後に下がる時間[秒]
KP_BACK = 0.6           # 後退中、向きのずれ[度] → 左右の速度差[%]（後退はこれが主役）
LINE_BACK_KP = 2        # 後退中のラインセンサの補正（0 にすると使わない。大きくすると不安定になる）

# --- スタート直後（line_start）---
#   1. 最初の物体（START_COLOR）へカメラで進む → car_pos_obj で止まる → つかむ
#      → 18cm 後退して引く → はなす → 中央の白線まで後退
#   2. 相手に引かれた物体をカメラで見つける
#   3. 90度回って、中央の白線に沿ってライントレースで進み、x 座標を物体に合わせる
#   4. 90度回って、ライントレース＋カメラで car_pos_obj まで進む → つかむ
#      → 18cm 後退して引く → はなす → 中央の白線まで後退  → 2 に戻る
USE_LINE_START = True   # True: 上の流れで動く（False なら最初からカメラだけの制御）
START_COLOR = "green"   # 最初に引く物体の色（見つからなければ正面の一番近い物体）
HOME_LINE_Y = 100       # 自陣の「中央の白線」（中央ライン LINE_Y と平行な横線）の y[px]。実際の値に変える
HOME_TOL = 10           # 白線よりこれ以上[px]後ろに下がりすぎたら、前進して白線に戻る
LINE_LOOP = True        # True: 2〜4 を競技終了までくり返す / False: 1回やったらカメラだけの制御へ
START_SPEED = 40        # 正面の物体へ進む速度[%]
FRONT_CONE_DEG = 25     # 車の正面からこの角度[度]以内にある物体を「正面の物体」とする
START_FIND_TIME = 2.0   # 正面の物体をこの秒数探しても見つからなければ、つかむのを飛ばす
START_GO_TIME = 5.0     # 正面の物体へ進むのにこの秒数かかったら諦める
BACK_LINE_TIME = 5.0    # 中央の白線まで後退する最大時間[秒]
LINE_SPEED = 35         # 中央の白線に沿って横へ動く速度[%]
X_TOL = 10              # レーンの x とのずれがこれ以下[px]なら「x が合った」
X_SLOW_DIST = 40        # レーンまでこの距離[px]以内になったら MIN_SPEED に落とす（行き過ぎ防止）
ALIGN_TIME = 6.0        # x 合わせ・y 合わせのそれぞれ、この秒数で諦める
TURN_TIMEOUT = 4.0      # その場旋回でこの秒数たっても向けなければ諦める

# --- ラインセンサ ---
LINE_PINS = [17, 27, 22, 4]   # GPIO番号を「左から順」に（sensor モードで確認する）
WHITE_VALUE = 0         # 白線の上でセンサが出す値（sensor モードで白線の上が □ なら 1 にする）
LINE_WEIGHTS = [-3, -1, 1, 3]  # 各センサの位置（左が −、右が +）
LINE_KP = 6             # 線のずれ → 左右の速度差[%]（フラフラするなら下げる、曲がり切れないなら上げる）
LINE_APPROACH_SPEED = 32    # ブロックへライントレースで近づく速度[%]
LINE_LOST_TIME = 0.3    # 線を見失ってもこの秒数は直前の曲がり方を続ける。過ぎたらカメラの向きで直進
CROSS_COUNT = 3         # 白のセンサがこの個数以上なら「横線（交差点）」→ 真っすぐ進む
MIN_SPEED = 25          # これより遅いとTTモーターが回らない[%]
LINE_TEST_TIME = 4.0    # line / back モードで走る時間[秒]

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
    base.py の M1 = 左, M2 = 右（base.py の「右回転 = (1, -1)」より）
    MIN_SPEED 未満の値は、半分未満なら 0、それ以上なら MIN_SPEED にする
    （小さい値を全部 MIN_SPEED に上げると左右差がなくなり、曲がれなくなるため）"""
    if DRY_RUN:
        return
    for pwm, dir_pin, sign, v in ((base.m1_pwm, base.m1_dir, base.M1_SIGN, left),
                                  (base.m2_pwm, base.m2_dir, base.M2_SIGN, right)):
        v = max(-100, min(100, v))
        if abs(v) < MIN_SPEED * 0.5:
            base.set_motor(pwm, dir_pin, 0, 0)
        else:
            d = 1 if v > 0 else -1
            base.set_motor(pwm, dir_pin, d * sign, max(abs(v), MIN_SPEED))


def stop():
    if not DRY_RUN:
        base.stop()


def spin(err):
    """その場旋回（err + なら右へ）"""
    t = TURN_SPEED if err > 0 else -TURN_SPEED
    drive_lr(t, -t)


def turn_step(err):
    """向きを合わせる1回分。ずれが小さいときは「少し回して止める」
    （カメラの遅れで回りすぎるのを防ぐ。止まっている間に次の位置が届く）"""
    spin(err)
    if abs(err) <= SPIN_SLOW_DEG:
        time.sleep(SPIN_PULSE)
        stop()


def gripper(close):
    if DRY_RUN:
        print("  [サーボ]", "閉じる" if close else "開く")
        return
    servo.angle(SERVO_CLOSE if close else SERVO_OPEN)
    time.sleep(SERVO_WAIT)


def clamp(v, lim):
    return max(-lim, min(lim, v))


# ---------------- ラインセンサ ----------------
# active_state=True を付けないと「floating, but active_state is not defined」エラーになる
line_sensors = [Pin(p, Pin.IN, active_state=True) for p in LINE_PINS]


def line_read():
    """[左, 左中, 右中, 右]  白=1, 黒=0"""
    return [1 if s.value() == WHITE_VALUE else 0 for s in line_sensors]


class LineFollower:
    """ラインセンサで線に沿って走る。見失ったときはカメラの向きで真っすぐ進む"""

    def __init__(self):
        self.last_pos = 0.0     # 最後に見えた線のずれ（左 −, 右 +）
        self.t_seen = 0.0       # 最後に線が見えた時刻
        self.cross = False      # 今、横線の上か

    def reset(self):
        self.last_pos, self.t_seen = 0.0, time.time()

    def read(self):
        """線のずれを返す。見えなければ None"""
        w = line_read()
        n = sum(w)
        self.cross = n >= CROSS_COUNT
        if n == 0:
            return None
        if self.cross:                      # 横線 → ずれ 0 として真っすぐ
            pos = 0.0
        else:
            pos = sum(k * v for k, v in zip(LINE_WEIGHTS, w)) / n
        self.last_pos, self.t_seen = pos, time.time()
        return pos

    def forward(self, speed, cam_err=None):
        """前進しながら線に沿う。cam_err はカメラで見た目標へのずれ[度]（無ければ None）
        戻り値: 状態の説明"""
        pos = self.read()
        if pos is not None:
            st = f"ライン ずれ{pos:+.1f}"
        elif time.time() - self.t_seen < LINE_LOST_TIME or cam_err is None:
            pos = self.last_pos             # 一瞬の見失い → 直前の曲がり方を続ける
            st = "ライン見失い（直前の向きで）"
        else:                               # しばらく見えない → カメラの向きで直進
            e = 0 if abs(cam_err) < HEAD_DEADBAND else cam_err
            diff = clamp(KP * e, speed)
            drive_lr(speed + diff, speed - diff)
            return f"ラインなし→カメラで直進 ずれ{cam_err:+.0f}°"
        # 内側の車輪は「止まる」まで。逆回転はさせない（その場旋回するとフラフラするため）
        diff = clamp(LINE_KP * pos, speed)
        drive_lr(speed + diff, speed - diff)
        return st

    def backward(self, speed, head_err=None):
        """後退しながら真っすぐ下がる。head_err は「ラインの方を向く角度」とのずれ[度]
        ※ センサは車の前側にあるので、後退中に線だけで曲がると不安定になる。
          カメラの向きを主に使い、線は弱く（前進と逆向きに）補正する。"""
        pos = self.read()
        diff = 0.0
        if head_err is not None:
            diff = KP_BACK * (0 if abs(head_err) < HEAD_DEADBAND else head_err)
            if pos is not None:
                diff -= LINE_BACK_KP * pos
        diff = clamp(diff, speed * 0.6)
        drive_lr(-speed + diff, -speed - diff)


lf = LineFollower()


# ---------------- 競技時間 ----------------
t_end = None            # この時刻を過ぎたら止まる（start_clock() で決まる）


def start_clock():
    global t_end
    if t_end is None:
        t_end = time.time() + MATCH_TIME
        print("スタート")


def time_up():
    return t_end is not None and time.time() > t_end


# ---------------- 位置の計算 ----------------
def wrap180(a):
    """角度を -180〜180 に収める"""
    return (a + 180) % 360 - 180


def aim(car, x, y):
    """点(x, y) までの距離[px] と、向きのずれ[度]（+ = 右側, - = 左側）"""
    dx, dy = x - car["x"], y - car["y"]
    bearing = math.degrees(math.atan2(dy, dx))       # 画面座標（右=0, 下=+90）
    return math.hypot(dx, dy), wrap180(bearing - car["angle"]) * TURN_SIGN


def face_err(car):
    """ラインの方を向く角度とのずれ[度]（+ なら右へ回る）"""
    return wrap180(FACE_LINE_ANGLE - car["angle"]) * TURN_SIGN


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


def lane_of(x):
    """ブロックの x に一番近いレーンの x（LANE_XS が空ならそのまま）"""
    return min(LANE_XS, key=lambda lx: abs(lx - x)) if LANE_XS else x


def grab_rest(car, obj):
    """car_pos_obj（= 物体の y − 腕の長さ）まで、あと何[px]進むか。0 以下なら着いた（行き過ぎ）"""
    car_pos_obj = obj["y"] - OPP * ARM_LEN
    return (car_pos_obj - car["y"]) * OPP


def steer_to(car, x, y):
    """点(x, y) へカメラを見ながら向かう。戻り値は (距離, ずれ, 動作の説明)
    ※ 着いたかどうかは呼ぶ側で判定する（近すぎると方向がブレて蛇行するため）"""
    dist, err = aim(car, x, y)
    if abs(err) > TURN_IN_PLACE_DEG:
        turn_step(err)
        return dist, err, "旋回"
    if not safe_ahead(car):
        stop()
        return dist, err, "前方が壁/ラインに近い"
    e = 0 if abs(err) < HEAD_DEADBAND else err
    diff = clamp(KP * e, FWD_SPEED)         # ずれに比例して左右の速さを変える
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


watch = Watcher()


def sense():
    """最新の (車, 物体) を取り、ブロックの見張りも更新する"""
    car, objs = car_receiver.get_location()
    watch.update(objs)
    return car, objs


# ---------------- つかむ → 引き返す ----------------
def back_straight(duration, stop_fn=None):
    """カメラの向き＋ラインセンサで真っすぐ後退する。stop_fn(car, objs) が True で止める"""
    t0 = time.time()
    while time.time() - t0 < duration and not time_up():
        c, objs = sense()
        if c is not None and not in_board(*point_ahead(c, -FRONT_OFFSET)):
            print("  後ろが壁に近いので止まる")
            break
        if stop_fn is not None and stop_fn(c, objs):
            break
        lf.backward(PULL_SPEED, face_err(c) if c is not None else None)
        time.sleep(LOOP_DT)
    stop()


def pull_and_release(target):
    """サーボでつかむ → 真っすぐ後退して PULL_CM 引く（途中で止まったらそこでOK）→ はなす"""
    print(f"{target} をつかむ")
    gripper(True)
    print(f"{PULL_CM}cm 引く（真っすぐ後退）")
    pull_px = PULL_CM * PX_PER_CM
    st = {"y0": None, "x": None, "y": None, "t": time.time()}

    def pulled_enough(c, objs):
        if c is None:
            return False
        now = time.time()
        if st["y0"] is None:
            st.update(y0=c["y"], x=c["x"], y=c["y"], t=now)
        if (st["y0"] - c["y"]) * OPP >= pull_px:
            print(f"  {PULL_CM}cm 引いた")
            return True
        if math.hypot(c["x"] - st["x"], c["y"] - st["y"]) > STALL_PX:
            st.update(x=c["x"], y=c["y"], t=now)        # 動いている
        elif now - st["t"] > STALL_TIME:
            moved = (st["y0"] - c["y"]) * OPP / PX_PER_CM
            print(f"  {moved:.0f}cm で止まった → OK")
            return True
        return False

    back_straight(PULL_TIME, pulled_enough)
    print("はなす")
    gripper(False)
    watch.cool[target] = time.time() + COOLDOWN


# ---------------- スタート直後：中央の白線を使う作戦 ----------------
def at_home_line(c):
    """車の中心が中央の白線（HOME_LINE_Y）まで下がったか"""
    return (c["y"] - HOME_LINE_Y) * OPP <= 0


def back_to_line():
    """中央の白線まで真っすぐ後退する（下がりすぎていたら前進して戻る）"""
    print("中央の白線まで後退")

    def reached(c, objs):
        if c is None:
            return lf.cross             # カメラが無いときはラインセンサの横線で止める
        return at_home_line(c)

    back_straight(BACK_LINE_TIME, reached)

    # 引いた結果、白線より後ろにいたら前進して白線に戻る
    t0 = time.time()
    while not time_up() and time.time() - t0 < BACK_LINE_TIME:
        c, _ = sense()
        if c is None or (c["y"] - HOME_LINE_Y) * OPP >= -HOME_TOL:
            break
        e = face_err(c)
        diff = clamp(KP * (0 if abs(e) < HEAD_DEADBAND else e), MIN_SPEED)
        drive_lr(MIN_SPEED + diff, MIN_SPEED - diff)
        time.sleep(LOOP_DT)
    stop()


def reach_and_pull(target, car, obj, e, line):
    """car_pos_obj へ1ステップ近づく。着いたら つかむ→引く→はなす→白線まで後退 して "done"。
    進めないときは "fail"、進んだら None。line=True ならライントレース、False ならカメラで進む"""
    rest = grab_rest(car, obj)
    if rest <= GRAB_TOL or (not safe_ahead(car) and rest <= GRAB_TOL + 20):
        stop()
        print(f"  car_pos_obj に到着（残り {rest:.0f}px）")
        pull_and_release(target)
        back_to_line()
        return "done"
    if not safe_ahead(car):
        print("  前方がラインに近く届かない")
        return "fail"
    speed = (LINE_APPROACH_SPEED if line else START_SPEED) if rest > GRAB_SLOW_DIST else MIN_SPEED
    if line:
        lf.forward(speed, cam_err=e)            # 線に沿う。見失ったらカメラで直進
    elif abs(e) > TURN_IN_PLACE_DEG:
        turn_step(e)
    else:
        diff = clamp(KP * (0 if abs(e) < HEAD_DEADBAND else e), speed)
        drive_lr(speed + diff, speed - diff)
    return None


def turn_to(ang):
    """その場で角度 ang[度]（右=0, 下=+90, 上=-90, 左=180）を向く。向けたら True"""
    t0 = time.time()
    while time.time() - t0 < TURN_TIMEOUT and not time_up():
        car, _ = sense()
        if car is None:
            stop()
        else:
            e = wrap180(ang - car["angle"]) * TURN_SIGN
            if abs(e) <= FACE_TOL:
                stop()
                return True
            turn_step(e)
        time.sleep(LOOP_DT)
    stop()
    print(f"  {ang}° を向けなかった")
    return False


def track(target, t_lost):
    """target を取り出す。見失って LOST_TIME 過ぎた／相手コートに入ったら "lost" を返す"""
    _, objs = car_receiver.get_location()
    obj = objs.get(target)
    if obj is None:
        return ("lost" if time.time() - t_lost > LOST_TIME else None), t_lost
    if not our_side(obj["y"], 0):
        return "lost", t_lost
    return obj, time.time()


def grab_front():
    """1. 最初の物体（START_COLOR）へカメラで進み、つかむ→引く→はなす→中央の白線まで後退"""
    print(f"最初の物体（{START_COLOR}）を探す")
    target, t0 = None, time.time()
    while target is None and time.time() - t0 < START_FIND_TIME:
        car, objs = sense()
        if car is not None:
            color, front = [], []
            for o in objs.values():
                if o["id"] in SKIP_NAMES or not our_side(o["y"], 0):
                    continue
                d, e = aim(car, o["x"], o["y"])
                if o.get("color") == START_COLOR:
                    color.append((d, o["id"]))
                if abs(e) <= FRONT_CONE_DEG:
                    front.append((d, o["id"]))
            if color or front:
                target = min(color or front)[1]   # 指定の色を優先。無ければ正面で一番近いもの
        time.sleep(LOOP_DT)
    if target is None:
        print("最初の物体が見つからない → 中央の白線へ")
        back_to_line()
        return

    print(f"{target} へ進む")
    t0 = t_lost = time.time()
    while not time_up():
        if time.time() - t0 > START_GO_TIME:
            print("  時間切れ")
            break
        car, _ = sense()
        obj, t_lost = track(target, t_lost)
        if obj == "lost":
            print(f"  {target} を見失った")
            break
        if car is None or obj is None:
            stop()
            time.sleep(LOOP_DT)
            continue
        _, e = aim(car, obj["x"], obj["y"])
        r = reach_and_pull(target, car, obj, e, line=False)
        if r == "done":
            return
        if r == "fail":
            break
        time.sleep(LOOP_DT)
    stop()
    back_to_line()


def wait_pulled():
    """2. 中央の白線の上で、相手に引かれた物体が出るまで待つ。物体の名前を返す（時間切れなら None）"""
    print("相手に引かれた物体を見張り中")
    turn_to(FACE_LINE_ANGLE)
    while not time_up():
        _, objs = sense()
        pulled = watch.pulled(objs, time.time())
        if pulled:
            o = pulled[0]
            print(f"{o['id']} が {watch.loss(o):.0f}px 引かれた → 取り返しに行く")
            return o["id"]
        time.sleep(LOOP_DT)
    return None


def align_x(target):
    """3. 中央の白線に沿ってライントレースし、ブロックのレーンの x に合わせる"""
    car, objs = sense()
    obj = objs.get(target)
    if car is None or obj is None:
        return False
    lane = lane_of(obj["x"])
    if abs(lane - car["x"]) <= X_TOL:
        return True
    ang = 0 if lane > car["x"] else 180             # 右へ行くなら 0°、左なら 180°
    sgn = 1 if ang == 0 else -1
    print(f"x 合わせ: x={car['x']:.0f} → レーン x={lane:.0f}")
    if not turn_to(ang):
        return False
    lf.reset()
    t0 = t_lost = time.time()
    while not time_up() and time.time() - t0 < ALIGN_TIME:
        car, _ = sense()
        obj, t_lost = track(target, t_lost)
        if obj == "lost":
            break
        if car is None:
            stop()
            time.sleep(LOOP_DT)
            continue
        if obj is not None:
            lane = lane_of(obj["x"])
        rest = (lane - car["x"]) * sgn                  # レーンまでの残り[px]（− なら行き過ぎ）
        if rest <= X_TOL:
            stop()
            print("  x が合った")
            return True
        if not safe_ahead(car):
            print("  前方が壁に近い")
            break
        speed = LINE_SPEED if rest > X_SLOW_DIST else MIN_SPEED
        lf.forward(speed, cam_err=wrap180(ang - car["angle"]) * TURN_SIGN)
        time.sleep(LOOP_DT)
    stop()
    print("  x 合わせ失敗")
    return False


def align_y(target):
    """4. ラインの方を向き、レーンをライントレース＋カメラで car_pos_obj まで進む
    → つかむ → 引く → はなす → 中央の白線まで後退"""
    print("y 合わせ: レーンを car_pos_obj まで進む")
    if not turn_to(FACE_LINE_ANGLE):
        return False
    lf.reset()
    t0 = t_lost = time.time()
    while not time_up() and time.time() - t0 < ALIGN_TIME:
        car, _ = sense()
        obj, t_lost = track(target, t_lost)
        if obj == "lost":
            break
        if car is None or obj is None:
            stop()
            time.sleep(LOOP_DT)
            continue
        _, e = aim(car, obj["x"], obj["y"])
        r = reach_and_pull(target, car, obj, e, line=True)
        if r == "done":
            return True
        if r == "fail":
            break
        time.sleep(LOOP_DT)
    stop()
    print("  y 合わせ失敗")
    return False


def line_start():
    """スタート直後の作戦。正面の物体を引く → 引かれた物体を中央の白線を使って取り返す"""
    grab_front()
    while not time_up():
        target = wait_pulled()
        if target is None:
            return
        if not (align_x(target) and align_y(target)):
            watch.cool[target] = time.time() + COOLDOWN     # 失敗したらしばらく狙わない
            back_to_line()
        if not LINE_LOOP:
            return


# ---------------- メイン（状態遷移） ----------------
def main():
    car_receiver.start()
    gripper(False)                      # 最初は開いておく

    if USE_LINE_START:
        start_clock()                   # 動き出した時点から2分を数える
        line_start()

    print("カメラの制御に切り替え（データが来なければ board_vision.py を確認）")
    state, phase, target, lane = "STANDBY", "", None, 0
    standby = clamp_point(STANDBY_X, LINE_Y - OPP * STANDBY_DEPTH)
    if USE_LINE_START:
        c, _ = car_receiver.get_location()
        if c is not None:
            standby = clamp_point(c["x"], c["y"])   # 今いる場所で待つ
    t_lost = time.time()
    last_print = 0

    while True:
        car, objs = sense()
        now = time.time()

        if car is not None:
            start_clock()
        if time_up():
            stop()
            print("競技時間終了。停止します")
            return

        msg = ""

        # ---- 車が見えない／通信が切れた → その場で止まる ----
        if car is None:
            stop()
            msg = "車が見えません（QR・通信を確認）"

        # ---- 待機：引かれたブロックが出るまで待つ ----
        elif state == "STANDBY":
            pulled = watch.pulled(objs, now)
            if pulled:
                o = pulled[0]
                target, lane, t_lost = o["id"], lane_of(o["x"]), now
                state, phase = "APPROACH", "MOVE"
                stop()
                msg = f"{target} が {watch.loss(o):.0f}px 引かれた → レーン x={lane:.0f} から取り返しに行く"
            else:
                dist, _ = aim(car, *standby)
                if dist > STANDBY_TOL:
                    _, _, act = steer_to(car, *standby)
                    msg = f"待機位置へ（残り {dist:.0f}px）{act}"
                else:
                    fe = face_err(car)
                    if abs(fe) > FACE_TOL * 2:
                        turn_step(fe)               # ラインの方を向く
                        msg = "待機：向きを合わせる"
                    else:
                        stop()
                        msg = "待機中（相手が引くのを見張り中）"

        # ---- レーンの手前へ移動 → ラインの方を向く → ライントレースで一直線 ----
        elif state == "APPROACH":
            obj = objs.get(target)
            if obj is None or not our_side(obj["y"], 0):     # ラインを越えたら諦める
                stop()
                if obj is not None or now - t_lost > LOST_TIME:
                    state = "STANDBY"               # 見失った／相手コートに入った → 諦めて待機
                    standby = clamp_point(car["x"], car["y"])
                msg = f"{target} を見失い中"
            else:
                t_lost = now
                dist_b, err_b = aim(car, obj["x"], obj["y"])
                behind = (car["y"] - obj["y"]) * OPP     # − なら車がブロックより自陣側
                if phase != "MOVE" and (behind > 0 or abs(car["x"] - lane) > LANE_TOL):
                    phase = "MOVE"                  # 追い越した／レーンから外れた → 移動し直す

                if phase == "MOVE":
                    sx, sy = clamp_point(lane, obj["y"] - OPP * STAGE_DIST)
                    d, _ = aim(car, sx, sy)
                    if d <= STAGE_TOL:
                        stop()
                        phase = "FACE"
                        msg = "レーンの手前に到着"
                    else:
                        _, _, act = steer_to(car, sx, sy)
                        msg = f"レーンの手前へ（残り {d:.0f}px）{act}"

                elif phase == "FACE":
                    fe = face_err(car)
                    if abs(fe) <= FACE_TOL:
                        stop()
                        lf.reset()
                        phase = "LINE"
                        msg = "ラインの方を向いた → ライントレース開始"
                    else:
                        turn_step(fe)
                        msg = f"ラインの方へ回転 ずれ{fe:+.0f}°"

                elif phase == "LINE":
                    rest = grab_rest(car, obj)
                    if rest <= GRAB_TOL:
                        stop()
                        state = "GRAB"
                        msg = "car_pos_obj に到着"
                    elif not safe_ahead(car):
                        stop()
                        if rest <= GRAB_TOL + 20:
                            state = "GRAB"
                            msg = "前方がラインに近いのでここでつかむ"
                        else:
                            watch.cool[target] = now + COOLDOWN
                            state = "STANDBY"
                            standby = clamp_point(car["x"], car["y"])
                            msg = "前方がラインに近く届かない → 諦める"
                    else:
                        speed = LINE_APPROACH_SPEED if rest > GRAB_SLOW_DIST else MIN_SPEED
                        msg = lf.forward(speed, cam_err=err_b)
                msg = f"<{phase}> {target}: 距離 {dist_b:.0f}px  {msg}"

        # ---- つかむ → 引き返す → はなす → その場で待機 ----
        elif state == "GRAB":
            pull_and_release(target)
            back_straight(BACK_OFF_TIME)    # はなした後も真っすぐ下がる（次のブロックへ動きやすく）
            c, _ = car_receiver.get_location()
            c = c or car
            standby = clamp_point(c["x"], c["y"])   # 下がった場所で次を待つ（次のレーンへ近い）
            state = "STANDBY"

        if msg and now - last_print > 0.5:
            print(f"[{state}] {msg}")
            last_print = now
        time.sleep(LOOP_DT)


# ---------------- 確認用モード ----------------
def sensor_check():
    """ラインセンサの値を表示するだけ。白線の上で ■ になるか、左右の順番が合っているか確かめる"""
    print("白線の上 = ■ になるか確認。逆なら WHITE_VALUE を変える")
    print("車を右へずらすと、■ が左へ動けば順番OK。逆なら LINE_PINS の順を逆にする")
    while True:
        raw = [s.value() for s in line_sensors]
        w = line_read()
        pos = lf.read()
        p = "なし" if pos is None else f"{pos:+.1f}"
        print("左→右:", " ".join("■" if v else "□" for v in w), f"  生の値 {raw}  ずれ {p}")
        time.sleep(0.2)


def line_test():
    """ライントレースだけを試す（カメラ不要）"""
    print(f"{LINE_TEST_TIME}秒 ライントレースします（速度 {LINE_APPROACH_SPEED}, LINE_KP {LINE_KP}）")
    lf.reset()
    t0 = last = time.time()
    while time.time() - t0 < LINE_TEST_TIME:
        st = lf.forward(LINE_APPROACH_SPEED)
        if time.time() - last > 0.2:
            print(" ", st)
            last = time.time()
        time.sleep(0.02)
    stop()


def back_test():
    """後退だけを試す（カメラ必要）。ラインの方を向けて置いてから実行する"""
    car_receiver.start()
    time.sleep(1)
    print(f"{LINE_TEST_TIME}秒 真っすぐ後退します")
    back_straight(LINE_TEST_TIME)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
    try:
        {"auto": main, "sensor": sensor_check, "line": line_test, "back": back_test}[mode]()
    except KeyboardInterrupt:
        print("\n中断されました。")
    finally:
        stop()          # どの経路で抜けても必ずモーターを止める
