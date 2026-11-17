"""
main.py - 引かれた物体へ自動で向かう（ラズパイで実行）

流れ:
  1. 起動直後 BASELINE_TIME 秒間、各物体の位置を「基準位置」として覚える
     （この間は物体に触らない）
  2. どれかの物体が基準位置から PULL_CM 以上動いたら、それを目標にする
  3. 車を線の向き（目標のある側）に向け、ライントレースで進む
  4. 車と目標の ALIGN_AXIS 座標（y）が一致したら止まる
  5. 目標の方を向いて、線を使わずにカメラの向きを見ながら直進
  6. STOP_DIST_CM まで近づいたら停止。物体が元に戻されたら 2 から繰り返す

使い方:
  python3 main.py          自動制御
  python3 main.py check    動かさずに、センサ・車・物体の移動量を表示（設定の確認用）
  python3 main.py turntest 右回転を一瞬して、回転方向の設定が正しいか調べる
"""
import math, sys, time
import linetrace        # 注意: linetrace と base はどちらも import 時に reset_mcu() する。
import base             #       モータ設定が消えないよう、base を必ず後に import する
import car_receiver

# ==== 盤の設定（要測定） =====
BOARD_W, BOARD_H = 600, 400       # board_vision.py の BOARD_W, BOARD_H と同じ値
BOARD_W_CM, BOARD_H_CM = 180, 120 # ← ブラウザで四隅をクリックした範囲の実寸[cm]（横, 縦）

# ==== 動作の設定 =====
PULL_CM = 12          # この距離以上動いた物体を「引かれた」とみなす（15cm引く想定なので少し手前）
BASELINE_TIME = 2.0   # 起動後、基準位置を覚える時間[秒]
ALIGN_AXIS = "y"      # 線に沿って進むとき、目標と一致させる座標（"x" か "y"）
ALIGN_TOL_CM = 2      # この差以内なら「一致」
STOP_DIST_CM = 10     # 目標の中心からこの距離まで近づいたら停止
LOST_TIME = 2.0       # 目標や車がこの秒数見えなければ、あきらめて待機に戻る

# ==== 速度・ゲイン（0〜100） =====
SPEED_LINE = 35       # ライントレースの速度
KP_LINE = 10          # ライントレースの曲がる強さ
SPEED_TURN = 35       # その場回転の速度
SPEED_DRIVE = 35      # 直進の速度
KH_DRIVE = 0.8        # 直進中の向きの補正（1度ずれるごとに足す速度）
MIN_SPEED = 20        # これより遅いとモータが回らない場合の最低速度
TURN_SIGN = 1         # turntest で「angle が減った」と出たら -1 にする

# ==== ラインセンサ =====
# 車の左→右の順に並べる。linetrace.py のコメント（センサ1〜4）の順にしてある
SENSORS = [linetrace.line1, linetrace.line3, linetrace.line4, linetrace.line2]
LINE_ON = 1           # 白線の上で value() が返す値。python3 linetrace.py で確認して 0 か 1 に

LOOP_DT = 0.05        # 制御周期[秒]

# ---------------------------------------------------------------------
def wrap(a):
    """角度を -180〜180 に"""
    return (a + 180) % 360 - 180

def dist_cm(ax, ay, bx, by):
    return math.hypot((ax - bx) * BOARD_W_CM / BOARD_W, (ay - by) * BOARD_H_CM / BOARD_H)

def axis_cm(d):
    """ALIGN_AXIS 方向の差[px] → [cm]"""
    return d * (BOARD_H_CM / BOARD_H if ALIGN_AXIS == "y" else BOARD_W_CM / BOARD_W)

def drive_lr(left, right):
    """左右のモータを -100〜100 で回す（m1=左, m2=右。base.py の右回転=(1,-1)から）"""
    for v, pwm, pin, sign in ((left, base.m1_pwm, base.m1_dir, base.M1_SIGN),
                              (right, base.m2_pwm, base.m2_dir, base.M2_SIGN)):
        if abs(v) < 1:
            base.set_motor(pwm, pin, 0, 0)
        else:
            sp = min(100, max(MIN_SPEED, abs(v)))
            base.set_motor(pwm, pin, (1 if v > 0 else -1) * sign, sp)

def read_sensors():
    return [s.value() == LINE_ON for s in SENSORS]

# ---------------------------------------------------------------------
class AutoController:
    """カメラの座標とラインセンサから、左右のモータ速度を決める"""
    def __init__(self):
        self.state = "BASELINE"
        self.t_start = None
        self.base = {}          # 基準位置  id -> [x, y, color, 回数]
        self.target = None      # 目標の id
        self.last_seen = 0      # 目標と車を最後に両方見た時刻
        self.last_err = 0       # ライントレースで最後に線が見えた位置
        self.info = ""

    # --- 基準位置と「引かれた」判定 ---
    def _update_base(self, objs, now):
        if self.state == "BASELINE":
            for o in objs.values():
                b = self.base.setdefault(o["id"], [0, 0, o["color"], 0])
                b[0] += o["x"]; b[1] += o["y"]; b[3] += 1
            if now - self.t_start >= BASELINE_TIME:
                for b in self.base.values():
                    b[0] /= b[3]; b[1] /= b[3]; b[3] = 1
                print(f"基準位置を記録: {len(self.base)} 個")
                self.state = "IDLE"
            return
        for o in objs.values():
            if o["id"] not in self.base:
                # 番号が変わった物体 → 今見えていない同じ色の基準位置で一番近いものを引き継ぐ
                cands = [(dist_cm(o["x"], o["y"], b[0], b[1]), k) for k, b in self.base.items()
                         if k not in objs and b[2] == o["color"]]
                if cands:
                    old = min(cands)[1]
                    self.base[o["id"]] = self.base.pop(old)
                    if self.target == old:            # 目標の番号が変わったら追いかける
                        self.target = o["id"]
                else:
                    self.base[o["id"]] = [o["x"], o["y"], o["color"], 1]
            b = self.base[o["id"]]
            if self.displacement(o) < 1:          # 1cm未満のブレだけ、基準をごくゆっくり追従
                b[0] += (o["x"] - b[0]) * 0.01    # （ゆっくり引かれても基準がついていかないように）
                b[1] += (o["y"] - b[1]) * 0.01

    def displacement(self, o):
        b = self.base.get(o["id"])
        return 0 if b is None else dist_cm(o["x"], o["y"], b[0], b[1])

    # --- 1周期ぶんの制御 ---
    def step(self, car, objs, sensors, now):
        """戻り値: (左, 右) のモータ速度"""
        if self.t_start is None:
            if not objs:
                self.info = "物体のデータ待ち"
                return 0, 0
            self.t_start = now
        self._update_base(objs, now)
        if self.state == "BASELINE":
            self.info = f"基準位置を記録中（物体に触らないで） {now - self.t_start:.1f}s"
            return 0, 0

        # 目標を決める
        if self.state == "IDLE":
            pulled = [(self.displacement(o), o["id"]) for o in objs.values()
                      if self.displacement(o) >= PULL_CM]
            if not pulled:
                self.info = "待機中（引かれた物体なし）"
                return 0, 0
            d, self.target = max(pulled)
            self._go("ORIENT", f"目標 {self.target}（{d:.0f}cm 引かれた）")
            self.last_seen = now

        t = objs.get(self.target)
        if car is None or t is None:            # 見えない → 止まって待つ
            if now - self.last_seen > LOST_TIME:
                self._go("IDLE", f"{'車' if car is None else self.target} を見失ったので待機に戻る")
                self.target = None
            self.info = "車か目標が見えないので一時停止"
            return 0, 0
        self.last_seen = now

        heading = car["angle"]
        if ALIGN_AXIS == "y":
            diff = axis_cm(t["y"] - car["y"])
            along = 90 if diff > 0 else -90
        else:
            diff = axis_cm(t["x"] - car["x"])
            along = 0 if diff > 0 else 180
        bearing = math.degrees(math.atan2(t["y"] - car["y"], t["x"] - car["x"]))
        dist = dist_cm(t["x"], t["y"], car["x"], car["y"])

        if self.state == "ORIENT":              # 線に沿って目標側を向く
            if abs(diff) <= ALIGN_TOL_CM:
                return self._go("TURN", "すでに位置が一致")
            err = wrap(along - heading)
            if abs(err) < 25:
                self.dir_sign = 1 if diff > 0 else -1
                return self._go("LINE", f"ライントレース開始（{ALIGN_AXIS}の差 {diff:.0f}cm）")
            self.info = f"線の向きへ回転中 ずれ{err:.0f}°"
            return self._spin(err)

        if self.state == "LINE":                # ライントレースで目標の横まで
            if abs(diff) <= ALIGN_TOL_CM or diff * self.dir_sign < 0:   # 一致 or 行き過ぎ
                return self._go("TURN", f"{ALIGN_AXIS} が一致 → 目標の方を向く")
            self.info = f"ライントレース中 残り{abs(diff):.0f}cm"
            return self._line(sensors)

        if self.state == "TURN":                # その場で目標の方を向く
            err = wrap(bearing - heading)
            if abs(err) < 10:
                return self._go("DRIVE", f"直進開始 距離{dist:.0f}cm")
            self.info = f"目標の方へ回転中 ずれ{err:.0f}°"
            return self._spin(err)

        if self.state == "DRIVE":               # 線を使わずに直進
            if dist <= STOP_DIST_CM:
                return self._go("ARRIVED", f"{self.target} に到着")
            err = wrap(bearing - heading)
            if abs(err) > 40:
                return self._go("TURN", "向きが大きくずれたので回転し直す")
            c = KH_DRIVE * err * TURN_SIGN
            self.info = f"直進中 残り{dist:.0f}cm ずれ{err:.0f}°"
            return SPEED_DRIVE + c, SPEED_DRIVE - c

        if self.state == "ARRIVED":             # 到着。物体が元に戻されたら待機へ
            if self.displacement(t) < PULL_CM / 2:
                self._go("IDLE", f"{self.target} が戻されたので待機へ")
                self.target = None
            self.info = "到着して停止中"
            return 0, 0
        return 0, 0

    def _go(self, state, msg):
        print(f"[{self.state} → {state}] {msg}")
        self.state = state
        return 0, 0

    def _spin(self, err):
        """err>0（時計回りに回りたい）なら右回転"""
        s = SPEED_TURN if err * TURN_SIGN > 0 else -SPEED_TURN
        return s, -s

    def _line(self, sensors):
        on = [w for w, s in zip((-3, -1, 1, 3), sensors) if s]
        if on:
            err = sum(on) / len(on)
            self.last_err = err
        else:                                   # 線を見失った → 最後に見えた側へ曲がる
            err = 3 if self.last_err > 0 else -3 if self.last_err < 0 else 0
        return SPEED_LINE + KP_LINE * err, SPEED_LINE - KP_LINE * err

# ---------------------------------------------------------------------
def run_auto():
    car_receiver.start()
    ctl = AutoController()
    last_print = 0
    while True:
        car, objs = car_receiver.get_location()
        l, r = ctl.step(car, objs, read_sensors(), time.time())
        drive_lr(l, r)
        if time.time() - last_print > 0.5:
            print(f"  {ctl.info}  モータ L{l:.0f} R{r:.0f}")
            last_print = time.time()
        time.sleep(LOOP_DT)

def run_check():
    """動かさずに状態を表示"""
    car_receiver.start()
    ctl = AutoController()
    while True:
        car, objs = car_receiver.get_location()
        ctl.step(car, objs, read_sensors(), time.time())
        print("センサ(左→右):", ["■" if s else "□" for s in read_sensors()], " 車:", car)
        for o in objs.values():
            print(f"   {o['id']:>9} {o['color']:>6} ({o['x']},{o['y']})  移動 {ctl.displacement(o):5.1f}cm")
        print(" ", ctl.info)
        time.sleep(0.5)

def run_turntest():
    car_receiver.start()
    time.sleep(1)
    car, _ = car_receiver.get_location()
    if car is None:
        print("車(QR)が見えません"); return
    a0 = car["angle"]
    drive_lr(SPEED_TURN, -SPEED_TURN)         # 右回転のつもり
    time.sleep(0.6)
    drive_lr(0, 0)
    time.sleep(1)
    car, _ = car_receiver.get_location()
    if car is None:
        print("回転後に車(QR)が見えません"); return
    d = wrap(car["angle"] - a0)
    print(f"angle: {a0} → {car['angle']}（{d:+.0f}°）")
    print("TURN_SIGN = 1 のままでOK" if d > 0 else "TURN_SIGN = -1 にしてください")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
    try:
        {"auto": run_auto, "check": run_check, "turntest": run_turntest}[mode]()
    except KeyboardInterrupt:
        print("\n中断されました。")
    finally:
        base.stop()          # どの経路で抜けても必ずモーターを止める
