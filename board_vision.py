#!/usr/bin/env python3
"""
board_vision.py - 天井Webカメラで盤面を認識（ノートPCで実行）

  - Webカメラは ノートPC に USB 接続
  - 盤の四隅をブラウザ上でクリック → 真上視点に射影変換
  - 色の濃い領域（彩度が高い所）を物体として拾う
    （黒い盤・白線・ロープは彩度がほぼ0なので自動で除外される）
  - 物体を1個ずつクリックして「色」と「名前」を登録 → 以後その名前で追跡
  - 車に貼った ArUcoマーカー（または QRコード）で車の位置と向きを検出
  - 結果を UDP でラズパイ（car_receiver.py）へ送信

使い方:  pip install opencv-python numpy
         python board_vision.py
         ブラウザで http://localhost:8000 を開く
"""
import cv2
import numpy as np
import json, threading, time, os, socket, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# ===== 設定 =====
PI_IP = "192.168.96.201"      # ラズパイのIPアドレス
PI_PORT = 5005                # car_receiver.py と合わせる
SEND_HZ = 10                  # ラズパイへ送る頻度[回/秒]
CAM_ID = 1                    # USBカメラ
CAP_W, CAP_H = 1920, 1080     # カメラの解像度（高いほどQRが読みやすい。非対応なら自動で下がる）
RAW_VIEW_W = 960              # ブラウザに表示するカメラ映像の横幅[px]（表示用に縮小するだけ）
BOARD_W, BOARD_H = 600, 400   # 補正後の盤面画像サイズ[px]。盤の実寸の縦横比に合わせる
LINE_Y = 200                  # 補正後画像でこの y を超えた物体を「ライン越え」とする
MIN_AREA = 150                # これより小さい色領域はノイズとして無視
CAR_HOLD = 0.3                # QRを一瞬見失っても、この秒数は直前の位置を保持
CAR_MARKER = "auto"           # 車の目印: "aruco"（推奨・小さくても読める） / "qr" / "auto"（両方探す）
CAR_ANGLE_OFFSET = 0          # マーカー(QR)の上辺が車のどちらを向いているか: 前=0, 左=90, 右=-90, 後ろ=180
                              # （QRの中身は何でもOK。読んだ文字列は "text" として送るだけ）
TRACK_DIST = 60               # 未登録の物体：前フレームからこの距離[px]以内なら同じ番号を引き継ぐ
TRACK_TIMEOUT = 3.0           # 未登録の物体：見えなくなってもこの秒数は番号を保持
PORT = 8000
CONF = "board_config.json"    # 四隅・登録物体の保存先

# 色の初期お手本(RGB)。盤の写真から測った値。物体を登録すると、その物体の実測色もお手本に加わる
DEFAULT_COLORS = {
    "red":    [200, 32, 35],
    "orange": [213, 148, 11],
    "yellow": [220, 194, 11],
    "green":  [10, 130, 101],
    "blue":   [15, 78, 190],
}
BGR_DRAW = {"red": (0, 0, 255), "orange": (0, 140, 255), "yellow": (0, 220, 255),
            "green": (0, 160, 0), "blue": (255, 80, 0)}

cfg = {"corners": [], "chroma_min": 60, "colors": DEFAULT_COLORS, "objects": []}
if os.path.exists(CONF):
    cfg.update(json.load(open(CONF)))
    for k in ("rgb_lo", "rgb_hi"):            # 旧形式の設定は無視
        cfg.pop(k, None)

def save():
    json.dump(cfg, open(CONF, "w"), ensure_ascii=False, indent=1)

qr = cv2.QRCodeDetector()
QR_DETECTORS = ([cv2.QRCodeDetectorAruco()] if hasattr(cv2, "QRCodeDetectorAruco") else []) + [qr]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# ArUco（OpenCV 4.7以降/以前の両方に対応）。印刷用マーカーは aruco_marker.pdf（DICT_4X4_50 の ID 0）
_adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
if hasattr(cv2.aruco, "ArucoDetector"):
    _adet = cv2.aruco.ArucoDetector(_adict, cv2.aruco.DetectorParameters())
    detect_aruco = lambda g: _adet.detectMarkers(g)
else:
    _ap = cv2.aruco.DetectorParameters_create()
    detect_aruco = lambda g: cv2.aruco.detectMarkers(g, _adict, parameters=_ap)
latest_frame = None           # 画像保存ボタン用
qr_state = {"c": None, "size": 0, "text": "", "text_t": 0, "found": False}   # 前回見つけたQR（カメラ画像の座標）
raw_scale = 1.0               # カメラ画像 ÷ ブラウザ表示 の倍率（クリック座標の換算用）
lock = threading.Lock()
tlock = threading.Lock()       # 追跡データ（tracker・登録）用
jpgs = {"raw": None, "board": None, "mask": None}
result = {}
latest_dets = []              # クリック登録用（直近フレームの物体と輪郭）
selected_id = None            # ブラウザで選択中の物体の名前（映像に白丸で表示）

def order(pts):
    """4点を 左上, 右上, 右下, 左下 の順に並べる（クリック順は自由）"""
    pts = np.array(pts, np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()],
                     pts[s.argmax()], pts[d.argmax()]], np.float32)

def classify(rgb):
    """平均RGBを一番近いお手本の色に分類。明るさの影響を減らすため r:g:b の比率で比べる。
    お手本 = 初期の5色 + 登録した各物体の実測色"""
    refs = list(cfg["colors"].items()) + [(o["color"], o["rgb"]) for o in cfg["objects"]]
    v = np.array(rgb, float); v = v / (v.sum() + 1e-6)
    best, bd = None, 1e9
    for name, ref in refs:
        r = np.array(ref, float); r = r / (r.sum() + 1e-6)
        d = np.linalg.norm(v - r)
        if d < bd:
            best, bd = name, d
    return best

class Tracker:
    """・登録した物体（名前つき）：同じ色の中で一番近いものに対応づけて追跡
       ・未登録の物体：左から object1, object2 ... と自動で通し番号（登録名と重ならない番号）"""
    def __init__(self):
        self.pos = {}          # 登録物体の最後に見えた位置  name -> (x, y)
        self.auto = {}         # 未登録物体  "object1" -> {"color", "no", "x", "y", "t"}
        self.renumber_req = False

    @staticmethod
    def _greedy(pairs):
        """(距離, 検出index, キー) を近い順に1対1で確定"""
        used_d, used_k, res = set(), set(), []
        for _, i, k in sorted(pairs):
            if i in used_d or k in used_k:
                continue
            used_d.add(i); used_k.add(k); res.append((i, k))
        return res

    def update(self, dets):
        now = time.time()
        # 1) 登録済みの物体
        pairs = []
        for i, d in enumerate(dets):
            for o in cfg["objects"]:
                if o["color"] == d["color"]:
                    px, py = self.pos.get(o["name"], (o["x"], o["y"]))
                    pairs.append((np.hypot(d["x"] - px, d["y"] - py), i, o["name"]))
        used_d, seen = set(), set()
        for i, name in self._greedy(pairs):
            used_d.add(i); seen.add(name)
            dets[i]["id"], dets[i]["registered"] = name, True
            self.pos[name] = (dets[i]["x"], dets[i]["y"])

        # 2) 未登録の物体：前フレームの近い同色物体の番号を引き継ぐ
        reg_names = {o["name"] for o in cfg["objects"]}
        if self.renumber_req:          # 番号の振り直し → 今の位置で左から付け直す
            self.auto, self.renumber_req = {}, False
        self.auto = {k: v for k, v in self.auto.items()
                     if now - v["t"] < TRACK_TIMEOUT and k not in reg_names}
        rest = [i for i in range(len(dets)) if i not in used_d]
        pairs = []
        for i in rest:
            for k, t in self.auto.items():
                if t["color"] == dets[i]["color"]:
                    dist = np.hypot(dets[i]["x"] - t["x"], dets[i]["y"] - t["y"])
                    if dist < TRACK_DIST:
                        pairs.append((dist, i, k))
        for i, k in self._greedy(pairs):
            used_d.add(i)
            self._auto_set(dets[i], k, self.auto[k]["no"], now)
        # 新しく現れた物体：左→右の順に、空いている一番小さい番号（object1, object2 ...）
        for i in sorted([i for i in rest if i not in used_d], key=lambda i: (dets[i]["x"], dets[i]["y"])):
            taken = reg_names | set(self.auto)
            no = 1
            while f"object{no}" in taken:
                no += 1
            self._auto_set(dets[i], f"object{no}", no, now)

        names = [o["name"] for o in cfg["objects"]]
        missing = [n for n in names if n not in seen]
        def order_key(d):              # object1, object2 ... の番号順。独自の名前は後ろに登録順
            n = d["id"][6:] if d["id"].startswith("object") else ""
            if n.isdigit():
                return (0, int(n))
            return (1, names.index(d["id"]) if d["registered"] else 0)
        dets.sort(key=order_key)
        return dets, missing

    def _auto_set(self, d, key, no, now):
        d["id"], d["registered"] = key, False
        self.auto[key] = {"color": d["color"], "no": no, "x": d["x"], "y": d["y"], "t": now}

    def forget(self, name=None):
        if name is None:
            self.pos.clear()
        else:
            self.pos.pop(name, None)

tracker = Tracker()

def _find_aruco(gray):
    """ArUcoマーカーの4隅を探す（等倍→2倍に拡大）。戻り値: (4隅, ID) または None"""
    for s in (1, 2):
        g = gray if s == 1 else cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        corners, ids, _ = detect_aruco(g)
        if ids is not None and len(ids):
            return np.asarray(corners[0]).reshape(4, 2) / s, int(np.asarray(ids).ravel()[0])
    return None

def _find_marker(gray):
    """戻り値: (4隅, 種類, QRを見つけた画像, 倍率) または None"""
    if CAR_MARKER in ("aruco", "auto"):
        hit = _find_aruco(gray)
        if hit:
            return hit[0], f"aruco{hit[1]}", None, 1
    if CAR_MARKER in ("qr", "auto"):
        hit = _find_qr(gray)
        if hit:
            return hit[0], "qr", hit[1], hit[2]
    return None

def _find_qr(gray):
    """白黒画像からQRの4隅を探す。小さいQRのため、拡大・コントラスト強調・2種類の検出器を順に試す。
    戻り値: (4隅[元画像の座標], 見つけた画像, 倍率) または None"""
    h, w = gray.shape[:2]
    for s in (1, 2, 3):
        if s > 1 and max(h, w) * s > 2600:      # 拡大しすぎると遅いので上限
            break
        g = gray if s == 1 else cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        for im in (g, clahe.apply(g)):
            for det in QR_DETECTORS:
                try:
                    ok, p = det.detect(im)
                except cv2.error:
                    continue
                if ok and p is not None and p.size == 8:
                    return p.reshape(4, 2) / s, im, s
    return None

def detect_car(frame, M, corners):
    """車のマーカー（ArUco/QR）を元のカメラ画像（高解像度）で探し、盤面座標に変換する"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    found = None
    # 1) 前回見つけた場所の周辺だけを拡大して探す（速くて小さいQRにも強い）
    if qr_state["c"] is not None:
        cx, cy = qr_state["c"]
        r = int(max(qr_state["size"] * 2.5, 80))
        x0, y0 = max(0, int(cx) - r), max(0, int(cy) - r)
        x1, y1 = min(W, int(cx) + r), min(H, int(cy) + r)
        hit = _find_marker(gray[y0:y1, x0:x1])
        if hit:
            found = (hit[0] + [x0, y0], hit[1], hit[2], hit[3], (x0, y0))
    # 2) 見つからなければ盤の範囲全体を探す
    if found is None:
        c = np.array(corners)
        x0, y0 = np.maximum(c.min(0).astype(int) - 20, 0)
        x1, y1 = np.minimum(c.max(0).astype(int) + 20, [W, H])
        hit = _find_marker(gray[y0:y1, x0:x1])
        if hit:
            found = (hit[0] + [x0, y0], hit[1], hit[2], hit[3], (x0, y0))
    if found is None:
        qr_state["c"], qr_state["found"] = None, False
        return None
    pts, kind, im, sc, (ox, oy) = found
    qr_state["c"] = pts.mean(0)
    qr_state["size"] = float(np.linalg.norm(pts[0] - pts[2]) / 1.414)
    qr_state["found"] = True
    # 中身の文字列は1秒に1回だけ読む（位置と向きには不要で、読むのは重いため）
    if kind != "qr":
        qr_state["text"] = kind                      # ArUco は "aruco0" のようにIDを入れる
    elif time.time() - qr_state["text_t"] > 1.0:
        qr_state["text_t"] = time.time()
        try:
            t, _ = qr.decode(im, ((pts - [ox, oy]) * sc).astype(np.float32).reshape(1, 4, 2))
            if t:
                qr_state["text"] = t
        except cv2.error:
            pass
    # 4隅は マーカー自身の向きで 左上,右上,右下,左下 の順に返る（ArUco・QRとも）
    p = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), M).reshape(-1, 2)
    center = p.mean(0)
    top = (p[0] + p[1]) / 2             # QRの上辺の中点
    ang = float(np.degrees(np.arctan2(top[1] - center[1], top[0] - center[0])))
    ang = (ang + CAR_ANGLE_OFFSET + 180) % 360 - 180    # 貼った向きの補正（-180〜180に）
    L = np.linalg.norm(top - center) * 1.5
    front = center + L * np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
    return {"text": qr_state["text"], "x": int(center[0]), "y": int(center[1]),
            "angle": round(ang, 1), "pts": p, "front": front, "raw_pts": pts,
            "qr_px": round(qr_state["size"]), "kind": kind}

def loop():
    global latest_dets, latest_frame
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)   # Windowsは起動が速いDirectShow
    else:
        cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # 高解像度でもコマ落ちしにくい形式
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    if not cap.isOpened():
        print("カメラが開けません。CAM_ID を 0/1/2 と変えて試してください")
        os._exit(1)
    global raw_scale
    ok, frame = cap.read()
    while not ok:
        time.sleep(0.1); ok, frame = cap.read()
    fh, fw = frame.shape[:2]
    print(f"カメラ解像度: {fw}x{fh}")
    raw_scale = fw / RAW_VIEW_W
    ow, oh = cfg.get("frame_size", [640, 480])      # 四隅を指定したときの解像度（旧設定は640x480）
    if cfg["corners"] and (ow, oh) != (fw, fh):
        if abs(ow / oh - fw / fh) < 0.03:            # 縦横比が同じ → 四隅を換算
            cfg["corners"] = [[x * fw / ow, y * fh / oh] for x, y in cfg["corners"]]
            print(f"四隅の座標を {ow}x{oh} → {fw}x{fh} に換算しました")
        else:                                        # 縦横比が違う → 写る範囲が変わるので指定し直し
            cfg["corners"] = []
            print("解像度の縦横比が変わったので、ブラウザで四隅を指定し直してください")
    cfg["frame_size"] = [fw, fh]
    save()
    dst = np.float32([[0, 0], [BOARD_W, 0], [BOARD_W, BOARD_H], [0, BOARD_H]])
    font = cv2.FONT_HERSHEY_SIMPLEX
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    last_print = last_send = 0
    last_car, last_car_t = None, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        corners4 = list(cfg["corners"])
        chroma_min = cfg["chroma_min"]

        raw = cv2.resize(frame, (RAW_VIEW_W, int(frame.shape[0] / raw_scale)))   # 表示用に縮小
        for i, (x, y) in enumerate(corners4):
            x, y = int(x / raw_scale), int(y / raw_scale)
            cv2.circle(raw, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(raw, str(i + 1), (x + 6, y - 6), font, 0.6, (0, 0, 255), 2)

        out = {"objects": [], "missing": [], "car": None}
        dets = []
        if len(corners4) == 4:
            M = cv2.getPerspectiveTransform(order(corners4), dst)
            board = cv2.warpPerspective(frame, M, (BOARD_W, BOARD_H))
            blur = cv2.GaussianBlur(board, (5, 5), 0)
            vis = board.copy()
            cv2.line(vis, (0, LINE_Y), (BOARD_W, LINE_Y), (0, 255, 255), 2)

            # --- 色の濃い領域 = 物体 ---
            b16 = blur.astype(np.int16)
            chroma = b16.max(2) - b16.min(2)          # RGBの最大-最小（白・黒・灰色は小さい）
            mask = ((chroma > chroma_min) * 255).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) < MIN_AREA:
                    continue
                m = cv2.moments(c)
                cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
                inner = np.zeros(mask.shape, np.uint8)
                cv2.drawContours(inner, [c], -1, 255, -1)
                bb, gg, rr = cv2.mean(blur, inner & mask)[:3]
                rgb = [int(rr), int(gg), int(bb)]
                dets.append({"color": classify(rgb), "x": cx, "y": cy,
                             "crossed": cy > LINE_Y, "rgb": rgb, "_cnt": c})

            # --- 登録した名前を付ける ---
            with tlock:
                dets, missing = tracker.update(dets)
            out["missing"] = missing
            for d in dets:
                c = d["_cnt"]
                cx, cy = d["x"], d["y"]
                col = BGR_DRAW.get(d["color"], (255, 255, 255))
                cv2.drawContours(vis, [c], -1, col, 3 if d["registered"] else 1)   # 登録済みは太枠
                cv2.drawContours(vis, [c], -1, (255, 255, 255) if d["crossed"] else (0, 0, 0), 1)
                cv2.circle(vis, (cx, cy), 4, (255, 255, 255), -1)
                label = d["id"] if d["registered"] else d["id"] + "*"   # * = 未登録（自動の名前）
                cv2.putText(vis, f"{label}{' !' if d['crossed'] else ''}", (cx + 6, cy - 4),
                            font, 0.55, (255, 255, 255), 2)
                cv2.putText(vis, f"({cx},{cy})", (cx + 6, cy + 12), font, 0.4, (255, 255, 255), 1)
                out["objects"].append({k: d[k] for k in
                                       ("id", "color", "x", "y", "crossed", "registered", "rgb")})
            for d in dets:
                if d["id"] == selected_id:
                    cv2.circle(vis, (d["x"], d["y"]), 28, (255, 255, 255), 3)
                    cv2.circle(vis, (d["x"], d["y"]), 31, (0, 0, 0), 1)
            if missing:
                cv2.putText(vis, "missing: " + ", ".join(missing), (8, BOARD_H - 10),
                            font, 0.5, (0, 0, 255), 2)

            # --- QRで車検出 ---
            try:
                car = detect_car(frame, M, corners4)
            except Exception as e:                 # 検出で予期しないエラーが出ても認識ループは止めない
                print("車の検出でエラー:", e)
                car, qr_state["found"] = None, False
            now = time.time()
            if car:
                last_car, last_car_t = car, now
            elif last_car and now - last_car_t < CAR_HOLD:
                car = last_car
            if qr_state["found"]:
                cv2.polylines(raw, [(car["raw_pts"] / raw_scale).astype(np.int32)], True, (255, 0, 255), 2)
                cv2.putText(vis, f"CAR: {car['kind']} OK ({car['qr_px']}px)", (8, 20), font, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(vis, f"CAR: NOT FOUND ({CAR_MARKER})", (8, 20), font, 0.6, (0, 0, 255), 2)
            if car:
                out["car"] = {k: car[k] for k in ("text", "x", "y", "angle")}
                cv2.polylines(vis, [car["pts"].astype(np.int32)], True, (255, 0, 255), 2)
                cv2.arrowedLine(vis, (car["x"], car["y"]),
                                (int(car["front"][0]), int(car["front"][1])), (255, 0, 255), 2)
                cv2.putText(vis, f"car {car['angle']}deg", (car["x"] + 8, car["y"] + 18),
                            font, 0.5, (255, 0, 255), 2)
        else:
            vis = np.zeros((BOARD_H, BOARD_W, 3), np.uint8)
            mask = vis[:, :, 0]
            cv2.putText(vis, "Click 4 board corners on RAW view", (20, BOARD_H // 2),
                        font, 0.7, (255, 255, 255), 2)

        with lock:
            latest_frame = frame
            latest_dets = dets
            jpgs["raw"] = cv2.imencode(".jpg", raw)[1].tobytes()
            jpgs["board"] = cv2.imencode(".jpg", vis)[1].tobytes()
            jpgs["mask"] = cv2.imencode(".jpg", mask)[1].tobytes()
            result.clear()
            result.update(out)
        if time.time() - last_send > 1 / SEND_HZ:
            out["t"] = time.time()
            try:
                sock.sendto(json.dumps(out).encode(), (PI_IP, PI_PORT))
            except OSError as e:
                print("送信エラー:", e)
            last_send = time.time()
        if time.time() - last_print > 1:
            print(out)
            last_print = time.time()

def find_hit(x, y):
    """クリック位置の物体を返す（外していても 30px 以内なら一番近い物体）"""
    with lock:
        dets = list(latest_dets)
    for d in dets:
        if cv2.pointPolygonTest(d["_cnt"], (float(x), float(y)), False) >= 0:
            return d
    if dets:
        d = min(dets, key=lambda d: np.hypot(d["x"] - x, d["y"] - y))
        if np.hypot(d["x"] - x, d["y"] - y) < 30:
            return d
    return None

def register(obj_id, color, name):
    """選択中の物体（今の名前 obj_id）を、指定の色・名前で登録"""
    global selected_id
    if color not in DEFAULT_COLORS:
        return "色を選んでください"
    with lock:
        hit = next((d for d in latest_dets if d["id"] == obj_id), None)
    if hit is None:
        return f"{obj_id} が今見えていません。もう一度物体を選び直してください"
    with tlock:
        _register(hit, color, name)
    selected_id = name
    return "ok"

def _register(hit, color, name):
    if hit["registered"]:
        old = hit["id"]                    # 登録済みの物体の名前・色を変更
    else:
        old = None
        tracker.auto.pop(hit["id"], None)  # 自動の名前は使わなくなる
    objs = [o for o in cfg["objects"] if o["name"] not in (name, old)]
    for o in cfg["objects"]:
        if o not in objs:
            tracker.forget(o["name"])
    objs.append({"name": name, "color": color, "rgb": hit["rgb"], "x": hit["x"], "y": hit["y"]})
    cfg["objects"] = objs
    tracker.pos[name] = (hit["x"], hit["y"])
    save()
    print(f"登録: {name} = {color}  RGB={hit['rgb']}  位置=({hit['x']},{hit['y']})")

PAGE = """<!doctype html><meta charset=utf-8><title>Board Vision</title>
<style>body{font-family:sans-serif;margin:12px} img{max-width:100%;border:1px solid #888;margin:2px}
table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:2px 8px}
.sw{display:inline-block;width:14px;height:14px;vertical-align:middle;border:1px solid #888}
.panel{background:#f3f3f3;padding:10px;margin:6px 0;border-radius:6px}
.cb{color:#fff;border:3px solid transparent;border-radius:6px;padding:6px 12px;font-size:15px;cursor:pointer;margin:2px}
.cb.on{border-color:#000;box-shadow:0 0 0 2px #fff inset}
.db{margin:2px;padding:3px 8px;cursor:pointer;border:2px solid #999;border-radius:4px;background:#fff}
.db.on{border-color:#000;font-weight:bold}</style>
<h3>Board Vision</h3>
<p>① RAW映像で盤の四隅を4回クリック（順番自由。5回目でやり直し）
<button onclick="fetch('/reset')">四隅リセット</button>
<button onclick="fetch('/snapshot').then(r=>r.text()).then(t=>alert(t))">カメラ画像を保存（原因調査用）</button></p>
<img id=raw src=/stream/raw style="cursor:crosshair">
<p>② 物体の名前付け：<b>物体を選ぶ → 色ボタンを押す → 名前を確認 → 「登録」</b>（全部の物体について繰り返す）<br>
<button onclick="fetch('/renumber')">番号を振り直す（今の位置で左から object1, 2, 3…）</button><br>
<small>何もしなくても左から object1, object2 … と自動で名前が付きます（映像では末尾に * ）。登録すると太枠になり名前が固定されます。選択中の物体は映像に白い丸が付きます。</small></p>
<img id=board src=/stream/board style="cursor:crosshair"> <img src=/stream/mask width=300>
<div class=panel>
<div>物体を選ぶ（映像の物体をクリック、または下のボタン）：</div>
<div id=dets></div>
<div style="margin-top:8px">選択中: <b id=sel>なし</b></div>
<div style="margin-top:8px">色:
<button class=cb data-c=red style="background:#d22">red</button>
<button class=cb data-c=orange style="background:#e80">orange</button>
<button class=cb data-c=yellow style="background:#db0;color:#000">yellow</button>
<button class=cb data-c=green style="background:#080">green</button>
<button class=cb data-c=blue style="background:#04c">blue</button></div>
<div style="margin-top:8px">名前: <input id=objname size=14 placeholder="半角英数字">
<button id=regbtn style="font-size:16px;padding:4px 16px">登録</button>
<span id=msg style="color:#c00"></span></div>
<small>名前は半角英数字（映像の文字表示が日本語に対応していないため）。Enterキーでも登録できます。</small>
</div>
<p>登録済みの物体 <button onclick="if(confirm('全部消しますか？'))fetch('/clear').then(load)">全部消す</button></p>
<table id=tbl></table>
<p>彩度しきい値 <input type=range id=cm min=10 max=200 oninput="setcm()"> <span id=cmv></span>
（白線や影まで拾うなら上げる／物体が欠けるなら下げる）</p>
<pre id=out></pre>
<script>
const $ = id => document.getElementById(id);
let selId = null;       // 選択中の物体の今の名前
let selColor = null;    // 選んだ色
const COLORS = {red:'#d22', orange:'#e80', yellow:'#db0', green:'#080', blue:'#04c'};
function pos(img, e) {
  const b = img.getBoundingClientRect();
  return [(e.clientX - b.left) * img.naturalWidth / b.width,
          (e.clientY - b.top) * img.naturalHeight / b.height];
}
function setColor(c) {
  selColor = c;
  document.querySelectorAll('.cb').forEach(b => b.classList.toggle('on', b.dataset.c == c));
}
let lastObjs = [], regNames = [];
function showSel(d) {
  if (d.error) { $('msg').textContent = d.error; return; }
  selId = d.id;
  $('sel').textContent = `${d.id}${d.registered ? '（登録済み）' : '（未登録）'}  位置(${d.x},${d.y})`;
  $('objname').value = d.id;
  setColor(d.color);
  $('msg').textContent = '';
  drawDets();
}
document.querySelectorAll('.cb').forEach(b => b.onclick = () => setColor(b.dataset.c));
$('raw').onclick = e => { const [x, y] = pos($('raw'), e); fetch(`/click?x=${x}&y=${y}`); };
$('board').onclick = e => {
  const [x, y] = pos($('board'), e);
  fetch(`/select?x=${x}&y=${y}`).then(r => r.json()).then(showSel);
};
function pick(id) { fetch(`/select?id=${encodeURIComponent(id)}`).then(r => r.json()).then(showSel); }
$('regbtn').onclick = doRegister;
$('objname').onkeydown = e => { if (e.key == 'Enter') doRegister(); };
function doRegister() {
  if (!selId) { $('msg').textContent = '先に物体を選んでください'; return; }
  if (!selColor) { $('msg').textContent = '色ボタンを押してください'; return; }
  const n = $('objname').value.trim();
  if (!/^[A-Za-z0-9_-]+$/.test(n)) { $('msg').textContent = '名前は半角英数字で入力してください'; return; }
  fetch(`/register?id=${encodeURIComponent(selId)}&color=${selColor}&name=${encodeURIComponent(n)}`)
    .then(r => r.text()).then(t => {
      if (t == 'ok') {
        $('msg').textContent = '';
        $('sel').textContent = `${n}（${selColor}）を登録しました。次の物体を選んでください`;
        selId = n;
      } else $('msg').textContent = t;
      load();
    });
}
function drawDets() {     // 今見えている物体のボタン一覧
  $('dets').innerHTML = lastObjs.map(o =>
    `<button class="db${o.id == selId ? ' on' : ''}" onclick="pick('${o.id}')">
      <span class=sw style="background:${COLORS[o.color]}"></span> ${o.id}${o.registered ? '' : '*'}</button>`).join('')
    || '（物体が見えていません。①で四隅を指定してください）';
}
function load() {
  fetch('/cfg').then(r => r.json()).then(c => {
    regNames = c.objects.map(o => o.name);
    $('tbl').innerHTML = '<tr><th>名前</th><th>色</th><th>実測RGB</th><th>登録位置</th><th></th></tr>' +
      c.objects.map(o => `<tr><td>${o.name}</td><td>${o.color}</td>
        <td><span class=sw style="background:rgb(${o.rgb})"></span> ${o.rgb}</td>
        <td>(${o.x},${o.y})</td>
        <td><button onclick="fetch('/delete?name=${o.name}').then(load)">削除</button></td></tr>`).join('');
    $('cm').value = c.chroma_min; $('cmv').textContent = c.chroma_min;
  });
}
function setcm() { $('cmv').textContent = $('cm').value; fetch('/chroma?v=' + $('cm').value); }
load();
setInterval(() => fetch('/result').then(r => r.json()).then(j => {
  $('out').textContent = JSON.stringify(j, null, 1);
  const key = j.objects.map(o => o.id + o.color + o.registered).join();
  if (key != drawDets.key) { lastObjs = j.objects; drawDets.key = key; drawDets(); }
}), 500);
</script>"""

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, body, ctype="text/plain; charset=utf-8"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            return self.reply(PAGE.encode(), "text/html; charset=utf-8")
        if u.path.startswith("/stream/"):
            key = u.path.split("/")[-1]
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while True:
                    with lock:
                        j = jpgs.get(key)
                    if j:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
                                         % len(j) + j + b"\r\n")
                    time.sleep(0.07)
            except (BrokenPipeError, ConnectionResetError):
                return
        if u.path == "/click":
            if len(cfg["corners"]) >= 4:
                cfg["corners"] = []
            cfg["corners"] = cfg["corners"] + [[float(q["x"][0]) * raw_scale, float(q["y"][0]) * raw_scale]]
            save()
        elif u.path == "/snapshot":
            with lock:
                f = None if latest_frame is None else latest_frame.copy()
            if f is None:
                return self.reply("まだ画像がありません".encode())
            fn = time.strftime("snapshot_%H%M%S.png")
            cv2.imwrite(fn, f)
            print("保存:", os.path.abspath(fn))
            return self.reply(f"{os.path.abspath(fn)} に保存しました".encode())
        elif u.path == "/reset":
            cfg["corners"] = []
            save()
        elif u.path == "/select":
            global selected_id
            if "id" in q:                                   # 一覧のボタンから選択
                with lock:
                    d = next((d for d in latest_dets if d["id"] == q["id"][0]), None)
            else:                                           # 映像のクリックで選択
                d = find_hit(float(q["x"][0]), float(q["y"][0]))
            if d is None:
                body = {"error": "物体が見つかりません。物体の上をクリックしてください"}
            else:
                selected_id = d["id"]
                body = {k: d[k] for k in ("id", "color", "x", "y", "registered")}
            return self.reply(json.dumps(body).encode(), "application/json")
        elif u.path == "/register":
            msg = register(q["id"][0], q["color"][0], q["name"][0])
            return self.reply(msg.encode())
        elif u.path == "/renumber":
            tracker.renumber_req = True
        elif u.path == "/delete":
            name = q["name"][0]
            with tlock:
                cfg["objects"] = [o for o in cfg["objects"] if o["name"] != name]
                tracker.forget(name)
            save()
        elif u.path == "/clear":
            with tlock:
                cfg["objects"] = []
                tracker.forget()
            save()
        elif u.path == "/chroma":
            cfg["chroma_min"] = int(q["v"][0])
            save()
        elif u.path == "/cfg":
            return self.reply(json.dumps(cfg).encode(), "application/json")
        elif u.path == "/result":
            with lock:
                body = json.dumps(result).encode()
            return self.reply(body, "application/json")
        self.reply(b"ok")

if __name__ == "__main__":
    threading.Thread(target=loop, daemon=True).start()
    print(f"ブラウザで http://localhost:{PORT} を開いてください（送信先 {PI_IP}:{PI_PORT}）")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
