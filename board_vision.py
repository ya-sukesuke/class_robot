#!/usr/bin/env python3
"""
board_vision.py - 天井Webカメラで盤面を認識（ノートPCで実行）

  - Webカメラは ノートPC に USB 接続
  - 盤の四隅をブラウザ上でクリック → 真上視点に射影変換
  - RGB の範囲で物体を検出し、座標とライン越えを判定
  - 車に貼った QRコード で車の位置と向きを検出
  - 結果を UDP でラズパイ（car_receiver.py）へ送信

使い方:  pip install opencv-python numpy
         下の PI_IP をラズパイのIPに書き換えて  python board_vision.py
         ブラウザで http://localhost:8000 を開く
"""
import cv2
import numpy as np
import json, threading, time, os, socket, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# ===== 設定 =====
PI_IP = "192.168.96.201"        # ← ラズパイのIPアドレスに変更（ssh で使っているもの）
PI_PORT = 5005                # car_receiver.py と合わせる
SEND_HZ = 10                  # ラズパイへ送る頻度[回/秒]
CAM_ID = 1                    # ノートPC内蔵カメラが0なら、USBカメラは1のことが多い
CAP_W, CAP_H = 640, 480       # QRが読めないときは 1280, 720 に上げると改善することがある
BOARD_W, BOARD_H = 600, 400   # 補正後の盤面画像サイズ[px]。盤の実寸の縦横比に合わせる
LINE_Y = 200                  # 補正後画像でこの y を超えた物体を「ライン越え」とする
MIN_AREA = 150                # これより小さい色領域はノイズとして無視
PICK_MARGIN = 40              # 盤面クリックで色を拾うときの ±範囲
CAR_HOLD = 0.3                # QRを一瞬見失っても、この秒数は直前の位置を保持
PORT = 8000
CONF = "board_config.json"    # 四隅とRGB閾値の保存先

cfg = {"corners": [], "rgb_lo": [150, 0, 0], "rgb_hi": [255, 90, 90]}   # 初期値は赤
if os.path.exists(CONF):
    cfg.update(json.load(open(CONF)))

def save():
    json.dump(cfg, open(CONF, "w"))

qr = cv2.QRCodeDetector()
lock = threading.Lock()
jpgs = {"raw": None, "board": None, "mask": None}
result = {}
latest_board = None           # 色ピック用

def order(pts):
    """4点を 左上, 右上, 右下, 左下 の順に並べる（クリック順は自由）"""
    pts = np.array(pts, np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()],
                     pts[s.argmax()], pts[d.argmax()]], np.float32)

def detect_car(frame, M):
    """QRを元のカメラ画像で検出し（解像度が高い方が読みやすい）、盤面座標に変換"""
    text, pts, _ = qr.detectAndDecode(frame)
    if pts is None:
        ok, pts = qr.detect(frame)      # 読めなくても位置だけ取れることがある
        if not ok or pts is None:
            return None
        text = ""
    # QRの4隅は QR自身の向きで 左上,右上,右下,左下 の順に返る
    p = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), M).reshape(-1, 2)
    center = p.mean(0)
    front = (p[0] + p[1]) / 2           # QRの上辺の中点 = 車の前方
    ang = float(np.degrees(np.arctan2(front[1] - center[1], front[0] - center[0])))
    return {"text": text, "x": int(center[0]), "y": int(center[1]),
            "angle": round(ang, 1), "pts": p, "front": front}

def loop():
    global latest_board
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)   # Windowsは起動が速いDirectShow
    else:
        cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    if not cap.isOpened():
        print("カメラが開けません。CAM_ID を 0/1/2 と変えて試してください")
        os._exit(1)
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
        # OpenCVの画像は BGR 順なので、RGB設定を並べ替えて使う
        lo = np.array(cfg["rgb_lo"][::-1], np.uint8)
        hi = np.array(cfg["rgb_hi"][::-1], np.uint8)

        raw = frame.copy()
        for i, (x, y) in enumerate(corners4):
            cv2.circle(raw, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(raw, str(i + 1), (int(x) + 6, int(y) - 6), font, 0.6, (0, 0, 255), 2)

        out = {"objects": [], "car": None}
        if len(corners4) == 4:
            M = cv2.getPerspectiveTransform(order(corners4), dst)
            board = cv2.warpPerspective(frame, M, (BOARD_W, BOARD_H))
            blur = cv2.GaussianBlur(board, (5, 5), 0)
            vis = board.copy()
            cv2.line(vis, (0, LINE_Y), (BOARD_W, LINE_Y), (0, 255, 255), 2)

            # --- RGBで物体検出 ---
            mask = cv2.inRange(blur, lo, hi)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) < MIN_AREA:
                    continue
                m = cv2.moments(c)
                cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
                crossed = cy > LINE_Y
                out["objects"].append({"x": cx, "y": cy, "crossed": crossed})
                cv2.drawContours(vis, [c], -1, (0, 0, 255) if crossed else (0, 255, 0), 2)
                cv2.circle(vis, (cx, cy), 4, (255, 0, 0), -1)
                cv2.putText(vis, f"({cx},{cy})", (cx + 6, cy), font, 0.5, (255, 0, 0), 1)

            # --- QRで車検出 ---
            car = detect_car(frame, M)
            now = time.time()
            if car:
                last_car, last_car_t = car, now
            elif last_car and now - last_car_t < CAR_HOLD:
                car = last_car
            if car:
                out["car"] = {k: car[k] for k in ("text", "x", "y", "angle")}
                cv2.polylines(vis, [car["pts"].astype(np.int32)], True, (255, 0, 255), 2)
                cv2.arrowedLine(vis, (car["x"], car["y"]),
                                (int(car["front"][0]), int(car["front"][1])), (255, 0, 255), 2)
                cv2.putText(vis, f"car {car['angle']}deg", (car["x"] + 8, car["y"] + 18),
                            font, 0.5, (255, 0, 255), 2)
        else:
            board = None
            vis = np.zeros((BOARD_H, BOARD_W, 3), np.uint8)
            mask = vis[:, :, 0]
            cv2.putText(vis, "Click 4 board corners on RAW view", (20, BOARD_H // 2),
                        font, 0.7, (255, 255, 255), 2)

        with lock:
            latest_board = None if board is None else blur
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

PAGE = """<!doctype html><meta charset=utf-8><title>Board Vision</title>
<style>body{font-family:sans-serif;margin:12px} img{max-width:100%;border:1px solid #888;margin:2px}</style>
<h3>Board Vision</h3>
<p>① RAW映像で盤の四隅を4回クリック（順番自由。5回目でやり直し）
<button onclick="fetch('/reset')">四隅リセット</button></p>
<img id=raw src=/stream/raw style="cursor:crosshair">
<p>② 補正後の盤面：<b>検出したい物体をクリックすると、その色が自動で範囲に設定されます</b> / 右は色マスク（白=検出）</p>
<img id=board src=/stream/board style="cursor:crosshair"> <img src=/stream/mask width=300>
<p>③ RGB範囲の微調整（マスクで物体だけが白くなるように。自動保存）</p>
<div id=sl></div>
<pre id=out></pre>
<script>
function pos(img, e) {
  const b = img.getBoundingClientRect();
  return [(e.clientX - b.left) * img.naturalWidth / b.width,
          (e.clientY - b.top) * img.naturalHeight / b.height];
}
raw.onclick = e => { const [x, y] = pos(raw, e); fetch(`/click?x=${x}&y=${y}`); };
board.onclick = e => { const [x, y] = pos(board, e);
  fetch(`/pick?x=${x}&y=${y}`).then(load); };
const names = ["R_lo","G_lo","B_lo","R_hi","G_hi","B_hi"];
function load() {
  fetch('/cfg').then(r => r.json()).then(c => {
    const v = [...c.rgb_lo, ...c.rgb_hi];
    sl.innerHTML = names.map((n, i) =>
      `${n} <input type=range id=${n} min=0 max=255 value=${v[i]} oninput="send()"> <span id=${n}v>${v[i]}</span><br>`).join('');
  });
}
function send() {
  const v = names.map(n => { const e = document.getElementById(n);
    document.getElementById(n + 'v').textContent = e.value; return e.value; });
  fetch('/rgb?v=' + v.join(','));
}
load();
setInterval(() => fetch('/result').then(r => r.json())
  .then(j => out.textContent = JSON.stringify(j, null, 1)), 500);
</script>"""

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, body, ctype="text/plain"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
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
            cfg["corners"] = cfg["corners"] + [[float(q["x"][0]), float(q["y"][0])]]
            save()
        elif u.path == "/reset":
            cfg["corners"] = []
            save()
        elif u.path == "/pick":
            with lock:
                img = latest_board
            if img is not None:
                x, y = int(float(q["x"][0])), int(float(q["y"][0]))
                patch = img[max(0, y - 3):y + 4, max(0, x - 3):x + 4].reshape(-1, 3)
                b, g, r = patch.mean(0)
                rgb = [int(r), int(g), int(b)]
                cfg["rgb_lo"] = [max(0, v - PICK_MARGIN) for v in rgb]
                cfg["rgb_hi"] = [min(255, v + PICK_MARGIN) for v in rgb]
                save()
                print("クリックした色 RGB =", rgb)
        elif u.path == "/rgb":
            v = [int(x) for x in q["v"][0].split(",")]
            cfg["rgb_lo"], cfg["rgb_hi"] = v[:3], v[3:]
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
