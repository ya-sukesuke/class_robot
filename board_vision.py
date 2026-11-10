#!/usr/bin/env python3
"""
board_vision.py - 天井Webカメラで盤面を認識（ノートPCで実行）

  - Webカメラは ノートPC に USB 接続
  - 盤の四隅をブラウザ上でクリック → 真上視点に射影変換
  - 色の濃い領域（彩度が高い所）を物体として拾う
    （黒い盤・白線・ロープは彩度がほぼ0なので自動で除外される）
  - 物体を1個ずつクリックして「色」と「名前」を登録 → 以後その名前で追跡
  - 車に貼った QRコード で車の位置と向きを検出
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
PI_IP = "192.168.96.201"        # ← ラズパイのIPアドレスに変更（ssh で使っているもの）
PI_PORT = 5005                # car_receiver.py と合わせる
SEND_HZ = 10                  # ラズパイへ送る頻度[回/秒]
CAM_ID = 1                    # ノートPC内蔵カメラが0なら、USBカメラは1のことが多い
CAP_W, CAP_H = 640, 480
BOARD_W, BOARD_H = 600, 400   # 補正後の盤面画像サイズ[px]。盤の実寸の縦横比に合わせる
LINE_Y = 200                  # 補正後画像でこの y を超えた物体を「ライン越え」とする
MIN_AREA = 150                # これより小さい色領域はノイズとして無視
CAR_HOLD = 0.3                # QRを一瞬見失っても、この秒数は直前の位置を保持
CAR_ANGLE_OFFSET = 0          # QRの上辺が車のどちらを向いているか: 前=0, 左=90, 右=-90, 後ろ=180
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
cfg["objects"] = []   # 部屋の明るさで色が変わるので、物体は起動のたびにクリックで登録し直す（四隅としきい値は引き継ぐ）

def save():
    json.dump(cfg, open(CONF, "w"), ensure_ascii=False, indent=1)

qr = cv2.QRCodeDetector()
lock = threading.Lock()
jpgs = {"raw": None, "board": None, "mask": None}
result = {}
latest_dets = []              # クリック登録用（直近フレームの物体と輪郭）

def order(pts):
    """4点を 左上, 右上, 右下, 左下 の順に並べる（クリック順は自由）"""
    pts = np.array(pts, np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()],
                     pts[s.argmax()], pts[d.argmax()]], np.float32)

def classify(rgb):
    """平均RGBを一番近いお手本の色に分類。明るさの影響を減らすため r:g:b の比率で比べる。
    お手本 = 登録した各物体の実測色 + まだ1つも登録していない色の初期値
    （登録した色は初期値を使わない。初期値は別の部屋で測った値なので、今の明るさとずれるため）"""
    objects = cfg["objects"]
    reg_colors = {o["color"] for o in objects}
    refs = [(c, v) for c, v in cfg["colors"].items() if c not in reg_colors] + \
           [(o["color"], o["rgb"]) for o in objects]
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
         （最近まで見えていた物体は TRACK_DIST 以内の物体にだけ引き継ぐ。
           盤から外れた直後に、遠くの同色の物体へ名前が飛ぶのを防ぐ）
       ・未登録の物体：red1, red2 ... と自動で通し番号（登録名と重ならない番号）"""
    def __init__(self):
        self.pos = {}          # 登録物体の最後に見えた位置と時刻  name -> (x, y, t)
        self.auto = {}         # 未登録物体  "red1" -> {"color", "x", "y", "t"}

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
        objects = cfg["objects"]   # HTTP側はリストを丸ごと差し替えるので、1フレーム中はこの参照を使う
        # 1) 登録済みの物体
        pairs = []
        for o in objects:
            px, py, pt = self.pos.get(o["name"], (o["x"], o["y"], 0))
            recent = now - pt < TRACK_TIMEOUT
            for i, d in enumerate(dets):
                if o["color"] == d["color"]:
                    dist = np.hypot(d["x"] - px, d["y"] - py)
                    if not recent or dist < TRACK_DIST:
                        pairs.append((dist, i, o["name"]))
        used_d, seen = set(), set()
        for i, name in self._greedy(pairs):
            used_d.add(i); seen.add(name)
            dets[i]["id"], dets[i]["registered"] = name, True
            self.pos[name] = (dets[i]["x"], dets[i]["y"], now)

        # 2) 未登録の物体：前フレームの近い同色物体の番号を引き継ぐ
        reg_names = {o["name"] for o in objects}
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
            self._auto_set(dets[i], k, now)
        # 新しく現れた物体：左→右の順に、空いている一番小さい番号
        taken = reg_names | set(self.auto)
        for i in sorted([i for i in rest if i not in used_d], key=lambda i: (dets[i]["x"], dets[i]["y"])):
            color = dets[i]["color"]
            no = 1
            while f"{color}{no}" in taken:
                no += 1
            taken.add(f"{color}{no}")
            self._auto_set(dets[i], f"{color}{no}", now)

        names = [o["name"] for o in objects]
        missing = [n for n in names if n not in seen]
        order_r = {n: i for i, n in enumerate(names)}
        order_c = {c: i for i, c in enumerate(DEFAULT_COLORS)}   # 5色以外の色は後ろへ
        dets.sort(key=lambda d: (0, order_r.get(d["id"], len(names)), "", 0) if d["registered"]
                  else (1, order_c.get(d["color"], len(order_c)), d["color"], int(d["id"][len(d["color"]):])))
        return dets, missing

    def _auto_set(self, d, key, now):
        d["id"], d["registered"] = key, False
        self.auto[key] = {"color": d["color"], "x": d["x"], "y": d["y"], "t": now}

    def forget(self, name=None):
        if name is None:
            self.pos.clear()
            self.auto.clear()
        else:
            self.pos.pop(name, None)

tracker = Tracker()

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
    top = (p[0] + p[1]) / 2             # QRの上辺の中点
    ang = float(np.degrees(np.arctan2(top[1] - center[1], top[0] - center[0])))
    ang = (ang + CAR_ANGLE_OFFSET + 180) % 360 - 180    # 貼った向きの補正（-180〜180に）
    L = np.linalg.norm(top - center) * 1.5
    front = center + L * np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
    return {"text": text, "x": int(center[0]), "y": int(center[1]),
            "angle": round(ang, 1), "pts": p, "front": front}

def loop():
    global latest_dets
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
        chroma_min = cfg["chroma_min"]

        raw = frame.copy()
        for i, (x, y) in enumerate(corners4):
            cv2.circle(raw, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(raw, str(i + 1), (int(x) + 6, int(y) - 6), font, 0.6, (0, 0, 255), 2)

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
            dets, missing = tracker.update(dets)
            out["missing"] = missing
            for d in dets:
                c = d["_cnt"]
                cx, cy = d["x"], d["y"]
                col = BGR_DRAW.get(d["color"], (255, 255, 255)) if d["registered"] else (150, 150, 150)
                cv2.drawContours(vis, [c], -1, col, 3)
                cv2.drawContours(vis, [c], -1, (255, 255, 255) if d["crossed"] else (0, 0, 0), 1)
                cv2.circle(vis, (cx, cy), 4, (255, 255, 255), -1)
                label = d["id"] if d["registered"] else f"{d['id']}?"
                cv2.putText(vis, f"{label}{' !' if d['crossed'] else ''}", (cx + 6, cy - 4),
                            font, 0.55, (255, 255, 255), 2)
                cv2.putText(vis, f"({cx},{cy})", (cx + 6, cy + 12), font, 0.4, (255, 255, 255), 1)
                out["objects"].append({k: d[k] for k in
                                       ("id", "color", "x", "y", "crossed", "registered", "rgb")})
            if missing:
                cv2.putText(vis, "missing: " + ", ".join(missing), (8, BOARD_H - 10),
                            font, 0.5, (0, 0, 255), 2)

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
            vis = np.zeros((BOARD_H, BOARD_W, 3), np.uint8)
            mask = vis[:, :, 0]
            cv2.putText(vis, "Click 4 board corners on RAW view", (20, BOARD_H // 2),
                        font, 0.7, (255, 255, 255), 2)

        with lock:
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

def register(x, y, color, name):
    """クリック位置の物体を、指定の色・名前で登録（同じ名前なら上書き）"""
    with lock:
        dets = list(latest_dets)
    hit = None
    for d in dets:                      # クリックした点を含む物体
        if cv2.pointPolygonTest(d["_cnt"], (float(x), float(y)), False) >= 0:
            hit = d
            break
    if hit is None and dets:            # 外していたら 30px 以内で一番近い物体
        d = min(dets, key=lambda d: np.hypot(d["x"] - x, d["y"] - y))
        if np.hypot(d["x"] - x, d["y"] - y) < 30:
            hit = d
    if hit is None:
        return "物体が見つかりません。物体の上をクリックしてください"
    if name in tracker.auto and hit["id"] != name:
        return f"{name} は別の未登録の物体が使っています。別の名前にしてください"
    # 同じ名前、または同じ物体（すでに登録済み）の古い登録は消す
    objs = [o for o in cfg["objects"]
            if o["name"] != name and not (hit.get("registered") and o["name"] == hit["id"])]
    for o in cfg["objects"]:
        if o not in objs:
            tracker.forget(o["name"])
    objs.append({"name": name, "color": color, "rgb": hit["rgb"], "x": hit["x"], "y": hit["y"]})
    cfg["objects"] = objs
    tracker.pos[name] = (hit["x"], hit["y"], time.time())
    save()
    print(f"登録: {name} = {color}  RGB={hit['rgb']}  位置=({hit['x']},{hit['y']})")
    return "ok"

PAGE = """<!doctype html><meta charset=utf-8><title>Board Vision</title>
<style>body{font-family:sans-serif;margin:12px} img{max-width:100%;border:1px solid #888;margin:2px}
table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:2px 8px}
.sw{display:inline-block;width:14px;height:14px;vertical-align:middle;border:1px solid #888}</style>
<h3>Board Vision</h3>
<p>① RAW映像で盤の四隅を4回クリック（順番自由。5回目でやり直し）
<button onclick="fetch('/reset')">四隅リセット</button></p>
<img id=raw src=/stream/raw style="cursor:crosshair">
<p>② 物体の登録：<b>色を選ぶ → 名前を確認 → 盤面の物体をクリック</b>（全部の物体について繰り返す）<br>
色 <select id=color onchange="suggest()">
<option>red</option><option>orange</option><option>yellow</option><option>green</option><option>blue</option></select>
名前 <input id=name size=12>（半角英数字。画面の文字表示が日本語に対応していないため）
<span id=msg style="color:#c00"></span></p>
<img id=board src=/stream/board style="cursor:crosshair"> <img src=/stream/mask width=300>
<p>登録済みの物体 <button onclick="if(confirm('全部消しますか？'))fetch('/clear').then(load)">全部消す</button></p>
<table id=tbl></table>
<p>彩度しきい値 <input type=range id=cm min=10 max=200 oninput="setcm()"> <span id=cmv></span>
（白線や影まで拾うなら上げる／物体が欠けるなら下げる）</p>
<pre id=out></pre>
<script>
let objs = [];
function pos(img, e) {
  const b = img.getBoundingClientRect();
  return [(e.clientX - b.left) * img.naturalWidth / b.width,
          (e.clientY - b.top) * img.naturalHeight / b.height];
}
raw.onclick = e => { const [x, y] = pos(raw, e); fetch(`/click?x=${x}&y=${y}`); };
board.onclick = e => {
  const [x, y] = pos(board, e);
  const n = name.value.trim();
  if (!/^[A-Za-z0-9_-]+$/.test(n)) { msg.textContent = '名前は半角英数字で入力してください'; return; }
  fetch(`/register?x=${x}&y=${y}&color=${color.value}&name=${encodeURIComponent(n)}`)
    .then(r => r.text()).then(t => { msg.textContent = t == 'ok' ? '' : t; load(); });
};
let autoIds = [];
function suggest() {              // 次の空き番号を提案（red1, red2 ...）。未登録の物体が使っている番号も避ける
  const used = new Set(objs.map(o => o.name).concat(autoIds));
  let i = 1; while (used.has(color.value + i)) i++;
  name.value = color.value + i;
}
function load() {
  fetch('/cfg').then(r => r.json()).then(c => {
    objs = c.objects;
    tbl.innerHTML = '<tr><th>名前</th><th>色</th><th>実測RGB</th><th>登録位置</th><th></th></tr>' +
      objs.map(o => `<tr><td>${o.name}</td><td>${o.color}</td>
        <td><span class=sw style="background:rgb(${o.rgb})"></span> ${o.rgb}</td>
        <td>(${o.x},${o.y})</td>
        <td><button onclick="fetch('/delete?name=${o.name}').then(load)">削除</button></td></tr>`).join('');
    cm.value = c.chroma_min; cmv.textContent = c.chroma_min;
    suggest();
  });
}
function setcm() { cmv.textContent = cm.value; fetch('/chroma?v=' + cm.value); }
load();
setInterval(() => fetch('/result').then(r => r.json())
  .then(j => {
    out.textContent = JSON.stringify(j, null, 1);
    autoIds = (j.objects || []).filter(o => !o.registered).map(o => o.id);
  }), 500);
</script>"""

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, body, ctype="text/plain; charset=utf-8"):
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
        elif u.path == "/register":
            msg = register(float(q["x"][0]), float(q["y"][0]), q["color"][0], q["name"][0])
            return self.reply(msg.encode())
        elif u.path == "/delete":
            name = q["name"][0]
            cfg["objects"] = [o for o in cfg["objects"] if o["name"] != name]
            tracker.forget(name)
            save()
        elif u.path == "/clear":
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
