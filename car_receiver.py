#!/usr/bin/env python3
"""
car_receiver.py - ノートPC(board_vision.py)から座標を受け取る（ラズパイで実行）

使い方:  python3 car_receiver.py
受信データ例:
  {"objects": [{"x": 120, "y": 250, "crossed": true}],
   "car": {"text": "car1", "x": 300, "y": 80, "angle": -90.0}, "t": 1731200000.1}
  angle: 画像座標での車の向き[度]。右=0, 下=+90, 上=-90, 左=180
"""
import socket, json, time

PORT = 5005
TIMEOUT = 0.5   # この秒数データが来なければ車を止める（安全のため）

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT))
sock.settimeout(TIMEOUT)
print(f"UDP {PORT} で待受中...")

def stop_car():
    pass  # TODO: Robot HAT のモータを停止する処理

def control(data):
    """ここに車の制御を書く"""
    car = data.get("car")
    targets = [o for o in data.get("objects", []) if o["crossed"]]
    if car is None:
        stop_car()          # 車(マーカー)が見えないときは止める
        return
    if targets:
        t = targets[0]
        print(f"車({car['x']},{car['y']}) 向き{car['angle']}° → 目標({t['x']},{t['y']})")
        # TODO: 目標へ向かうモータ制御

last_print = 0
while True:
    try:
        msg, addr = sock.recvfrom(4096)
    except socket.timeout:
        stop_car()
        continue
    data = json.loads(msg)
    control(data)
    if time.time() - last_print > 1:
        print("受信:", data)
        last_print = time.time()
