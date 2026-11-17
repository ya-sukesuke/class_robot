#!/usr/bin/env python3
"""
car_receiver.py - ノートPC(board_vision.py)から座標を受け取る（ラズパイで実行）

main.py からの使い方:
    import car_receiver
    car_receiver.start()                       # 受信を裏で開始（最初に1回）
    car, objs = car_receiver.get_location()    # 最新の車と物体の座標を取る（何度でも）

  car  : {"text": "car1", "x": 300, "y": 80, "angle": -90.0}
         見えていない／データが古いときは None
  objs : {"object1": {"id": "object1", "color": "red", "x": 120, "y": 250,
                      "crossed": true, "registered": false, "rgb": [...]}, ...}
         objs["object3"] のように名前で取り出せる。データが古いときは {}

  x, y : 補正後の盤面の座標[px]。左上が(0,0)、右が x、下が y
  angle: 車の向き[度]。右=0, 下=+90, 上=-90, 左=180

単体で動かすと、受信したデータを表示するだけ:  python3 car_receiver.py
"""
import socket, json, threading, time

PORT = 5005
TIMEOUT = 0.5   # この秒数データが来なければ「古い」とみなす（main側で車を止める）

_lock = threading.Lock()
_latest = None          # 最後に受信したデータ
_latest_t = 0.0         # 受信した時刻（ラズパイの時計）
_thread = None

def _recv_loop():
    global _latest, _latest_t
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT))
    print(f"UDP {PORT} で待受中...")
    while True:
        msg, _ = sock.recvfrom(65535)
        try:
            data = json.loads(msg)
        except ValueError:
            continue
        with _lock:
            _latest, _latest_t = data, time.time()

def start():
    """受信スレッドを開始（2回呼んでも1つだけ）"""
    global _thread
    if _thread is None:
        _thread = threading.Thread(target=_recv_loop, daemon=True)
        _thread.start()

def get_data():
    """最新の受信データ全体（dict）。古い・未受信なら None"""
    with _lock:
        if _latest is None or time.time() - _latest_t > TIMEOUT:
            return None
        return _latest

def get_location():
    """(car, objs) を返す。データが古いときは (None, {})"""
    data = get_data()
    if data is None:
        return None, {}
    objs = {o["id"]: o for o in data.get("objects", [])}
    return data.get("car"), objs

if __name__ == "__main__":
    start()
    while True:
        car, objs = get_location()
        print("車:", car)
        for o in objs.values():
            print(f"  {o['id']}: {o['color']} ({o['x']},{o['y']}) ライン越え={o['crossed']}")
        time.sleep(1)
