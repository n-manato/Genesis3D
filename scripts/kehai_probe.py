#!/usr/bin/env python3
"""KEHAI の実機（headless Chrome）で JS を評価する。**依存なしの最小 CDP クライアント。**

## なぜ自前で書くか

生成したアバターが KEHAI で本当に動くかは、構造検査（`check_kehai_model.py`）では
分からない。実機でモーフが動くか・spring bone で髪が揺れるかを見る必要がある。
Chrome DevTools Protocol は WebSocket 越しなので普通は `websocket-client` を使うが、
この機材のどの conda env にも入っていない。CDP に必要なのは
「マスク付きテキストフレームを送って、マスク無しで返るのを読む」だけなので、
標準ライブラリで足りる。

使い方:
    python3 scripts/kehai_probe.py --expr "location.href"
    python3 scripts/kehai_probe.py --file probe.js --await
    python3 scripts/kehai_probe.py --model subject_a_v3.yaml   # モデルを切り替える
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import struct
import sys
import urllib.request
from pathlib import Path


class WS:
    """CDP を叩くのに足るだけの WebSocket クライアント。"""

    def __init__(self, url: str, timeout: float = 30.0):
        assert url.startswith("ws://"), url
        rest = url[5:]
        hostport, _, path = rest.partition("/")
        host, _, port = hostport.partition(":")
        self.sock = socket.create_connection((host, int(port or 80)), timeout)
        self.sock.settimeout(timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (f"GET /{path} HTTP/1.1\r\nHost: {hostport}\r\n"
               f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
        self.sock.sendall(req.encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise SystemExit("ハンドシェイクが切れた")
            buf += chunk
        if b"101" not in buf.split(b"\r\n")[0]:
            raise SystemExit(f"Upgrade 失敗: {buf.split(chr(13).encode())[0]!r}")
        self.buf = buf.split(b"\r\n\r\n", 1)[1]
        self._id = 0

    def _recv(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise SystemExit("接続が切れた")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def send(self, payload: str) -> None:
        data = payload.encode()
        head = bytearray([0x81])
        n = len(data)
        if n < 126:
            head.append(0x80 | n)
        elif n < 1 << 16:
            head.append(0x80 | 126)
            head += struct.pack(">H", n)
        else:
            head.append(0x80 | 127)
            head += struct.pack(">Q", n)
        mask = os.urandom(4)
        head += mask
        self.sock.sendall(bytes(head) + bytes(b ^ mask[i % 4]
                                              for i, b in enumerate(data)))

    def recv(self) -> str:
        while True:
            b0, b1 = self._recv(2)
            op, n = b0 & 0x0F, b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._recv(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._recv(8))[0]
            data = self._recv(n)
            if op == 0x1:
                return data.decode()
            if op == 0x8:
                raise SystemExit("サーバが閉じた")

    def call(self, method: str, **params):
        self._id += 1
        mid = self._id
        self.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(self.recv())
            if msg.get("id") == mid:
                return msg


def pick_target(port: int, url_filter: str):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=10) as r:
        targets = json.load(r)
    pages = [t for t in targets
             if t.get("type") == "page" and url_filter in (t.get("url") or "")]
    if not pages:
        raise SystemExit(f"'{url_filter}' のページが無い: "
                         f"{[(t.get('type'), t.get('url')) for t in targets]}")
    return pages[0]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=9222)
    ap.add_argument("--url", default="12394", help="対象ページの URL に含まれる文字列")
    ap.add_argument("--expr", help="評価する JS")
    ap.add_argument("--file", type=Path, help="JS をファイルから読む")
    ap.add_argument("--await", dest="awaited", action="store_true",
                    help="Promise を待つ")
    ap.add_argument("--model", help="localStorage.lastConfigFile を差し替えて再読込")
    ap.add_argument("--shot", type=Path, help="画面を撮る（Page.captureScreenshot）")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    t = pick_target(args.port, args.url)
    ws = WS(t["webSocketDebuggerUrl"], args.timeout)
    ws.call("Runtime.enable")

    def ev(expr, awaited=False):
        r = ws.call("Runtime.evaluate", expression=expr, returnByValue=True,
                    awaitPromise=awaited, userGesture=True)
        res = r.get("result", {})
        if "exceptionDetails" in res:
            d = res["exceptionDetails"]
            raise SystemExit("JS 例外: " + json.dumps(
                d.get("exception", {}).get("description", d), ensure_ascii=False))
        return res.get("result", {}).get("value")

    if args.model:
        print(f"モデルを {args.model} に切り替えて再読込")
        ev(f"localStorage.setItem('lastConfigFile', {json.dumps(args.model)}); "
           f"location.reload(); 'ok'")
        return

    if args.shot:
        r = ws.call("Page.captureScreenshot", format="png")
        data = r.get("result", {}).get("data")
        if not data:
            raise SystemExit(f"撮影に失敗: {json.dumps(r)[:200]}")
        args.shot.parent.mkdir(parents=True, exist_ok=True)
        args.shot.write_bytes(base64.b64decode(data))
        print(f"{args.shot}（{args.shot.stat().st_size/1024:.0f}KB）")
        if not (args.expr or args.file):
            return

    expr = args.file.read_text() if args.file else args.expr
    if not expr:
        raise SystemExit("--expr か --file か --model のいずれかを指定")
    out = ev(expr, args.awaited)
    print(json.dumps(out, ensure_ascii=False, indent=2)
          if isinstance(out, (dict, list)) else out)


if __name__ == "__main__":
    main()
