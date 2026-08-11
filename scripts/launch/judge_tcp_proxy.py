#!/usr/bin/env python3
"""Forward local TLS connections to a judge endpoint from the host network."""

from __future__ import annotations

import argparse
import selectors
import socket
import threading


def relay(left: socket.socket, right: socket.socket) -> None:
    selector = selectors.DefaultSelector()
    selector.register(left, selectors.EVENT_READ, right)
    selector.register(right, selectors.EVENT_READ, left)
    try:
        while selector.get_map():
            for key, _ in selector.select():
                source = key.fileobj
                destination = key.data
                try:
                    chunk = source.recv(65536)
                except OSError:
                    chunk = b""
                if not chunk:
                    return
                destination.sendall(chunk)
    finally:
        selector.close()
        left.close()
        right.close()


def serve(client: socket.socket, upstream_host: str, upstream_port: int) -> None:
    try:
        upstream = socket.create_connection((upstream_host, upstream_port), timeout=15)
    except OSError:
        client.close()
        return
    relay(client, upstream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-port", type=int, default=18443)
    parser.add_argument("--upstream-host", required=True)
    parser.add_argument("--upstream-port", type=int, default=443)
    args = parser.parse_args()

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", args.listen_port))
    listener.listen()
    while True:
        client, _ = listener.accept()
        threading.Thread(target=serve, args=(client, args.upstream_host, args.upstream_port), daemon=True).start()


if __name__ == "__main__":
    main()
