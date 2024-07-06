#!/bin/python

import sys
import socket

req = b'\x00\x01\x00\x00!\x12\xa4B\x95Q:\xce;\x90:\x9ew\xcc\x05E'
def do_stun(sport, server='stun.l.google.com'):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', sport))
    sock.sendto(req, (server, 19302))
    resp = sock.recv(1024)
    
    def xor(str1, str2):
        return bytes(a ^ b for a, b in zip(str1, str2))
    
    pb = xor(resp[26:28], b'\x21\x12')
    ipb = xor(resp[28:], b'\x21\x12\xa4\x42')
    return f"{ipb[0]}.{ipb[1]}.{ipb[2]}.{ipb[3]}", pb[0] * 256 + pb[1]

if __name__ == "__main__":
    for server in ['stun.l.google.com', 'stun2.l.google.com']:
        ip, port = do_stun(int(sys.argv[1]), server)
        print(f"server {server} says: {ip}:{port}")
