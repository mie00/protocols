import struct
from fcntl import ioctl
from stun import do_stun
import random
import socket
import threading

sport = random.randrange(30000, 50000)

dstn = do_stun(sport)
print(dstn[0], dstn[1])

print("copy the above line to the target server and from the target server to here")
remote=input("paste the host port and press enter: ")
remote_host, remote_port = remote.split(' ')


t = socket.SOCK_DGRAM
sock = socket.socket(family=socket.AF_INET, type=t)
sock.bind(('0.0.0.0', int(dstn[1])))
sock.connect((remote_host, int(remote_port)))
SOL_SOCKET = 1
SO_SNDBUF = 7
SO_RCVBUF = 8




def main(tun):
    with open("/dev/net/tun", "r+b", buffering=0) as tun:
        def handle_packets():
            while True:
                packet = sock.recv(2048)[3:]
                print('w', packet)
                tun.write(packet)
        
        
        LINUX_IFF_TUN = 0x0001
        LINUX_IFF_NO_PI = 0x1000
        LINUX_TUNSETIFF = 0x400454CA
        flags = LINUX_IFF_TUN | LINUX_IFF_NO_PI
        ifs = struct.pack("16sH22s", b"tun1", flags, b"")
        ioctl(tun, LINUX_TUNSETIFF, ifs)
        threading.Thread(target=handle_packets).start()
        while True:
            packet = tun.read(2048)
            print('r', packet)
            sock.sendall(b'\x17\xfe\xfd' + packet)

if __name__ == "__main__":

    curr_r_opt = 2110000000
    curr_w_opt = 2110000000
    
    sock.setsockopt(SOL_SOCKET, SO_SNDBUF, curr_r_opt)
    sock.setsockopt(SOL_SOCKET, SO_RCVBUF, curr_w_opt)
    
    import sys
    main("tun1" if len(sys.argv) == 1 else sys.argv[1])

