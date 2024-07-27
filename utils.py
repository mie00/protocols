import socket

class ReaderWrapper:
    def __init__(self, initial_value=b''):
        self.buffer = initial_value
        self.done = False
        
    def connect(self, source_port=None):
        if source_port is not None:
            self.sock.bind(('0.0.0.0', source_port))
        self.sock.connect((self.host, self.port))

    def close(self):
        self.sock.close()
        
    def send(self, data):
        self.sock.sendall(data)
        
    def _read_to_buffer(self):
        if not self.done:
            buffer = self._read()
            if buffer == b'':
                self.done = True
                return False
            self.buffer += buffer
            return True
        else:
            raise RuntimeError("socket closed can't read")

    def peek_once(self):
        return self.buffer

    def read_once(self):
        self._read_to_buffer()
        out, self.buffer = self.buffer, b''
        return out

    def read_line(self):
        return self.read_until(b'\r\n')

    def read_until(self, delim):
        while delim not in self.buffer:
            self._read_to_buffer()
        cut = self.buffer.index(delim) + len(delim)
        out, self.buffer = self.buffer[:cut], self.buffer[cut:]
        return out
    
    def read_len(self, l):
        while len(self.buffer) < l:
            self._read_to_buffer()
        out, self.buffer = self.buffer[:l], self.buffer[l:]
        return out

    def read_all(self):
        while self._read_to_buffer():
            pass
        out, self.buffer = self.buffer, b''
        return out
        
class SocketReader(ReaderWrapper):
    def __init__(self, host, port, type='tcp', initial_value=b'', alpn=['http/1.1']):
        self.host = host
        self.port = port
        t = socket.SOCK_DGRAM if type == 'udp' else socket.SOCK_STREAM
        self.sock = socket.socket(family=socket.AF_INET, type=t)
        if type == 'tls':
            import ssl
            context = ssl._create_default_https_context()
            context.set_alpn_protocols(['http/1.1'])
            if context.post_handshake_auth is not None:
                context.post_handshake_auth = True
            self.sock = context.wrap_socket(self.sock, server_hostname=host)
        super().__init__()

    def _read(self):
        return self.sock.recv(4096)


