
import math
import re

r0 = lambda x: x.replace(' ', '').replace('\n', '')
r1 = lambda x: re.sub(r'([0-9]+)B', lambda m: str(8 * int(m.group(1))), x)
r2 = lambda x: re.sub(r'0x(.+?)\b', lambda m: str(int(m.group(1), 16)), x)

def parse(xx):
    print = lambda *x: 1
    print(xx)
    depth = 0
    start = -1
    ls = []
    o = ''
    for i in range(len(xx)):
        if xx[i] == '(':
            if depth == 0:
                start = i
            depth += 1
        elif xx[i] == ')':
            depth -=1
            if depth == 0:
                ls.append(parse(xx[start+1:i]))
                o += "!" + str(len(ls) - 1)
        else:
            if depth == 0:
                o += xx[i]
    return list(map(lambda x: x.split('-'), o.split(':'))), ls


import string

class Repr:
    def __init__(self, val, r='h', strict_length=False):
        self.val = val
        self.r = r
        self.strict_length=strict_length
    def __len__(self):
        return len(self.val)
    def __repr__(self):
#         return str(self.value())
        if len(self.val) == 0:
            return "''"
        if len(self.val) % 8 != 0:
            if self.strict_length:
                return "c_bin('" + self.val + "')"
            else:
                return "0b" + self.val
        if all(i in string.printable for i in self.str()):
            return '"' + self.str().replace('"', r'\"') + '"'
        if self.strict_length:
            return "c_bin('" + self.hex() + "')"
        else:
            return "0x" + self.hex()
    def __str__(self):
        return str(self.value())
    def int(self):
        return int(self.val, 2)
    def hex(self):
        return ''.join("%02x"%(int(self.val[i:i+8], 2)) for i in range(0, len(self.val), 8))
    def str(self):
        return ''.join(chr(int(self.val[i:i+8], 2)) for i in range(0, len(self.val), 8))
    def bytes(self):
        return bytes(bytearray(int(self.val[i:i+8], 2) for i in range(0, len(self.val), 8)))
    def value(self):
        if self.r == 'b':
            return self.val
        elif self.r == 's':
            return self.str()
        elif self.r == 'h':
            return self.hex()
        elif self.r == 'i':
            return self.int()
        else:
            raise RuntimeError("unknown r: %s"%(self.r))

def sum_match(i):
    s = 0
    if i is None:
        s += 0
    elif type(i) is dict:
        s += sum(sum_match(v) for v in i.values())
    elif type(i) is list:
        s += sum(sum_match(v) for v in i)
    else:
        s += len(i)
    return s

def match(p, line, print=lambda *x: 1):
    if line == '':
        return Repr('', 'h')
    print(len(line), line)
    matches = []
    ci = 0
    pp = p[0]
    ch = p[1]
    cs = None
    selector = None
    for e in pp:
        if '?' not in e[0]:
            cs = e
            break
        l, v = e[0].split('?')[0].split('==')
        if int(line[ci:ci+int(l)], 2) == int(v):
            cs = [e[0].split('?')[1]] + e[1:]
            selector = Repr(line[ci:ci+int(l)], 'h')
            ci += int(l)
            break
    if cs is None:
        raise RuntimeError("unable to find %s for %s"%(p, line))
    print(cs)
    ms = {}
    for i, c in enumerate(cs):
        name = re.search(r'@[a-z0-9_]+', c)
        name = name.group(0)[1:] if name else '_undefined'
        c = re.sub(r'@[a-z0-9_]+', '', c)
        rep = '...' in c
        if '...' in c:
            c = c[3:]
        mmm = []
        while len(line[ci:]):
            print(c)
            if c == '':
                mmm.append(Repr(line[ci:], 'h'))
                ci = len(line)
            elif '!' not in c:
                mmm.append(Repr(line[ci:ci+int(c)], 'h'))
                ci += int(c)
            else:
                has_length2 = None
                has_suffix = None
                cline = line[ci:]
                if c[0] != '!':
                    spec_len = c.split('!')[0]
                    if c[0] != '~':
                        mod = 0
                        if '+' in spec_len or '-' in spec_len:
                            segs = spec_len.replace('-', '+-').split('+')
                            spec_len = segs[0]
                            mod = int(segs[1])
                        ll = int(spec_len)
                        ccll = (int(cline[:ll], 2) + mod) * 8
                        print(ll, ccll)
                        has_length2 = Repr(line[ci:ci+ll], 'i')
                        ci += ll
                        cline = cline[ll:ll+ccll]
                    else:
                        suffix = spec_len[1:].encode()
                        suffix_bin = bytes2bin(suffix)
                        print(suffix_bin)
                        for inn in range(math.ceil(ci/8)*8, len(line), 8):
                            print(line[inn:])
                            if line[inn:].startswith(suffix_bin):
                                has_suffix = Repr(suffix_bin, 'b')
                                cline = line[ci:inn]
                                break
                chn = int(c.split('!')[1])
                chm = match(ch[chn], cline)
                cll = sum_match(chm)
                ci += cll
                if has_length2 or has_suffix:
                    if type(chm) is not dict:
                        print('unexpected type')
                        chm = {
                            "_data": chm,
                        }
                    if has_length2:
                        chm["_length"] = has_length2
                    elif has_suffix:
                        chm["_suffix"] = has_suffix
                        ci += len(has_suffix)
                mmm.append(chm)
                if c[0] != '!' and cll != len(cline):
                    raise RuntimeError("fixed length child not matching %s %d != %d %s %s"%(c, cll, len(cline), cline, chm))
            if not rep:
                break
        if not rep:
            mmm = mmm[0] if len(mmm) else ''
        if name in ms:
            raise RuntimeError("%s already exists"%name)
        ms[name] = mmm
        if selector:
            ms["_selector"] = selector
    if len(ms) == 1 and '_undefined' in ms:
        ms = ms['_undefined']
    return ms

def hex2bin(b):
    return bytes2bin(bytes.fromhex(b))

def bytes2bin(b):
    return ''.join([("{0:08b}").format(i) for i in b])

def bin2bytes(b):
    return Repr(b).bytes()

def expand_bytes(b, l):
    if type(b) is Repr:
        b = b.val.encode()
    elif type(b) is str:
        b = b.decode()
        b = bytes2bin(b).encode().lstrip(b'0')
    else:
        b = bytes(b)
        b = bytes2bin(b).encode().lstrip(b'0')
    if len(b) > l:
        if b[:len(b) - l] == b'0' * (len(b) - l):
            return b[len(b) - l:]
        else:
            raise RuntimeError("%s has more bits than %d"%(b, l))
    return b'0' * (l - len(b)) + b
def expand_length(i, l):
    return expand_bytes(i.to_bytes(int(l/8), 'big'), l)

def _construct(p, obj, print=lambda *x: 1):
    ret = bytearray()
    pp = p[0]
    ch = p[1]
    cs = None
    for e in pp:
        if '?' not in e[0]:
            cs = e
            break
        l, v = e[0].split('?')[0].split('==')
        if '_selector' in obj and obj['_selector'].int() == int(v):
            cs = [e[0].split('?')[1]] + e[1:]
            ret += expand_bytes(obj['_selector'], int(l))
            break
    if cs is None:
        raise RuntimeError("unable to find %s for %s"%(p, obj))
    for i, c in enumerate(cs):
        print('raw_c', c)
        name = re.search(r'@[a-z0-9_]+', c)
        name = name.group(0)[1:] if name else '_undefined'
        print('name', name)
        c = re.sub(r'@[a-z0-9_]+', '', c)
        print('c', c)
        rep = '...' in c
        if '...' in c:
            c = c[3:]
            
        data = obj
        if rep:
            if type(obj) is Repr:
                data = [obj]
            elif type(obj) is not list:
                if '_data' in obj and type(obj['_data']) is list:
                    data = obj['_data']
                elif name in obj and type(obj[name]) is list:
                    data = obj[name]
                else:
                    raise RuntimeError("cannot get list out of %s", obj)
        else:
            data = [obj]
        print(data)
        for d in data:
            print('d', d)
            el = d
            if type(el) is dict and name in el:
                    el = el[name]
            if type(el) is dict and '_data' in el:
                    el = el['_data']
            print('el', el)
            if c == '':
                print(repr(c), el)
                ret += el.val.encode()
            elif '!' not in c:
                print(c, data, name, d, el, c)
                ret += expand_bytes(el, int(c))
            else:
                chn = int(c.split('!')[1])
                if type(el) is Repr and len(el) == 0:
                    chm = b''
                else:
                    chm = _construct(ch[chn], el, print=print)
                if c[0] != '!':
                    spec_len = c.split('!')[0]
                    if c[0] != '~':
                        mod = 0
                        if '+' in spec_len or '-' in spec_len:
                            segs = spec_len.replace('-', '+-').split('+')
                            spec_len = segs[0]
                            mod = int(segs[1])
                        ll = int(spec_len)
                        print('chm', chm, len(chm))
                        ret += expand_length(int(len(chm)/8) - mod, ll)
                ret += chm
                if c[0] != '!':
                    spec_len = c.split('!')[0]
                    if c[0] == '~':
                        suffix = spec_len[1:]
                        ret += bytes2bin(suffix.encode()).encode()
            print('ret', ret)
    return ret
                    

def all_match(p, line, allow_rem=False, print=lambda *x: 1):
    m = match(p, line, print=print)
    if not allow_rem and sum_match(m) != len(line):
        raise RuntimeError("not matched: %d, %d"%(sum_match(m), len(line)))
    if allow_rem:
        return m, line[sum_match(m):]
    return m

def process_spec(spec):
    return parse(r2(r1(r0(spec))))



class c_hex:
    def __init__(self, val):
        self.val = val


def preconstruct(tree):
    if type(tree) is Repr:
        return tree
    elif type(tree) is int:
        b = bin(tree)[2:]
        b = '0' * (-len(b) % 8) + b
        return Repr(b)
    elif type(tree) is c_hex:
        return Repr(hex2bin(tree.val))
    elif type(tree) is str:
        return Repr(bytes2bin(tree.encode()))
    elif type(tree) is bytes:
        return Repr(bytes2bin(tree))
    elif type(tree) is dict:
        ret = {}
        for i in tree:
            ret[i] = preconstruct(tree[i])
        return ret
    elif type(tree) is list:
        ret = []
        for i in tree:
            ret.append(preconstruct(i))
        return ret
    else:
        raise RuntimeError("unable to decode %s of type %s", tree, type(tree))


def construct(p, obj, print=lambda *x: 1):
    return bin2bytes(_construct(p, preconstruct(obj), print=print))



import json
class MyEncoder(json.JSONEncoder):
        def default(self, o):
            if type(o) is Repr:
                return o.value()
            return super().default(o)

def pp(m):
    return json.dumps(m, cls=MyEncoder, indent=2)


#=====================================

import random
import os

def public_construct(spec, data):
    return construct(process_spec(spec), data)

def public_match(spec, data):
    obj, rem = all_match(process_spec(spec), bytes2bin(data), allow_rem=True)
    return repr(obj), bin2bytes(rem)

def public_match_exact(spec, data):
    rand = b''
    while rand not in data:
        rand = os.urandom(5)
    obj, rem = all_match(process_spec(spec), bytes2bin(data), allow_rem=True)
    if len(rem) == len(rand):
        return repr(obj)
    elif len(rem) < len(rand):
        raise RuntimeError("need more data")
    else:
        raise RuntimeError("has more data")

def repr(r):
    if type(r) is list:
        return [repr(i) for i in r]
    if type(r) is not dict:
        return r
    
    
    if "_length" in r:
        del r["_length"]
    if "_suffix" in r:
        del r["_suffix"]
    if len(r.keys()) == 1:
        for k in r.keys():
            if k[0] == '_':
                return repr(r[k])
    return {k: repr(v) for (k, v) in r.items()}
