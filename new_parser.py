#!/usr/bin/env python3

failed_any = False

def expect(case, spec, inp, out, **kwargs):
    global failed_any
    print("testing:", case)
    oe = ie = None
    if not isinstance(inp, Exception):
        try:
            o = construct(spec, eval(inp), **kwargs)
        except Exception as e:
            oe = e
            if kwargs.get('should_raise', False):
                raise
        if oe is not None:
            if type(oe) is type(out) and oe.args == out.args:
                print("\x1b[32m passed construct\x1b[0m")
            else:
                failed_any = True
                print("\x1b[31m failed construct\x1b[0m")
                print("raised", oe)
        elif isinstance(out, Exception):
            failed_any = True
            print("\x1b[31m failed construct\x1b[0m")
            print("didn't throw")
            print(out)
        elif o == out:
            print("\x1b[32m passed construct\x1b[0m")
        else:
            failed_any = True
            print("\x1b[31m failed construct\x1b[0m")
            print("values do not match")
            print("expected", out)
            print("got", o)
    if not isinstance(out, Exception):
        try:
            i, rem = match(spec, out, **kwargs)
        except Exception as e:
            ie = e
            if kwargs.get('should_raise', False):
                raise
        if ie is not None:
            if type(ie) is type(inp) and ie.args == inp.args:
                print("\x1b[32m passed match\x1b[0m")
            else:
                failed_any = True
                print("\x1b[31m failed match\x1b[0m")
                print("raised", ie)
        elif isinstance(inp, Exception):
            failed_any = True
            print("\x1b[31m failed match\x1b[0m")
            print("didn't throw")
            print("expected", inp)
        elif repr(i).replace('\n', '').replace(' ', '') == inp.replace('\n', '').replace(' ', ''):
            if rem == kwargs.get(rem, b''):
                print("\x1b[32m passed match\x1b[0m")
                try:
                    rev = construct(spec, i, **kwargs)
                    if rev == out:
                        print("\x1b[32m passed revert\x1b[0m")
                    else:
                        failed_any = True
                        print("\x1b[31m failed revert\x1b[0m")
                        print("expected", out)
                        print("got", rev)
                except Exception as re:
                    failed_any = True
                    print("\x1b[31m failed revert\x1b[0m")
                    if kwargs.get('should_raise', False):
                        raise
                    print("raised", re)
            else:
                failed_any = True
                print("\x1b[31m failed match\x1b[0m")
                print(" remainings do not match\x1b[0m")
                print("expected", kwargs.get(rem, b''))
                print("got", rem)
        else:
            failed_any = True
            print("\x1b[31m failed match\x1b[0m")
            print("values do not match\x1b[0m")
            print("expected", inp)
            print("got", i)

import ply.lex as lex
import ply.yacc as yacc

# Define the tokens
tokens = (
    'LPAREN',
    'RPAREN',
    'CONCAT',
    'OR',
    'REFERENCE_SIZE',
    'EQUALS',
    'SIZE',
    'NAME',
    'TYPE',
    'SUFFIX',
    'REPETITION',
)


t_LPAREN = r'\('
t_RPAREN = r'\)'
t_CONCAT = r'-'
t_REFERENCE_SIZE = r'\{[a-zA-Z0-9_]+\}'
t_NAME = r'@[a-zA-Z0-9_]+'
t_TYPE = r'\[[a-z0-9]+\]'
t_SUFFIX = r'~[a-zA-Z0-9\\]+'
t_REPETITION = r'\.\.\.'
t_OR = r'\|'
t_SIZE = r'\d+[Bb]?'

t_ignore = ' \t'


# Define a rule so we can track line numbers
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

# Define a rule for comments
def t_comment(t):
    r'\#.*'
    pass  # No return value. Token discarded

def t_EQUALS(t):
    r'==(?P<equals_hex>(?:0x)|(?:0b)|)(?P<equals_val>[0-9a-f]+)'
    t.value = int(t.lexer.lexmatch['equals_val'], 16 if t.lexer.lexmatch['equals_hex'] == '0x' else 2 if t.lexer.lexmatch['equals_hex'] == '0b' else 10)
    return t

def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

lexer = lex.lex()

# Define the precedence and associativity of operators
precedence = (
    ('left', 'OR'),
    ('left', 'CONCAT'),
)

# Define the grammar rules
def p_expression_or(p):
    'expression : expression OR expression'
    p[0] = ('or', p[1], p[3])

# Define the grammar rules
def p_expression_concat(p):
    'expression : expression CONCAT expression'
    p[0] = ('concat', p[1], p[3])

# Define the grammar rules
def p_expression_leading_dash(p):
    'expression : CONCAT term'
    p[0] = p[2]

def p_expression_term(p):
    'expression : term'
    p[0] = p[1]

def p_term_repetition(p):
    'term : optional_reference_size REPETITION term'
    p[0] = ('repetition', p[1], p[3])

def p_term_field(p):
    'term : all_size optional_name optional_type'
    p[0] = ('field', p[1], p[2], p[3])

def p_term_grouped(p):
    'term : all_size optional_name optional_type LPAREN expression RPAREN'
    p[0] = ('grouped', p[1], p[2], p[3], p[5])

def p_term_grouped_empty(p):
    'term : all_size optional_name optional_type LPAREN RPAREN'
    p[0] = ('len_field', p[1], p[2], p[3])

def p_all_size(p):
    '''all_size : SIZE optional_equals
                | REFERENCE_SIZE
                | SUFFIX
                | empty'''
    if p[1] is None:
        p[0] = ('rest',)
    elif p[1][0] == '~':
        p[0] = ('suffix', p[1][1:])
    elif p[1][0] == '{':
        p[0] = ('reference', p[1][1:-1])
    else:
        if p[1][-1] in 'Bb':
            val = int(p[1][:-1]) * 8
        else:
            val = int(p[1])
        p[0] = ('literal', val, p[2])

def p_optional_equals(p):
    '''optional_equals : EQUALS
                       | empty'''
    if len(p) == 2 and p[1] is not None:
        p[0] = p[1]
    else:
        p[0] = None

def p_optional_reference_size(p):
    '''optional_reference_size : REFERENCE_SIZE
                               | empty'''
    if len(p) == 2 and p[1] is not None:
        p[0] = p[1][1:-1]
    else:
        p[0] = None
    

def p_optional_name(p):
    '''optional_name : NAME
                     | empty'''
    if len(p) == 2 and p[1] is not None:
        p[0] = p[1][1:]
    else:
        p[0] = None

def p_optional_type(p):
    '''optional_type : TYPE
                     | empty'''
    if len(p) == 2 and p[1] is not None:
        p[0] = p[1][1:-1]
    else:
        p[0] = None

def p_empty(p):
    'empty :'
    pass

def p_error(p):
    print(f"Syntax error at '{p.value}'")


parser = yacc.yacc()


def encode_num(val, size, type_=None):
    if type_ is not None and type_ != 'int':
        if type_ == 'int8':
            if size is None:
                size = 8
            elif size != 8:
                raise ValueError("cannot encode {} into size {}".format(type_, size))
        elif type_ == 'int16':
            if size is None:
                size = 16
            elif size != 16:
                raise ValueError("cannot encode {} into size {}".format(type_, size))
        elif type_ == 'int32':
            if size is None:
                size = 32
            elif size != 32:
                raise ValueError("cannot encode {} into size {}".format(type_, size))
    else:
        if size is None:
            raise ValueError("cannot encode int with unknown size")
            
    orig_val = val
    ret = bytearray(b'0' * (size))
    for i in range(len(ret) - 1, -1, -1):
        ret[i] = ord(b'1') if val % 2 else ord(b'0')
        val //= 2
    if val != 0:
        raise ValueError("number {} doesn't fit into {} bits".format(orig_val, size))
    return ret

def encode_str(val, size):
    if size is not None:
        if size % 8 != 0:
            raise ValueError("don't know how to encode a string into {} bits".format(size))
        if len(val) != size // 8:
            raise ValueError("string {} doesn't fit into {} bytes".format(repr(val), size // 8))
    return b''.join(bytes(format(ord(i),'08b'), 'ascii') for i in val)

def encode_bytes(val, size):
    if size is not None:
        if size % 8 != 0:
            raise ValueError("don't know how to encode bytes into {} bits".format(size))
        if len(val) != size // 8:
            raise ValueError("bytes {} doesn't fit into {} bytes".format(repr(val), size // 8))
    return b''.join(bytes(format(i,'08b'), 'ascii') for i in val)

def encode_ip(val):
    return b''.join(bytes(format(int(i),'08b'), 'ascii') for i in val.split('.'))

def encode(val, size, type_=None):
    if type_ == 'ip':
        assert not size or size == 4*8, "ip has to be encoded into 4 bytes"
        return encode_ip(val)
    if isinstance(val, str):
        return encode_str(val, size)
    elif isinstance(val, bytes) or isinstance(val, bytearray):
        return encode_bytes(val, size)
    elif isinstance(val, int):
        return encode_num(val, size, type_)
    elif isinstance(val, float) and val % 1 == 0:
        return encode_num(int(val), size)
    elif isinstance(val, Repr):
        v = str.encode(val._value('bin'))
        if size is not None and size != len(v):
            raise ValueError("size do not match value")
        return v
    else:
        raise ValueError("cannot encode type {}".format(type(val)))

class Repr:
    def __init__(self, binary_string, type_):
        self.binary_string = binary_string
        self.type_ = type_

    def _guess_type(self, type_=None):
        type_ = type_ or self.type_
        if type_ == '' or type_ is None:
            if len(self.binary_string) % 8 != 0:
                type_ = 'bin'
            else:
                if len(self.binary_string) in (1, 2, 4, 8):
                    type_ = 'int'
                else:
                    type_ = 'hex'
        return type_
        
    
    def value(self, type_=None):
        return self
    
    def _value(self, type_=None):
        binary_string = self.binary_string
        type_ = self._guess_type(type_)
        if type_ == 'bytes':
            return bytes(int(binary_string[i:i+8], 2) for i in range(0, len(binary_string), 8))
        if type_ == 'str':
            return bytes(int(binary_string[i:i+8], 2) for i in range(0, len(binary_string), 8)).decode("utf-8")
        elif type_ == 'bin':
            return binary_string
        elif type_ == 'hex':
            return ''.join('{0:02x}'.format(int(binary_string[i:i+8], 2)) for i in range(0, len(binary_string), 8))
        elif type_.startswith('int'):
            return int(binary_string, 2)
        elif type_ == 'ip':
            return '.'.join('{}'.format(int(binary_string[i:i+8], 2)) for i in range(0, len(binary_string), 8))
        else:
            raise ValueError(f"Unknown type {type_}")

    def __repr__(self):
        binary_string = self.binary_string
        type_ = self._guess_type()
        val = self._value()
        if type_ in ('bytes', 'str', 'ip') or type_.startswith('int'):
            return repr(val)
        elif type_ == 'bin':
            return '0b' + val
        elif type_ == 'hex':
            return '0x' + val
        else:
            raise ValueError(f"Unknown type {type_}")

    @staticmethod
    def __get_eq_type(other):
        other_type = None
        if isinstance(other, int):
            other_type = 'int'
        elif isinstance(other, str):
            other_type = 'str'
        elif isinstance(other, bytearray) or isinstance(other, bytes):
            other_type = 'bytes'
        else:
            raise ValueError("type not defined")
        return other_type

    def __eq__(self, other):
        other_type = Repr.__get_eq_type(other)
        return self._value(other_type) == other
    
    def __add__(self, other):
        if isinstance(other, Repr):
            if self.type_ is None or other.type_ is None or self.type_ == other.type_:
                return Repr(self.binary_string + other.binary_string, type_=self.type_ or other.type_)
            raise RuntimeError("unable to add reprs with differnet types")
        else:
            return self._value(Repr.__get_eq_type(other)) + other
    
    def __mul__(self, other):
        return self._value(Repr.__get_eq_type(other)) * other
    
    def split(self, *args, **kwargs):
        return self._value().split(*args, **kwargs)
    
    def __iter__(self):
        for c in self._value('bytes'):
            yield c

    def __len__(self):
        if len(self.binary_string) % 8 != 0:
            raise ValueError("length not defind for partial bytes")
        return len(self.binary_string) // 8
    
    def __getitem__(self, item):
        return self._value('bytes')[item]

def decode(binary_string, type_):
    return Repr(binary_string, type_)

def get_mapping(mapping, k=None, v=None):
    try:
        n = next(i for i in mapping if (k is None or k == i[0]) and (v is None or v == i[1]))
        return n[1] if k is not None else n[0]
    except StopIteration:
        return k if k is not None else v

def evaluate(tree, context, **kwargs):
    mappings = kwargs.get('mappings', {})
    
    if tree[0] == 'or':
        res1 = None
        res2 = None
        try:
            res1 = evaluate(tree[1], context, **kwargs)
        except Exception as e:
            pass
        try:
            res2 = evaluate(tree[2], context, **kwargs)
        except Exception as e:
            pass
        if res1 is None and res2 is None:
            raise ValueError("input doesn't not match any field in root")
        # elif res1 is not None and res2 is not None:
            # raise ValueError("input matches both fields in root")
        elif res1 is not None:
            return res1
        else:
            return res2
    elif tree[0] == 'concat':
        res2 = evaluate(tree[2], context, **kwargs)
        res1 = evaluate(tree[1], context, **kwargs)
        return res1 + res2
    elif tree[0] in ('field', 'len_field', 'grouped'):
        size_option, name, type_ = tree[1], tree[2], tree[3]
        value = None
        should_update_size = False
        size = None
        actual_size = None
        if size_option[0] == 'literal':
            size = size_option[1]
            if size_option[2] is not None:
                if name is not None:
                    if name in context and decode(encode(context[name], size), 'int').value() != size_option[2]:
                        raise ValueError("value {} does not match {}".format(
                            decode(encode(context[name], size), 'int').value(),
                            size_option[2],
                        ))
                    else:
                        context[name] = size_option[2]
                else:
                    value = size_option[2]
        elif size_option[0] == 'suffix':
            pass
        elif size_option[0] == 'rest':
            pass
        else:
            if size_option[1] in context:
                actual_size = context[size_option[1]] * 8
            else:
                should_update_size = True
        
        if name is None:
            inner_ctx = context
        elif context is None:
            inner_ctx = context
        else:
            inner_ctx = context.get(name, None)
        if name in mappings:
            inner_ctx = get_mapping(mappings[name], k=inner_ctx)
        if tree[0] == 'field':
            if value is None:
                value = inner_ctx
            size = size or actual_size
            res1 = encode(value, size, type_)
            if should_update_size:
                context[size_option[1]] = len(res1) / 8
            if size_option[0] == 'suffix':
                return res1 + encode(size_option[1].encode('utf-8').decode('unicode-escape').encode('utf-8'), None)
            return res1    
        elif tree[0] == 'len_field':
            value = inner_ctx
            res1 = encode(value, actual_size, type_)
            if should_update_size:
                context[size_option[1]] = len(res1) / 8
            if size_option[0] == 'literal':
                return encode(len(res1)/8, size) + res1
            elif size_option[0] == 'suffix':
                return res1 + encode(size_option[1].encode('utf-8').decode('unicode-escape').encode('utf-8'), None)
            else:
                return res1
        elif tree[0] == 'grouped':
            content = tree[4]
            if inner_ctx is not None:
                res1 = evaluate(tree[4], inner_ctx, **kwargs)
            else:
                res1 = b''
            if should_update_size:
                context[size_option[1]] = len(res1) / 8
            if size_option[0] == 'literal':
                return encode(len(res1)/8, size) + res1
            elif size_option[0] == 'suffix':
                return res1 + encode(size_option[1].encode('utf-8').decode('unicode-escape').encode('utf-8'), None)
            else:
                return res1
    elif tree[0] == 'repetition':
        repetition, content = tree[1], tree[2]
        assert content[0] in ('field', 'grouped', 'len_field'), "unable to handle repetition"
        name = content[2]
        if isinstance(context, list):
            val = context
        else:
            if name is None:
                if content[0] == 'grouped' and content[4][0] in ('field', 'grouped', 'len_field'):
                    name = content[4][2]
                else:
                    raise ValueError("unable to get name of repetition")
            if name in context:
                val = context[name]
            else:
                if repetition is not None and repetition in context and context[repetition] == 0:
                    val = []
                else:
                    raise RuntimeError("unable to get stuff")
        if repetition is not None and repetition in context:
            if context[repetition] != len(val):
                raise ValueError("repetition doesn't equal input list")
        elif isinstance(context, dict):
            context[repetition] = len(val)
        if name is not None:
            return b''.join(evaluate(content, {name: val[i]}, **kwargs) for i in range(len(val)))
        else:
            return b''.join(evaluate(content, val[i], **kwargs) for i in range(len(val)))
    else:
        raise RuntimeError(f"Unknown tree node {tree[0]}")

def bin_index(data, substr):
    for i in range(0, len(data), 8):
        if data[i:i+len(substr)] == substr:
            return i
    raise ValueError("substring not found")

def deevaluate(tree, data, curr={}, **kwargs):
    orig_data = data
    context = {}
    mappings = kwargs.get('mappings', {})
    if tree[0] == 'or':
        res1 = None
        res2 = None
        try:
            res1, data1 = deevaluate(tree[1], data, curr=context, **kwargs)
        except Exception as e:
            pass
        try:
            res2, data2 = deevaluate(tree[2], data, curr=context, **kwargs)
        except Exception as e:
            pass

        if res1 is None and res2 is None:
            raise ValueError("input doesn't not match any field in root")
        # elif res1 is not None and res2 is not None:
            # raise ValueError("input matches both fields in root")
        elif res1 is not None:
            context.update(res1)
            return context, data1
        else:
            context.update(res2)
            return context, data2
    elif tree[0] == 'concat':
        res, data = deevaluate(tree[1], data, curr=context, **kwargs)
        context.update(res)
        res, data = deevaluate(tree[2], data, curr=context, **kwargs)
        context.update(res)
        return context, data
    elif tree[0] in ('field', 'len_field', 'grouped'):
        size_option, name, type_ = tree[1], tree[2], tree[3]
        if size_option[0] == 'literal':
            if tree[0] == 'field':
                size = size_option[1]
            else:
                size_size = size_option[1] 
                size = decode(data[:size_size], 'int').value() * 8
                data = data[size_size:] 
            res = decode(data[:size], type_)
            data = data[size:]
            if size_option[2] is not None:
                if res.value('int') != size_option[2]:
                    raise ValueError("value {} does not match {}".format(
                        res.value('int'),
                        size_option[2],
                    ))
        elif size_option[0] == 'suffix':
            suffix = encode(size_option[1].encode('utf-8').decode('unicode-escape').encode('utf-8'), None).decode('ascii')
            size = bin_index(data, suffix)
            res = decode(data[:size], type_)
            data = data[size+len(suffix):]
        elif size_option[0] == 'rest':
            res = decode(data, type_)
            data = data[len(data):]
        else:
            size = curr[size_option[1]].value(type_ = 'int') * 8
            if not kwargs.get('verbose', False):
                del curr[size_option[1]]
            res = decode(data[:size], type_)
            data = data[size:]
        if tree[0] == 'field':
            if name is None and len(size_option) >= 3:
                pass
            else:
                if name in mappings:
                    res = get_mapping(mappings[name], v=res)
                context[name] = res
            return context, data
        elif tree[0] == 'len_field':
            context[name] = res
            return context, data
        elif tree[0] == 'grouped':
            assert type_ is None, "cannot have type of a group"
            if len(res.binary_string) == 0:
                inner_context = None
            else:
                inner_context, remaining = deevaluate(tree[4], res.binary_string, curr=context, **kwargs)
                if size_option[0] != 'rest':
                    assert not remaining, "remaining when evaluating"
            if (len(res.binary_string) == 0 and kwargs.get('verbose', False)) or (len(res.binary_string) != 0):
                if name is None:
                    context.update(inner_context)
                else:
                    context[name] = inner_context
            if size_option[0] == 'rest':
                return context, remaining
            return context, data
    elif tree[0] == 'repetition':
        repetition, content = tree[1], tree[2]
        last_data = data
        count = None
        if repetition in curr:
            count = curr[repetition].value('int')
            if not kwargs.get('verbose', False):
                del curr[repetition]
        while True:
            if data == '':
                break
            elif count is not None and 0 == count:
                break
            res, data = deevaluate(content, data, curr=context, **kwargs)
            keys = list(res.keys())
            if len(keys) != 1:
                raise ValueError("more than one key exists in repetition")
            key = keys[0]
            if key not in context:
                context[key] = []
            context[key].append(res[key])
            if count is not None and len(context[key]) == count:
                break
            if data == last_data:
                raise ValueError("infinite loop decoding a list")
        if isinstance(context, dict) and list(context.keys()) == [None]:
            context = context[None]
        return context, data
    else:
        raise RuntimeError(f"Unknown tree node {tree[0]}")


def construct(expression, context, **kwargs):
    parse_tree = parser.parse(expression)
    binary_string = evaluate(parse_tree, context, **kwargs)
    byte_data = bytes(int(binary_string[i:i+8], 2) for i in range(0, len(binary_string), 8))
    return byte_data

def match(expression, data, **kwargs):
    parse_tree = parser.parse(expression)
    binary_string = ''.join('{v:08b}'.format(v=v) for v in data)
    res, rem = deevaluate(parse_tree, binary_string, **kwargs)
    if len(rem) % 8 != 0:
        raise ValueError("remaining has to be a multiple of bytes")
    return res, bytes(int(rem[i:i+8], 2) for i in range(0, len(rem), 8))

def match_exact(expression, data, **kwargs):
    res, rem = match(expression, data, **kwargs)
    if len(rem) != 0:
        raise ValueError("match exact returned data")
    return res

# In[12]:


def tests1():
    expect(
        "binary data",
        "1@a-2@b[int]-3@c[int]-4@d[int]-6@e[int]",
        "{'a': 0b1, 'b': 3, 'c': 6, 'd': 14, 'e': 60}",
        bytes.fromhex(hex(0b1111101110111100)[2:]),
        should_raise=True,
    )
    expect(
        "binary data reversed",
        "6@e[int]-4@d[int]-3@c[int]-2@b[int]-1@a",
        "{'e': 60, 'd': 14, 'c': 6, 'b': 3, 'a': 0b1}",
        bytes.fromhex(hex(0b1111001110110111)[2:]),
    )
    expect(
        "binary data typed",
        "1@a[bin]-2@b[bin]-3@c[bin]-4@d[hex]-6@e[int]",
        "{'a': 0b1, 'b': 0b11, 'c': 0b110, 'd': 0x0e, 'e': 60}",
        bytes.fromhex(hex(0b1111101110111100)[2:]),
    )
    expect(
        "input int doesn't fit",
        "1@a[bin]-2@b[bin]-3@c[bin]-4@d[hex]-6@e[int]",
        '{"a": 0b1, "b": 0b11, "c": 0b110, "d": 0xe, "e": 250}',
        ValueError("number 250 doesn't fit into 6 bits"),
    )
    expect(
        "multiple bytes string",
        "2B@f[str]-3B@g[str]",
        "{'f': 'ma', 'g': 'ple'}",
        b"maple",
    )
    expect(
        "multiple bytes hex",
        "2B@f[str]-3B@g[hex]",
        "{'f': 'ma', 'g': 0x706c65}",
        b"maple",
    )
    expect(
        "multiple bytes binary",
        "2B@f[str]-3B@g[bytes]",
        r"{'f': 'ma', 'g': b'\x12\n!'}",
        b"ma\x12\n!",
    )
    expect(
        "multiple bytes binary",
        "2B@f[str]-3B@g[bytes]",
        r"{'f': 'ma', 'g': b'\x12\n!'}",
        b"ma\x12\n!",
    )
    expect(
        "multiple bytes string, bigger input",
        "2B@f[str]-3B@g[str]",
        '{"f": "ma", "g": "plea"}',
        ValueError("string 'plea' doesn't fit into 3 bytes"),
    )
    expect(
        "multiple bytes string, smaller input",
        "2B@f[str]-3B@g[str]",
        '{"f": "ma", "g": "pl"}',
        ValueError("string 'pl' doesn't fit into 3 bytes"),
    )
    
    expect(
        "variable input string",
        "2B@f[str]-1B@h[str]()-3B@g[str]",
        "{'f': 'ma', 'h': 'rrrr', 'g': 'ple'}",
        b"ma\4rrrrple",
        should_raise=True,
    )
    expect(
        "variable input string, multi byte length",
        "2B@f[str]-2B@h[str]()-3B@g[str]",
        "{'f': 'ma', 'h': 'rrrr', 'g': 'ple'}",
        b"ma\0\x04rrrrple",
    )
    
    
    expect(
        "variable input string, reference length",
        "2B@str_len-2B@f[str]-{str_len}@h[str]()-3B@g[str]",
        "{'f': 'ma', 'h': 'rrrr', 'g': 'ple'}",
        b"\0\x04marrrrple",
        should_raise=True
    )
    expect(
        "fail with invalid reference str length",
        "2B@str_len-2B@f[str]-{str_len}@h[str]()-3B@g[str]",
        '{"str_len": 5, "f": "ma", "h": "rrrr", "g": "ple"}',
        ValueError("string 'rrrr' doesn't fit into 5 bytes"),
    )
    expect(
        "variable input string, reference length, typed",
        "2B@str_len[int]-2B@f[str]-{str_len}@h[str]()-3B@g[str]",
        "{'str_len': 4, 'f': 'ma', 'h': 'rrrr', 'g': 'ple'}",
        b"\0\x04marrrrple",
        verbose=True,
        should_raise=True,
    )
    
    
    expect(
        "variable input string, null terminated",
        r"2B@f[str]-~\0@h[str]()-3B@g[str]",
        "{'f': 'ma', 'h': 'rrrr', 'g': 'ple'}",
        b"marrrr\0ple",
    )
    expect(
        "variable input string, \\r\\n terminated",
        r"2B@f[str]-~\r\n@h[str]()-3B@g[str]",
        "{'f': 'ma', 'h': 'rrrr', 'g': 'ple'}",
        b"marrrr\r\nple",
    )
    expect(
        "nested",
        r"2B@f[str]-1B@h(2B@i[str]-3B@j[hex])-3B@g[str]",
        "{'f': 'ma', 'h': {'i': 'rr', 'j': 0x120a21}, 'g': 'ple'}",
        b"ma\x05rr\x12\n!ple",
    )
    expect(
        "nested, null terminated",
        r"2B@f[str]-~\0@h(2B@i[str]-3B@j[hex])-3B@g[str]",
        "{'f': 'ma', 'h': {'i': 'rr', 'j': 0x120a21}, 'g': 'ple'}",
        b"marr\x12\n!\0ple",
    )
    
    
    # expect(
    #     "repetition",
    #     r"2B@f[str]-...~\0@h(2B@i[str]-3b@j[hex])-3B@g[str]",
    #     '{"f": "ma", "h": [{"i": "rr", "j": 0x120a21}, {"i": "aq", "j": 0x111111}] , "g": "ple"}',
    #     b"marr\x12\n!\0aq\x11\x11\x11\0ple",
    #     should_raise=True,
    # )
    expect(
        "repetition, nested",
        r"2B@f[str]-~\0@h(2B@i[str]-...3b@j[hex])-3B@g[str]",
        "{'f': 'ma', 'h': {'i': 'rr', 'j': [0x120a21,  0x111111]}, 'g': 'ple'}",
        b"marr\x12\n!\x11\x11\x11\0ple",
    )
    expect(
        "repetition count",
        r"1B@k-2B@f[str]-{k}...~\0@h(2B@i[str]-3b@j[hex])-3B@g[str]",
        "{'f': 'ma', 'h': [{'i': 'rr', 'j': 0x120a21}, {'i': 'aq', 'j': 0x111111}] , 'g': 'ple'}",
        b"\x02marr\x12\n!\0aq\x11\x11\x11\0ple",
    )
    expect(
        "repetition count, verbose",
        r"1B@k-2B@f[str]-{k}...~\0@h(2B@i[str]-3b@j[hex])-3B@g[str]",
        "{'k': 2, 'f': 'ma', 'h': [{'i': 'rr', 'j': 0x120a21}, {'i': 'aq', 'j': 0x111111}] , 'g': 'ple'}",
        b"\x02marr\x12\n!\0aq\x11\x11\x11\0ple",
        verbose=True,
    )
    # expect(
    #     "repetition, nested, not at the end",
    #     r"2B@f[str]-~\0@h(...2B@i[str]-3b@j[hex])-3B@g[str]",
    #     "{'f': 'ma', 'h': {'i': ['rr', 'qa'], 'j': 0x120a21}, 'g': 'ple'}",
    #     b"marrqa\x12\n!\0ple",
    # )
    
    
    expect(
        "constant",
        r"2B@f[str]-1B==8-3B@g[str]",
        "{'f': 'ma', 'g': 'ple'}",
        b"ma\x08ple",
    )
    expect(
        "constant hex",
        r"2B@f[str]-1B==0x8-3B@g[str]",
        "{'f': 'ma', 'g': 'ple'}",
        b"ma\x08ple",
    )
    expect(
        "constant bin",
        r"2B@f[str]-1B==0b1000-3B@g[str]",
        "{'f': 'ma', 'g': 'ple'}",
        b"ma\x08ple",
    )
    expect(
        "constant hex, named",
        r"2B@f[str]-1B==0x8@m[bytes]-3B@g[str]",
        r"{'f': 'ma', 'm': b'\x08', 'g': 'ple'}",
        b"ma\x08ple",
    )
    expect(
        "constant hex, named, invalid, construct",
        r"2B@f[str]-1B==0x8@m[bin]-3B@g[str]",
        r"{'f': 'ma', 'm': b'\x09', 'g': 'ple'}",
        ValueError("value 9 does not match 8"),
    )
    expect(
        "constant hex, named, invalid, match",
        r"2B@f[str]-1B==0x8@m[bin]-3B@g[str]",
        ValueError("value 0b00001001 does not match 8"),
        b"ma\x09ple",
    )
    
    
    expect(
        "leading dash",
        r"-2B@f[str]-3B@g[str]",
        "{'f': 'ma', 'g': 'ple'}",
        b"maple",
    )
    expect(
        "leading dash, nested",
        r"2B@f[str]-1B@h(-2B@i[str]-3b@j[hex])-3B@g[str]",
        "{'f': 'ma', 'h': {'i': 'rr', 'j': 0x120a21}, 'g': 'ple'}",
        b"ma\x05rr\x12\n!ple",
    )
    
    
    expect(
        "ip",
        r"2B@f[str]-32@h[ip]-3B@g[str]",
        "{'f': 'ma', 'h': '172.217.23.196', 'g': 'ple'}",
        b"ma\xac\xd9\x17\xc4ple",
        should_raise=True,
    )
    
    expect(
        "ip, length",
        r"2B@f[str]-16@h[ip]()-3B@g[str]",
        "{'f': 'ma', 'h': '172.217.23.196', 'g': 'ple'}",
        b"ma\0\x04\xac\xd9\x17\xc4ple",
    )
    
    expect(
        "mapping",
        r"2B@f[str]-1B@h-3B@g[str]",
        "{'f': 'ma', 'h': 'B' , 'g': 'ple'}",
        b"ma\x02ple",
        mappings={'h': [('A', 1), ('B', 2)]},
    )
    expect(
        "mapping, missing",
        r"2B@f[str]-1B@h-3B@g[str]",
        "{'f': 'ma', 'h': 3 , 'g': 'ple'}",
        b"ma\x03ple",
        mappings={'h': [('A', 1), ('B', 2)]}
    )
    
    
    # expect(
    #     "nested, reference length",
    #     r"2B@f[str]-{len}@h(2B@i[str]-1B@len-3b@j[hex])-3B@g[str]",
    #     '{"f": "ma", "h": {"i": "rr", "j": 0x120a21}, "g": "ple"}',
    #     b"marr\x05\x12\n!ple",
    #     should_raise=True
    # )
    # expect(
    #     "nested, reference length, verbose",
    #     r"2B@f[str]-{len}@h(2B@i[str]-1B@len-3b@j[hex])-3B@g[str]",
    #     '{"f": "ma", "h": {"i": "rr", "len": 5, "j": 0x120a21}, "g": "ple"}',
    #     b"marr\x05\x12\n!ple",
    #     verbose=True,
    # )
    
    expect(
        "take the rest",
        r"2B@f[str]-@g[str]",
        "{'f': 'ma', 'g': 'ple'}",
        b"maple",
    )
    expect(
        "take the rest, with nested",
        r"2B@f[str]-@n(3B@a[str]-1B@b[str])",
        "{'f': 'ma', 'n': {'a': 'mie', 'b': 'e'}}",
        b"mamiee",
    )
    # expect(
    #     "take the rest, with nested",
    #     r"2B@f[str]-@n(3B@a[str]-1B@b[str])-@g[str]",
    #     '{"f": "ma", "n": {"a": "mie", "b": "e"}, "g": "ple"}',
    #     b"mamieeple",
    #     should_raise=True
    # )
    expect(
        "take the rest, with nested, 2",
        r"2B@f[str]-1B@n(3B@a[str]-@b[str])-@g[str]",
        "{'f': 'ma', 'n': {'a': 'mie', 'b': 'e'}, 'g': 'ple'}",
        b"ma\x04mieeple",
    )
    expect(
        "unnammed nested",
        r"2B@f[str]-1B(3B@a[str]-@b[str])-@g[str]",
        "{'f': 'ma', 'a': 'mie', 'b': 'e', 'g': 'ple'}",
        b"ma\x04mieeple",
        should_raise=True,
    )
    
    
    expect(
        "or, match first, unnamed",
        r"8==1-2B@f[str]|8==2-2B@g[str]",
        "{'f': 'ma'}",
        b"\x01ma",
    )
    expect(
        "or, match second, unnamed",
        r"8==1-2B@f[str]|8==2-1B@g[str]",
        "{'g': 'a'}",
        b"\x02a",
    )
    expect(
        "or, match both, unnamed",
        r"8==1-2B@f[str]|8==2-2B@f[str]",
        "{'f': 'ma'}",
        b'\x01ma',
    )
    expect(
        "or, nammed",
        r"8==1@selector-2B@f[str]|8==2@selector-2B@f[str]",
        "{'selector': 1, 'f': 'ma'}",
        b"\x01ma",
    )
    expect(
        "or, nammed, no match",
        r"8==1@selector-2B@f[str]|8==2@selector-2B@f[str]",
        ValueError("input doesn't not match any field in root"),
        b"\x03a",
    )

# In[7]:


def tests2():
    expect(
        "dns question",
        r"""
    16@id[hex]
    -1@qr-4@opcode[int]-1@aa-1@tc-1@rd-1@ra-3==0-4@rcode[int]
    -16@qcount[int]
    -16@ancount[int]
    -16@nscount[int]
    -16@arcount[int]
    -{qcount}...@questions(
        ~\0@question(
            ...8@parts[str]()
        )
        -16@type
        -16@class
    )
    -{ancount}...@answers(
        (
            2==0b11
            -14@pointer
            |
            ~\0@question(
                ...8@parts()
            )
        )
        -16==1@type
        -16@class
        -32@ttl
        -16@answer[ip]()
    )
    """,
        """{
            'id': 0x911f,
            'qr': 0b0,
            'opcode': 0,
            'aa': 0b0,
            'tc': 0b0,
            'rd': 0b1,
            'ra': 0b0,
            'rcode': 0,
            'qcount': 1,
            'ancount': 0,
            'nscount': 0,
            'arcount': 0,
            'questions': [{
                'question': {
                    'parts': [
                        'www', 'google', 'com'
                    ]
                },
                'type': 'A',
                'class': 'IN'
            }]
        }""",
        b'\x91\x1f\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x03www\x06google\x03com\x00\x00\x01\x00\x01',
        mappings={
            'type': [('A', 1), ('CNAME', 5)],
            'class': [('IN', 1)],
        },
        verbose=True
    )


# In[8]:


def tests3():
    expect(
        "or, no match",
        r"8==1@selector-2B@f[str]|8==2@selector-2B@f[str]",
        "{'selector': 3, 'f': 'ma'}",
        ValueError("input doesn't not match any field in root"),
    )
    expect(
        "or, match both, decode",
        r"8==2@selector-2B@f[str]|8==2@selector-2B@f[str]",
        r"{'selector': 2, 'f': 'ma'}",
        b"\x02ma",
    )
    
    expect(
        "field suffix",
        r"~\0@a[str]-2B@f[str]",
        "{'a': 'mie', 'f': 'ma'}",
        b'mie\0ma',
    )
    
    
    expect(
        "field reference size, exists",
        r"2B@l[int]-{l}@a[str]-2B@f[str]",
        "{'l': 3, 'a': 'mie', 'f': 'ma'}",
        b'\0\x03miema',
        verbose=True,
    )
    
    expect(
        "field reference size, does not exists",
        r"2B@l[int]-{l}@a[str]-2B@f[str]",
        "{'a': 'mie', 'f': 'ma'}",
        b'\0\x03miema',
    )
    
    expect(
        "len_field, rest",
        r"~\0@a[str]-@f[str]()",
        "{'a': 'mie', 'f': 'ma'}",
        b'mie\0ma',
        should_raise=True,
    )
    
    expect(
        "group reference size, exists",
        r"2B@l[int]-{l}@a(~\0@i[str])-2B@f[str]",
        "{'l': 4, 'a': {'i': 'mie'}, 'f': 'ma'}",
        b'\0\x04mie\0ma',
        verbose=True,
    )
    
    expect(
        "group reference size, does not exists",
        r"2B@l[int]-{l}@a(~\0@i[str])-2B@f[str]",
        "{'a': {'i': 'mie'}, 'f': 'ma'}",
        b'\0\x04mie\0ma',
    )
    
    expect(
        "repetition count, verbose, wrong",
        r"1B@k-2B@f[str]-{k}...~\0@h(2B@i[str]-3b@j[hex])-3B@g[str]",
        "{'k': 3, 'f': 'ma', 'h': [{'i': 'rr', 'j': 0x120a21}, {'i': 'aq', 'j': 0x111111}] , 'g': 'ple'}",
        ValueError("repetition doesn't equal input list"),
        verbose=True,
    )
    
    expect(
        "repetition count, verbose, wrong",
        r"1B@k-2B@f[str]-{k}...~\0@h(2B@i[str]-3b@j[hex])-3B@g[str]",
        "{'k': 3, 'f': 'ma', 'h': [{'i': 'rr', 'j': 0x120a21}, {'i': 'aq', 'j': 0x111111}] , 'g': 'ple'}",
        ValueError("repetition doesn't equal input list"),
        verbose=True,
    )
    
    # expect(
    #     "repetition count, verbose, wrong",
    #     r"2b@k[int]-{k}...@m(~\0@n[str])",
    #     "{'k': 3, 'm': [{'n': 'm'}, {'n': 'ma'}, {'n': 'mar'}]}",
    #     b'\0\x03m\0ma\0mar\0',
    #     verbose=True,
    # )
    
    
    expect(
        "repetition count, take the rest",
        r"2b@k[int]-{k}...~\0@m(@n[str])",
        "{'k': 3, 'm': [{'n': 'm'}, {'n': 'ma'}, {'n': 'mar'}]}",
        b'\0\x03m\0ma\0mar\0',
        verbose=True,
    )

    expect(
        "len field, literal size",
        r"2b@a[int16]()",
        "{'a': 3}",
        b'\0\x02\0\x03',
        # verbose=True,
        should_raise=True,
    )

    expect(
        "repetition, zero size",
        r"2b@a[int]-{a}...(2b@b[int])-2b@c[int]",
        "{'a': 0, 'c': 2}",
        b'\0\0\0\x02',
        verbose=True,
    )
    expect(
        "repetition, more than one key, encode",
        r"...(2b@b[int]-2b@c[int])",
        ValueError('more than one key exists in repetition'),
        b'\0\0\0\x02',
        verbose=True,
    )
    expect(
        "repetition, more than one key, decode",
        r"...(2b@b[int]-2b@c[int])",
        b"{'b': 0, 'c': 2}",
        ValueError('unable to get name of repetition'),
        verbose=True,
    )

    expect(
        "repetition, more than one key, decode",
        r"...(2b@c[int])",
        "{'c': [2]}",
        b'\0\x02',
        verbose=True,
        should_raise=True,
    )
    expect(
        "repetition, infinite-loop",
        r"...(0@b[int])-2b@c[int]",
        ValueError('infinite loop decoding a list'),
        b'\0\x02',
        verbose=True,
    )
    expect(
        "repetition, invalid remaining",
        r"6@a[bin]",
        ValueError('remaining has to be a multiple of bytes'),
        b'\0\x02',
        verbose=True,
    )
    expect(
        "substring not found",
        r"~\0@a[str]",
        ValueError("substring not found"),
        b'asd',
        verbose=True,
    )
    expect(
        "repetition, unnamed",
        r"...8[str]()",
        "['asd', 'eaf']",
        b'\x03asd\x03eaf',
        verbose=True,
    )
    expect(
        "repetition, groupped",
        r"...8@a(1b@b[int]-@c[str])",
        "{'a': [{'b': 1, 'c': 'mie'}, {'b': 2, 'c': 'm'}]}",
        b'\x04\x01mie\x02\x02m',
        verbose=True,
        should_raise=True,
    )
    expect(
        "grouped, non empty",
        r"8@a(8@c[int])-8@b[int]",
        "{'a': {'c': 3}, 'b': 2}",
        b'\x01\x03\x02',
        verbose=True,
        should_raise=True,
    )
    expect(
        "grouped, empty",
        r"8@a(8@c[int])-8@b[int]",
        "{'b': 2}",
        b'\0\x02',
    )
    expect(
        "grouped, empty, verbose",
        r"8@a(8@c[int])-8@b[int]",
        "{'a': None, 'b': 2}",
        b'\0\x02',
        verbose=True,
        should_raise=True,
    )

    expect(
        "newline",
        r"""8@a[int]-
        -8@b[int]""",
        "{'a': 2, 'b': 2}",
        b'\x02\x02',
        verbose=True,
        should_raise=True,
    )
    expect(
        "newline, comment at end of line",
        r"""8@a[int] # this is a comment
        -8@b[int]""",
        "{'a': 2, 'b': 2}",
        b'\x02\x02',
        verbose=True,
        should_raise=True,
    )
    expect(
        "newline, comment on line",
        r"""8@a[int]
        # this is a comment
        -8@b[int]""",
        "{'a': 2, 'b': 2}",
        b'\x02\x02',
        verbose=True,
        should_raise=True,
    )
    expect(
        "groups, partial bytes",
        r"1==1@a[int]-(2@b[int]-(5@c[int]))",
        "{'a': 1, 'b': 2, 'c': 5}",
        bytes.fromhex(hex(0b11000101)[2:]),
        should_raise=True,
    )

def tests():
    tests1()
    tests2()
    tests3()

if __name__ == "__main__":
    tests()
    if failed_any:
        print("\x1b[37m\x1b[41msome tests failed\x1b[0m")
        exit(1)
    else:
        print("\x1b[37m\x1b[42mall tests succeeded\x1b[0m")
