from __future__ import print_function

from math import gcd
import numpy as np

_char2id = {chr(i + ord('a')): i for i in range(26)}
_id2char = {i: chr(i + ord('a')) for i in range(26)}

char2id = lambda x: _char2id[x.lower()]
id2char = lambda x: _id2char[x]


def char_freq_map():
    """
    英语26个字母的词频map
    """
    s = """
    1 E 12.25 2 T 9.41 3 A 8.19 4 O 7.26 5 I 7.10 6 N 7.06 7 R 6.85 8 S 6.36 9 H 4.57 10 D 3.91 11 C 3.8312 12 L 3.77
    13 M 3.34 14 P 2.89 15 U 2.58 16 F 2.2617 17 G 1.71 18 W 1.59 19 Y 1.58 20 B 1.47
    21 K 0.4122 22 J 0.14 23 V 1.09 24 X 0.2125 25 Q 0.09
    26 Z 0.08
    """
    res = {}
    l = s.strip().split()
    for i in range(0, len(l) - 1, 3):
        res[l[i + 1]] = float(l[i + 2])
    return res


CHAR_FREQ_MAP = char_freq_map()


def show_dict(d):
    """
    d: dict
    return None
    print the dict with value desc order
    """
    for k, v in sorted(d.items(), key=lambda item: -item[1]):
        print(k, v)


def _format_cipher(cipher_text):
    """
    cipher: string
    return string
    format string, delete space and \n
    """
    return cipher_text.replace(" ", "").replace("\n", "").lower()


def move(target_string, move_size=2):
    """
    移位变换
    """
    target_string = _format_cipher(target_string)
    res = []
    for c in target_string:
        _id = (char2id(c) - move_size) % 26
        _c = id2char(_id)
        res.append(_c)
    print("".join(res))


def move_search(cipher_text):
    """
    移位密码,搜索解密
    """
    for m in range(0, 26):
        print(f"move: {m}, decrypt: ", end=" ")
        move(cipher_text, m)


def virgina_encrypt(plain_text, key):
    """
    弗吉尼亚加密 分组加密， key 是一个短字符串
    plain_text 先分组，组内每个字母做移位，移位位数对照key
    """
    get_move = lambda x: char2id(key[x % (len(key))])
    ve_res = []
    for i in range(len(plain_text)):
        _id = (char2id(plain_text[i]) + get_move(i)) % 26
        ve_res.append(id2char(_id))
    return "".join(ve_res)


class MyError(ValueError):
    pass


# 返回一个整数的模逆元素
def mod_rev(num, mod=26):
    if num == 0 or gcd(num, mod) > 1:
        raise MyError('modular inversion does not exists.')
    else:
        i = 1
        while i * num % mod != 1:
            i = i + 1
        return i


def mod_rev_v2(num, mod=26):
    if num == 0 or gcd(num, mod) > 1:
        print('modular inversion does not exists.')
        return None
    r_arr = [mod, num]
    q_arr = []
    while 1:
        m = r_arr[-2]
        n = r_arr[-1]
        r, q = m % n, m // n
        if r == 0: break
        r_arr.append(r)
        q_arr.append(q)
    t_prev, t_curr = 0, 1
    for q in q_arr:
        t_prev, t_curr = t_curr, (t_prev - q * t_curr)
    return t_curr % mod


def disp(mat):
    """
    display a matrix
    """
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            print(mat[i][j], end='\t')
        print('')
    print('')


def matrix_rev(A, M=26):
    from copy import deepcopy
    A = deepcopy(A)

    # 检查矩阵格式是否合法以及是否可逆
    def _check_matrix(A, M=26):
        if (not isinstance(A, list)) or (not isinstance(A[0], list)) or (not isinstance(A[0][0], int)):
            raise MyError('Invalid matrix format.')
        mat = np.array(A)
        D = int(np.linalg.det(A)) % M
        if gcd(D, M) > 1:
            print('Valid Matrix.')
            raise MyError('This matrix does not have a modular inversion matrix.')

    # 矩阵的第一类初等变换：交换矩阵第i行与第j行
    def _swap_row(A, i, j):
        A[i], A[j] = A[j], A[i]

    # 矩阵的第二类初等变换：将矩阵第i行乘以n
    def _mul_row(A, i, n, M=26):
        a = A[i]
        A[i] = [a[x] * n % M for x in range(len(a))]

    # 矩阵的第三类初等变换：矩阵第i行减去n倍的j行
    def _sub_row(A, i, j, n, M=26):
        a = A[i]
        b = A[j]
        A[i] = [(a[x] - n * b[x]) % M for x in range(len(a))]

    # 找到符合要求的第i行
    def _find_row(A, i, M=26):
        start = i
        while A[start][i] == 0 or gcd(A[start][i], M) > 1:
            start = start + 1
        return start

    try:
        _check_matrix(A, M)
        dim = len(A)

        # concatenate with a unit matrix
        for i in range(dim):
            for j in range(dim):
                if j == i:
                    A[i].append(1)
                else:
                    A[i].append(0)
        # transform
        for i in range(dim):
            target_row = _find_row(A, i, M)
            _swap_row(A, i, target_row)
            n = mod_rev(A[i][i], M)
            _mul_row(A, i, n, M)
            for j in range(dim):
                if j != i:
                    _sub_row(A, j, i, A[j][i], M)
        # get result
        A_rev = [A[i][dim:] for i in range(dim)]
        return A_rev
    except Exception as e:
        print(e)


# 希尔加密
def encrypt_hill(plain_text, K):
    """
    希尔加密， 分组加密
    K 是一个方阵
    对原文分组做矩阵变换 C = P * K
    """
    ids_from_plain_text = [char2id(x) for x in plain_text]
    ids_of_cipher_text = []
    group_size = len(K[0])
    if len(plain_text) % group_size != 0:
        print("size of plain_text is not times of len(K[0])")
        raise MyError()
    for i in range(0, len(plain_text), group_size):
        tmp = np.matmul(ids_from_plain_text[i: i + group_size], K) % 26
        ids_of_cipher_text += list(tmp)
    return "".join([id2char(x) for x in ids_of_cipher_text])


hill_encrypt = encrypt_hill


def decrypt_hill(cipher_text, K):
    K_rev = matrix_rev(K)
    return hill_encrypt(cipher_text, K_rev)


hill_decrypt = decrypt_hill


def encrypt_affine(plain_text, a, b):
    """
    仿射加密，e(x) = ax+b
    其中 gcd(a,26)=1
    """
    if gcd(a, 26) != 1:
        print(f"ERROR, gcd({a}, 26) != 1")
        raise MyError()
    ids = [char2id(x) for x in plain_text]
    affine_res = [id2char((a * x + b) % 26) for x in ids]
    return "".join(affine_res)


def decrypt_affine(cipher_text, a, b):
    _a = mod_rev_v2(a)
    return encrypt_affine(cipher_text, _a, -_a * b)


# 密码分析
def _show_dict(d, size):
    if size < 0:
        size = len(d)
    for k, v in sorted(d.items(), key=lambda item: -item[1])[:size]:
        print(k, v)


def text_stat(text, mode='123', size=-1):
    from collections import defaultdict
    s = _format_cipher(text)
    if '1' in mode:
        one_stat = {chr(x): 0 for x in range(ord("a"), ord("a") + 26)}
        for x in s:
            if x in one_stat:
                one_stat[x] += 1
        print("single char stat:")
        _show_dict(one_stat, size)
        print("---" * 5)
    if '2' in mode:
        bigram_stat = defaultdict(int)
        for i in range(0, len(s) - 1):
            x = s[i:i + 2]
            bigram_stat[x] += 1
        print("bigram stat:")
        _show_dict(bigram_stat, size)
        print("---" * 5)
    if '3' in mode:
        trigram_stat = defaultdict(int)
        for i in range(0, len(s) - 2):
            x = s[i:i + 3]
            trigram_stat[x] += 1
        print("trigram stat:")
        _show_dict(trigram_stat, size)


def affine_parameter_solver(s='EcAb'):
    """
    仿射加密参数求解,输入两组原文到密文的pair(4个字母)
    输入四个字母,x1,x2,x3,x4
    x1->x2;x3->x4 （原文->密文）
    x1,x3 一般在 {E,T,A}中
    """
    if len(s) != 4:
        print("error")
        return
    ids = [char2id(c) for c in s]
    x1, x2, x3, x4 = ids
    for a in range(1, 26):
        if gcd(a, 26) != 1: continue
        if (x1 - x3) * a % 26 == (x2 - x4) % 26:
            b = (x2 - (x1 * a)) % 26
            return a, b
    print("无解")
    return None


def solver_congruence(a_arr, m_arr):
    """
    中国剩余定理 Chinese remainder theorem
    x≡a1 (mode m1)
    x≡a2 (mode m2)
    x≡a3 (mode m3)
    ...
    求解x
    x = crt([a1,a2,a3...],[m1,m2,m3...])
    from sympy.ntheory.modular import crt
    """
    from functools import reduce
    M = reduce(lambda a, b: a * b, m_arr, 1)
    M_arr = [M // x for x in m_arr]

    y_arr = [mod_rev_v2(x, y) for (x, y) in zip(M_arr, m_arr)]

    res = sum([x * y * z for (x, y, z) in zip(a_arr, M_arr, y_arr)]) % M
    return res, M


crt = solver_congruence


def mod_exp(base, k, md):
    """
    计算大幂指数 base的k次方mod md
    """
    res = 1
    while k:
        if k & 1: res = res * base % md
        base = base * base % md
        k = k >> 1
    return res


# jacobi
def jacobi(m, n):
    """
    ref1: http://math.fau.edu/richman/jacobi.htm
    ref1: https://zhuanlan.zhihu.com/p/26186689
    
    计算jacobi符号,n是一个奇正整数, (m,n) = 1
    计算 jicobi符号 (n/m)
    """

    def _Euler(m):
        """
        欧拉定理/ jacobi 二次互反律
        """
        if m % 4 == 1: return 1
        if m % 4 == 3: return -1

    def _judge(m):
        """
        jacobi性质2
        """
        if m % 8 == 1 or m % 8 == 7:
            return 1
        if m % 8 == 3 or m % 8 == 5:
            return -1

    res = 1
    while 1:
        # 性质1
        m = m % n
        # 性质3
        while m % 2 == 0:
            m = m / 2
            res *= _judge(n)
        # 欧拉推论
        if m == 1:
            break
        if m == -1:
            res *= _Euler(n)
            break
        m, n = n, m
        if (n - 1) * (m - 1) % 8 != 0:
            res *= -1
    #     print(res)
    return res


if __name__ == "__main__":
    print(f"jacobi(2, 5): {jacobi(2, 5)}")
    print(f"mod_rev_v2(5, 26): {mod_rev_v2(5, 26)}")
    print(f"mod_rev_v2(9, 26): {mod_rev_v2(9, 26)}")
    K = [[17, 21, 2],
         [17, 18, 2],
         [5, 21, 19]]
    print("矩阵K：")
    disp(K)
    print("矩阵K模26的逆：")
    K_inv = matrix_rev(K, 26)
    disp(K_inv)

    mk = 2579
    res = (mod_rev_v2(mod_exp(435, 765, mk), mk) * 2396) % mk
    print(f"2396*(435^765)^-1 mod 2579 = {res}")
    print("韩信带1500名兵士打仗，战死四五百人，站3人一排，多出2人；站5人一排，多出4人；站7人一排，多出3人。韩信很快说出人数：1004。")
    print(f"crt([2, 4, 3], [3, 5, 7] : {crt([2, 4, 3], [3, 5, 7])}")
    print(f"50+105*9 = {105 * 9 + 59}")
