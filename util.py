from __future__ import  print_function

from math import gcd  # 计算最大公约数
import numpy as np  # 计算行列式的值

char2id = {chr(i + ord('a')): i for i in range(26)}
id2char = {i: chr(i + ord('a')) for i in range(26)}

"""
英语26个字母的词频map
"""
s = """
1 E 12.25 2 T 9.41 3 A 8.19 4 O 7.26 5 I 7.10 6 N 7.06 7 R 6.85 8 S 6.36 9 H 4.57 10 D 3.91 11 C 3.8312 12 L 3.77
13 M 3.34 14 P 2.89 15 U 2.58 16 F 2.2617 17 G 1.71 18 W 1.59 19 Y 1.58 20 B 1.47
21 K 0.4122 22 J 0.14 23 V 1.09 24 X 0.2125 25 Q 0.09
26 Z 0.08
"""
CHAR_FREQ_MAP = {}
l = s.strip().split()
for i in range(0, len(l) - 1, 3):
    CHAR_FREQ_MAP[l[i + 1]] = float(l[i + 2])


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
    移位变幻
    """
    target_string = _format_cipher(target_string)
    res = []
    for c in target_string:
        _id = (char2id[c] - move_size) % 26
        _c = id2char[_id]
        res.append(_c)
    print("".join(res))


def move_search(cipher_text):
    """
    移位密码,搜索解密
    """
    for m in range(0, 26):
        print(f"move: {m}, decrypt: ", end=" ")
        move(cipher_text, m)


class MyError(ValueError):
    pass


# 检查矩阵格式是否合法以及是否可逆
def check_matrix(A, M=26):
    if (not isinstance(A, list)) or (not isinstance(A[0], list)) or (not isinstance(A[0][0], int)):
        raise MyError('Invalid matrix format.')
    mat = np.array(A)
    D = int(np.linalg.det(A)) % M
    if gcd(D, M) > 1:
        print('Valid Matrix.')
        raise MyError('This matrix does not have a modular inversion matrix.')


# 矩阵的第一类初等变换：交换矩阵第i行与第j行
def swap_row(A, i, j):
    A[i], A[j] = A[j], A[i]


# 矩阵的第二类初等变换：将矩阵第i行乘以n
def mul_row(A, i, n, M=26):
    a = A[i]
    A[i] = [a[x] * n % M for x in range(len(a))]


# 矩阵的第三类初等变换：矩阵第i行减去n倍的j行
def sub_row(A, i, j, n, M=26):
    a = A[i]
    b = A[j]
    A[i] = [(a[x] - n * b[x]) % M for x in range(len(a))]


# 找到符合要求的第i行
def find_row(A, i, M=26):
    start = i
    while A[start][i] == 0 or gcd(A[start][i], M) > 1:
        start = start + 1
    return start


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
    try:
        check_matrix(A, M)
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
            target_row = find_row(A, i, M)
            swap_row(A, i, target_row)
            n = mod_rev(A[i][i], M)
            mul_row(A, i, n, M)
            for j in range(dim):
                if j != i:
                    sub_row(A, j, i, A[j][i], M)
        # get result
        A_rev = [A[i][dim:] for i in range(dim)]
        return A_rev
    except Exception as e:
        print(e)


def solver_congruence(a_arr, m_arr):
    """
    中国剩余定理
    from sympy.ntheory.modular import crt
    """
    from functools import reduce
    M = reduce(lambda a, b: a * b, m_arr, 1)
    M_arr = [M // x for x in m_arr]

    y_arr = [mod_rev_v2(x, y) for (x, y) in zip(M_arr, m_arr)]

    res = sum([x * y * z for (x, y, z) in zip(a_arr, M_arr, y_arr)]) % M
    return res


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
def Euler(m):
    """
    欧拉定理/ jacobi 二次互反律
    """
    if m % 4 == 1: return 1
    if m % 4 == 3: return -1


def judge(m):
    """
    jacobi性质2
    """
    if m % 8 == 1 or m % 8 == 7:
        return 1
    if m % 8 == 3 or m % 8 == 5:
        return -1


def jacobi(m, n):
    """
    ref1: http://math.fau.edu/richman/jacobi.htm
    ref1: https://zhuanlan.zhihu.com/p/26186689
    
    计算jacobi符号,n是一个奇正整数, (m,n) = 1
    计算 jicobi符号 (n/m)
    """
    res = 1
    while 1:
        # 性质1
        m = m % n
        # 性质3
        while m % 2 == 0:
            m = m / 2
            res *= judge(n)
        # 欧拉推论
        if m == 1:
            break
        if m == -1:
            res *= Euler(n)
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
