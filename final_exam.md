

## 7 


```python
from util import _format_cipher, move_search
```


```python
cipher="""
FGOS VSQK AFLW JFWL GXLZ AFYK
AKSO AVWD QVAK UMKK WVLG HAUS
EGFY JWKW SJUZ WJK
"""

cipher = _format_cipher(cipher)
```


```python
move_search(cipher)
```

    move: 0, decrypt:  fgosvsqkaflwjfwlgxlzafykaksoavwdqvakumkkwvlghausegfyjwkwsjuzwjk
    move: 1, decrypt:  efnrurpjzekvievkfwkyzexjzjrnzuvcpuzjtljjvukfgztrdfexivjvrityvij
    move: 2, decrypt:  demqtqoiydjuhdujevjxydwiyiqmytubotyiskiiutjefysqcedwhuiuqhsxuhi
    move: 3, decrypt:  cdlpspnhxcitgctiduiwxcvhxhplxstansxhrjhhtsidexrpbdcvgthtpgrwtgh
    move: 4, decrypt:  bckoromgwbhsfbshcthvwbugwgokwrszmrwgqiggsrhcdwqoacbufsgsofqvsfg
    move: 5, decrypt:  abjnqnlfvagreargbsguvatfvfnjvqrylqvfphffrqgbcvpnzbaterfrnepuref
    move: 6, decrypt:  zaimpmkeuzfqdzqfarftuzseuemiupqxkpueogeeqpfabuomyazsdqeqmdotqde
    move: 7, decrypt:  yzhloljdtyepcypezqestyrdtdlhtopwjotdnfddpoezatnlxzyrcpdplcnspcd
    move: 8, decrypt:  xygknkicsxdobxodypdrsxqcsckgsnovinscmeccondyzsmkwyxqbocokbmrobc
    move: 9, decrypt:  wxfjmjhbrwcnawncxocqrwpbrbjfrmnuhmrbldbbnmcxyrljvxwpanbnjalqnab
    move: 10, decrypt:  vweiligaqvbmzvmbwnbpqvoaqaieqlmtglqakcaamlbwxqkiuwvozmamizkpmza
    move: 11, decrypt:  uvdhkhfzpualyulavmaopunzpzhdpklsfkpzjbzzlkavwpjhtvunylzlhyjolyz
    move: 12, decrypt:  tucgjgeyotzkxtkzulznotmyoygcojkrejoyiayykjzuvoigsutmxkykgxinkxy
    move: 13, decrypt:  stbfifdxnsyjwsjytkymnslxnxfbnijqdinxhzxxjiytunhfrtslwjxjfwhmjwx
    move: 14, decrypt:  rsaehecwmrxivrixsjxlmrkwmweamhipchmwgywwihxstmgeqsrkviwievglivw
    move: 15, decrypt:  qrzdgdbvlqwhuqhwriwklqjvlvdzlghobglvfxvvhgwrslfdprqjuhvhdufkhuv
    move: 16, decrypt:  pqycfcaukpvgtpgvqhvjkpiukucykfgnafkuewuugfvqrkecoqpitgugctejgtu
    move: 17, decrypt:  opxbebztjoufsofupguijohtjtbxjefmzejtdvttfeupqjdbnpohsftfbsdifst
    move: 18, decrypt:  nowadaysinternetofthingsisawidelydiscussedtopicamongresearchers
    move: 19, decrypt:  mnvzczxrhmsdqmdsnesghmfrhrzvhcdkxchrbtrrdcsnohbzlnmfqdrdzqbgdqr
    move: 20, decrypt:  lmuybywqglrcplcrmdrfgleqgqyugbcjwbgqasqqcbrmngaykmlepcqcypafcpq
    move: 21, decrypt:  kltxaxvpfkqbokbqlcqefkdpfpxtfabivafpzrppbaqlmfzxjlkdobpbxozebop
    move: 22, decrypt:  jkswzwuoejpanjapkbpdejcoeowsezahuzeoyqooazpkleywikjcnaoawnydano
    move: 23, decrypt:  ijrvyvtndiozmizojaocdibndnvrdyzgtydnxpnnzyojkdxvhjibmznzvmxczmn
    move: 24, decrypt:  hiquxusmchnylhyniznbchamcmuqcxyfsxcmwommyxnijcwugihalymyulwbylm
    move: 25, decrypt:  ghptwtrlbgmxkgxmhymabgzlbltpbwxerwblvnllxwmhibvtfhgzkxlxtkvaxkl


move: 18, decrypt:  nowadaysinternetofthingsisawidelydiscussedtopicamongresearchers

Nowadays internet of things is a widely discussed topic among researchers.

---
## 8


```python
cipher = """
PSXU TJIT DBVS ZTRB YDRA ALPU AVYR VVYU RXOU TART DAIS PBLV
PVMS JMAT UASL TIUF SLJX UXMJ ASPU TURA SJRF QUNR D
"""
cipher = _format_cipher(cipher)
```


```python
pi = [17, 5, 8, 23, 20, 6, 25, 24, 12, 7, 22, 16, 15, 9, 18, 1, 10, 19, 0, 21, 11, 14, 13, 4, 3, 2]
```


```python

```


```python
from util import id2char, char2id
d_rev = {id2char(id):id2char(idx) for idx, id in enumerate(pi)}
res = [d_rev[x] for x in cipher]
s = "".join(res)
s
```




    'moderncryptographyassumesthattheadversaryscomputmtionisresourceboundedinsomereasonableway'



Modern cryptography assumes that the adversary's computmtion is resource bounded in some reasonable way



---
## 9


```python
from util import char2id

plain_text = 'thanksgiving'
cipher_text = _format_cipher('UITV DCEZ OJRH')
cipher_text
```




    'uitvdcezojrh'




```python
P_ids = [char2id(x) for x in plain_text]
P_ids
```




    [19, 7, 0, 13, 10, 18, 6, 8, 21, 8, 13, 6]




```python
C_ids = [char2id(x) for x in cipher_text]
C_ids
```




    [20, 8, 19, 21, 3, 2, 4, 25, 14, 9, 17, 7]




```python
import numpy as np
import copy
P = [P_ids[i:i + 3] for i in range(0, len(P_ids), 3)][:3]


C = [C_ids[i:i + 3] for i in range(0, len(C_ids), 3)][:3]
```


```python
from util import matrix_rev
K = np.matmul(matrix_rev(P), C)%26
K
```




    array([[ 3, 13, 22],
           [17,  3, 21],
           [ 4,  5,  0]])




```python
from util import encrypt_hill
```


```python
# double check
hill_K = [[ 3, 13, 22],
       [17,  3, 21],
       [ 4,  5,  0]]
print(f'plain_text: {plain_text}')
print(f'cipher_text: {encrypt_hill(plain_text, K=hill_K)}')
```

    plain : thanksgiving
    cipher: uitvdcezojrh





---
## 10


```python
from util import crt
```


```python
crt([3,5,7],[31,41,47])
```


    24400




```python
from sympy.ntheory.modular import crt
crt([31,41,47],[3,5,7])
```


    (24400, 59737)



## 11


```python

```


```python
p = 113
q = 127
b = 17

phi=(p-1)*(q-1)
phi
```


    14112




```python
from math import gcd
def f(x,m): 
    for a in range(1, m):
        if gcd(a, phi) != 1: continue
        if (x*a) % m == 1:
            return a
    print("无解")
    return None
a = f(17, phi)
a
```


    6641




```python
md = p*q
x = 7
mod_exp(x,b,md)
```


    1134




```python
from util import mod_exp
def e_rsa(x):
    return mod_exp(x, b, md)
def d_rsa(y):
    return mod_exp(y, a, md)
```


```python
print(f"e_rsa(7) = {e_rsa(7)}")
print(f"d_rsa(7) = {d_rsa(7)}")
```

    e_rsa(7) = 1134
    d_rsa(7) = 309



## 12


```python
p = 18313
alph = 10
a = 173
beta = mod_exp(alph, a, p)
beta
```


    16073




```python
from util import mod_rev_v2
p = 18313
alph = 10
a = 173
beta = 16073

def Eigamal_encrypt(x):
    import random
#     k = random.randint(0,p-1)
    k = 6
    y1 = mod_exp(alph, k, p)
    y2 = mod_exp(beta, k, p) * x
    return y1,y2


def Eigammal_decrypt(y1, y2):
    return (y2 * mod_rev_v2(mod_exp(y1,a, p),p))%p
```


```python
print(f"Eigamal_encrypt(389) = {Eigamal_encrypt(389)}")
print(f"Eigammal_decrypt(2521, 3280) = {Eigammal_decrypt(2521, 3280)}")
```

    Eigamal_encrypt(389) = (11098, 5990989)
    Eigammal_decrypt(2521, 3280) = 401



```python

```

## 13


```python
p = 8831
q = 883
alph = 2
a = 175
beta = mod_exp(alph, a, p)
print(f"beta: {beta}")

def DSA_encrypt(x, k=50):

    y1 = mod_exp(alph, k, p)%q
    # s = sha-1(x)
    s = 22
    y2 = (s + a*y1)*mod_rev_v2(k, q)
    y2 = y2%q
    return y1,y2


def DSA_decrypt(y1, y2):
    s = 22
    e1 = (s*mod_rev_v2(y2,q))%q
    e2 = (y1*mod_rev_v2(y2,q))%q
    print(f"e1: {e1} e2: {e2}")
    _ =  mod_exp(alph, e1, p)*mod_exp(beta,e2,p)
    _ = _ % p
    print(_%q)
    return _%q == y1
```

    beta: 2176



```python
print(f"DSA_encrypt(13) = {DSA_encrypt(13)}")

print(f"DSA_decrypt(5, 742) = {DSA_decrypt(5, 742)}")
```

    DSA_encrypt(13) = (5, 742)
    e1: 457 e2: 144
    5
    DSA_decrypt(5, 742) = True



