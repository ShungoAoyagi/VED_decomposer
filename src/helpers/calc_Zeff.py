from src.tasks.pre_processing.settings import Settings

# 簡易 Slater 計算（4p 電子を仮定した Fe の例）
Z = 26                               # 原子番号
conf = {
    '1s': 2, '2s': 2, '2p': 6,
    '3s': 2, '3p': 6, '3d': 6,
    '4s': 2, '4p': 1                # ← 該当電子を 1 としておく
}

# Slater の係数表（ns/np 電子用）
weights = {
    0: 0.35,          # 同じ (n, l=s/p) で自分以外
    1: 0.85,          # n-1 の全電子
    2: 1.00           # n-2 以下は一律 1.00
}

def calc_Zeff(nsorb: int, settings: Settings):
    n = int(nsorb[0])
    l = nsorb[1]                      # 's','p','d','f'
    if l not in 'sp':
        raise ValueError('Slater 表は s/p 用の簡易版です')
    s = 0.0
    for orb, n_e in conf.items():
        n_o = int(orb[0])
        if orb == nsorb:              # 自分は除く
            n_e -= 1
        dn = n - n_o
        if dn == 0 and orb[1] in 'sp':
            s += weights[0] * n_e
        elif dn == 1:
            s += weights[1] * n_e
        elif dn >= 2:
            s += weights[2] * n_e
    return Z - s
