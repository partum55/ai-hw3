"""Window arithmetic for long-text encoders (numpy only, testable without a GPU)."""
import numpy as np

def window_starts(n, W=510, stride=384, maxw=6):
    """Start offsets of <=W-token windows covering a text of n tokens; the last window is flush with the end;
    if more than maxw windows are needed, maxw of them are picked evenly (first and last always kept)."""
    if n <= W: return [0]
    s = list(range(0, n - W + 1, stride))
    if s[-1] != n - W: s.append(n - W)
    if len(s) > maxw: s = [s[round(k * (len(s) - 1) / (maxw - 1))] for k in range(maxw)]
    return s

def aggregate(probs, owner, n_texts):
    """probs [n_windows, 3], owner [n_windows] sorted text indices -> mean [n,3] and extras [n,7] = max(3), std(3), n_windows."""
    b = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]]); cnt = np.diff(np.r_[b, len(owner)]).astype(np.float32)
    mean = np.add.reduceat(probs, b) / cnt[:, None]
    std = np.sqrt(np.maximum(np.add.reduceat(probs ** 2, b) / cnt[:, None] - mean ** 2, 0))
    assert len(b) == n_texts, "every text must own at least one window"
    return mean, np.hstack([np.maximum.reduceat(probs, b), std, cnt[:, None]])

def positional(probs, owner, n_texts):
    """[n,13] = max over windows (3), first window (3), last window (3), mean of the middle windows (3; all windows if < 3), n_windows."""
    b = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]]); e = np.r_[b[1:], len(owner)]; assert len(b) == n_texts
    out = np.zeros((n_texts, 13), np.float32)
    for r, (s, t) in enumerate(zip(b, e)):
        q = probs[s:t]; mid = q[1:-1] if len(q) >= 3 else q
        out[r] = np.r_[q.max(0), q[0], q[-1], mid.mean(0), len(q)]
    return out

if __name__ == "__main__":
    assert window_starts(300) == [0] and window_starts(510) == [0]
    assert window_starts(800) == [0, 290] and window_starts(1000) == [0, 384, 490]
    s = window_starts(5000); assert len(s) == 6 and s[0] == 0 and s[-1] == 4490
    p = np.array([[.2, .8, 0], [.4, .6, 0], [1., 0, 0]], np.float32); m, x = aggregate(p, np.array([0, 0, 1]), 2)
    assert np.allclose(m, [[.3, .7, 0], [1, 0, 0]]) and np.allclose(x[0, :3], [.4, .8, 0]) and np.allclose(x[0, 3:6], [.1, .1, 0]) and x[:, 6].tolist() == [2, 1]
    q = positional(np.array([[.2, .8, 0], [.4, .6, 0], [1., 0, 0], [0, 0, 1.]], np.float32), np.array([0, 0, 0, 1]), 2)
    assert np.allclose(q[0], [1, .8, 0, .2, .8, 0, 1, 0, 0, .4, .6, 0, 3]) and np.allclose(q[1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1])
    print("ok")
