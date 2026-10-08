"""Audio: song bed + speed-aware game audio + kill shots re-synced + synthesized whoosh/boom, then loudness/limiter."""
import json, subprocess, numpy as np
from scipy.signal import butter, sosfiltfilt
import engine

ROOT = engine.ROOT
SR = 48000
import os
EDIT = os.environ.get('EDIT', 'edit')
BASE = f'{ROOT}/render' if EDIT == 'edit' else f'{ROOT}/render/{EDIT}'
edit = json.load(open(f'{ROOT}/{EDIT}.json'))
DUR = edit['duration']
N = int(DUR * SR)


def load(path, t0=None, dur=None):
    cmd = ['ffmpeg', '-v', 'error']
    if t0 is not None:
        cmd += ['-ss', f'{t0:.4f}']
    cmd += ['-i', path]
    if dur is not None:
        cmd += ['-t', f'{dur:.4f}']
    cmd += ['-af', 'aresample=async=1:first_pts=0', '-ac', '2', '-ar', str(SR), '-f', 'f32le', '-']
    a = np.frombuffer(subprocess.check_output(cmd), np.float32).reshape(-1, 2).copy()
    return a


def ss(a, b, x):
    return engine.smoothstep(a, b, x)


# ---------------- song ----------------
song = load(f'{ROOT}/audio/song.wav', edit['song_start'], DUR + 0.1)[:N]
song = np.pad(song, ((0, N - len(song)), (0, 0)))
t = np.arange(N) / SR
# low-pass "underwater" during B&W slow-mo setups, crossfaded
lp = sosfiltfilt(butter(4, 650, 'low', fs=SR, output='sos'), song, axis=0).astype(np.float32)
m = np.zeros(N, np.float32)
for p in edit['plays']:
    for a, b in p.get('bw', []):
        m = np.maximum(m, ss(a - 0.2, a + 0.1, t) * (1 - ss(b, b + 0.06, t)))
song = song * (1 - m[:, None]) + lp * m[:, None]
song *= ss(0, 0.12, t)[:, None]
song *= (1 - ss(DUR - 0.75, DUR, t))[:, None]

# ---------------- game audio ----------------
game = np.zeros((N, 2), np.float32)
shots = np.zeros((N, 2), np.float32)
for p in edit['plays']:
    f0, f1 = round(p['out'][0] * 60), round(p['out'][1] * 60)
    pl = engine.Play(p, f0, f1 - f0)
    lag = p['audio_lag']
    lo, hi = pl.src_range()
    a0 = max(0.0, lo - 0.3)
    src = load(f"{engine.CLIPS}/{engine.INDEX[p['clip']]}", a0, hi - a0 + 1.0)
    n0, n1 = int(p['out'][0] * SR), int(p['out'][1] * SR)
    to = np.arange(n0, n1) / SR
    st = pl.map(to) + lag                       # varispeed read position
    spd = np.maximum(pl.dmap(to), 0)
    g = ss(0.5, 0.8, spd) * (1 - ss(1.35, 1.9, spd))
    pos = (st - a0) * SR
    i = np.clip(pos.astype(np.int64), 0, len(src) - 2)
    fr = (pos - i)[:, None].astype(np.float32)
    seg = src[i] * (1 - fr) + src[i + 1] * fr
    # 8 ms fades at play edges
    e = np.minimum(1, np.minimum(np.arange(n1 - n0), np.arange(n1 - n0)[::-1]) / (0.008 * SR))
    # duck the continuous game audio around each kill; the aligned gunshot below replaces it
    duckg = np.ones(n1 - n0, np.float32)
    for ko, ks in p['kills']:
        duckg = np.minimum(duckg, 1 - 0.9 * ss(ko - 0.12, ko - 0.06, to) * (1 - ss(ko + 0.35, ko + 0.6, to)))
    game[n0:n1] += seg * (g * e * duckg)[:, None]
    # gunshot for every kill: find the real transient in the clip audio and put it exactly on the kill time
    hp = sosfiltfilt(butter(2, 400, 'high', fs=SR, output='sos'), src.mean(axis=1))
    hop = 48
    for (ko, ks), stg in zip(p['kills'], p['kill_strength']):
        w0 = int((ks + lag - 0.10 - a0) * SR); w1 = int((ks + lag + 0.22 - a0) * SR)
        w0 = max(w0, 0); w1 = min(w1, len(hp) - hop)
        en = np.sqrt(np.mean(hp[w0:w0 + (w1 - w0) // hop * hop].reshape(-1, hop) ** 2, axis=1))
        rise = np.diff(np.log(en + 1e-4), prepend=np.log(en[0] + 1e-4))
        j = int(np.argmax(rise * (en > 0.25 * en.max())))
        t_tr = w0 + j * hop                      # transient sample in src
        k0 = max(0, t_tr - int(0.004 * SR))
        L = int(0.7 * SR)
        clip = src[k0:k0 + L].copy()
        env = np.exp(-np.arange(len(clip)) / (0.28 * SR)) * np.minimum(1, np.arange(len(clip)) / 48)
        o = int((ko - 0.004) * SR)
        w = min(len(clip), N - o)
        shots[o:o + w] += clip[:w] * env[:w, None] * 0.95

# ---------------- synthesized sfx ----------------
rng = np.random.default_rng(3)
sfx = np.zeros((N, 2), np.float32)


def whoosh(center, length=0.42, gain=0.22):
    L = int(length * SR)
    noise = rng.normal(0, 1, (L, 2)).astype(np.float32)
    x = np.linspace(0, 1, L)
    env = (np.sin(np.pi * x) ** 2) * np.exp(-3 * (x - 0.6) ** 2)
    out = np.zeros_like(noise)
    # sweep a band-pass by chunking (cheap filter sweep)
    chunks = 12
    for c in range(chunks):
        a, b = c * L // chunks, (c + 1) * L // chunks
        fc = 400 + 3500 * np.sin(np.pi * (c + 0.5) / chunks)
        sos = butter(2, [fc * 0.6, min(fc * 1.6, 20000)], 'band', fs=SR, output='sos')
        out[a:b] = sosfiltfilt(sos, noise[max(0, a - 2000):b], axis=0)[-(b - a):]
    out *= env[:, None] / (np.abs(out).max() + 1e-6)
    o = int((center - length * 0.6) * SR)
    o2 = max(0, o)
    w = min(L - (o2 - o), N - o2)
    sfx[o2:o2 + w] += gain * out[o2 - o:o2 - o + w]


def boom(at, gain=0.5):
    L = int(0.9 * SR)
    x = np.arange(L) / SR
    f = 38 + 70 * np.exp(-x / 0.08)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * np.exp(-x / 0.35) + 0.25 * rng.normal(0, 1, L) * np.exp(-x / 0.015)
    o = int(at * SR)
    w = min(L, N - o)
    sfx[o:o + w] += gain * s[:w, None].astype(np.float32)


for a, b in zip(edit['plays'], edit['plays'][1:]):
    if a['tout'] in ('whip', 'flash'):
        whoosh(b['out'][0], gain=0.20 if a['tout'] == 'whip' else 0.14)
for p in edit['plays']:
    for (ko, ks), stg in zip(p['kills'], p['kill_strength']):
        if stg >= 1.5:
            boom(ko, 0.55)

# ---------------- mix + master ----------------
# duck the song a touch under each kill so the shot reads as percussion
duck = np.ones(N, np.float32)
for p in edit['plays']:
    for ko, ks in p['kills']:
        d = t - ko
        duck = np.minimum(duck, 1 - 0.22 * np.exp(-np.maximum(d, 0) / 0.12) * (d > -0.01))
mix = song * 0.80 * duck[:, None] + game * 0.55 + shots * 0.75 + sfx
np.save(f'{BASE}/mix_raw.npy', mix)
raw = f'{BASE}/mix_raw.f32'
mix.astype(np.float32).tofile(raw)
# measure loudness, normalise to -14 LUFS, true-peak limit -1 dBTP
meas = subprocess.run(['ffmpeg', '-hide_banner', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', raw, '-af',
                       'loudnorm=I=-14:TP=-1:LRA=11:print_format=json', '-f', 'null', '-'], capture_output=True, text=True).stderr
js = json.loads(meas[meas.rindex('{'):meas.rindex('}') + 1])
af = (f"loudnorm=I=-14:TP=-1:LRA=11:measured_I={js['input_i']}:measured_TP={js['input_tp']}:measured_LRA={js['input_lra']}:"
      f"measured_thresh={js['input_thresh']}:offset={js['target_offset']}:linear=true,aresample={SR}")
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', raw, '-af', af,
                '-c:a', 'pcm_s24le', f'{BASE}/mix.wav'], check=True)
print('input loudness', js['input_i'], 'LUFS, tp', js['input_tp'])
