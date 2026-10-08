"""CS2 montage render engine.

Time model: each play maps output time -> source time through anchors [(out_t, src_t), ...]
interpolated with a monotone cubic (PCHIP), so speed changes are smooth ramps (Twixtor-style).
Local speed s = d(src)/d(out).

Per output frame:
  s < SLOW_T      -> source sampled at fractional times from RIFE-interpolated frames (optical-flow slow-mo)
  any speed       -> vector motion blur along optical flow, length = shutter * s * flow (RSMB-style)
  s > FF_T        -> shutter drag: 12 fps stepped cadence, each step averages all source frames it spans
                     (each one vector-blurred) -> long-exposure look; desaturated to B&W
  near kills      -> colour returns, soft edge flash, radial chromatic aberration, zoom punch
"""
import json, math, os, shutil, subprocess, sys
from functools import lru_cache
import numpy as np
import cv2
from scipy.interpolate import PchipInterpolator

ROOT = os.path.dirname(os.path.abspath(__file__))
RIFE = os.path.abspath(os.environ.get('RIFE_BIN', f'{ROOT}/tools/ubuntu/rife-ncnn-vulkan'))
RIFE_MODEL = os.environ.get('RIFE_MODEL', 'rife-v4.26')
RIFE_GPU = os.environ.get('RIFE_GPU', '0')
VIDEO_CODEC = os.environ.get('VIDEO_CODEC', 'libx264')


def encoder_args(final=False):
    """Select portable CPU encoding or the original NVIDIA encoding settings."""
    if VIDEO_CODEC == 'libx264':
        return ['-c:v', VIDEO_CODEC, '-preset', 'medium', '-crf', '17' if final else '13']
    if VIDEO_CODEC == 'h264_nvenc':
        return ['-c:v', VIDEO_CODEC, '-preset', 'p7' if final else 'p6', '-rc', 'vbr',
                '-cq', '17' if final else '13', '-b:v', '0']
    raise ValueError('VIDEO_CODEC must be libx264 or h264_nvenc')
FPS = 60
W, H = 1920, 1080
SLOW_T = 0.92        # below this speed use RIFE sub-frames
FF_T = 2.2           # above this speed start the shutter-drag look
STEP = 5             # 60/5 = 12 fps stepped cadence in fast-forward
SHUTTER = 0.5        # 180 degree shutter for normal / slow footage
CLIPS = os.path.abspath(os.environ.get('CLIPS_DIR', f'{ROOT}/clips'))
INDEX_PATH = os.environ.get('CLIP_INDEX', f'{ROOT}/index.txt')
with open(INDEX_PATH) as index_file:
    INDEX = dict(l.rstrip('\n').split('|', 1) for l in index_file if l.strip())
cv2.setNumThreads(4)
# watermark / build-string removal (coordinates in the 1920x1080 scaled frame)
_ALLSTAR = ',delogo=x=46:y=34:w=240:h=76'
_LEETIFY = ',delogo=x=8:y=10:w=184:h=58'
_BUILD = ',delogo=x=1:y=1053:w=390:h=25'
DELOGO = {'18': _ALLSTAR, '20': _ALLSTAR, '17': _ALLSTAR, '21': _LEETIFY, '47': _LEETIFY,
          '03': _BUILD, '05': _BUILD, '08': _BUILD, '10': _BUILD,
          '00': _BUILD, '01': _BUILD, '02': _BUILD, '06': _BUILD, '07': _BUILD, '13': _BUILD}
# Original clip-specific boxes are opt-in; arbitrary clips must not inherit them.
if os.environ.get('CS2_DELOGO') != '1':
    DELOGO = {}


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------------------------------------
# Time mapping
# ----------------------------------------------------------------------------------------------
class Play:
    def __init__(self, spec, out_start_frame, n_frames):
        self.spec = spec
        self.clip = spec['clip']
        self.f0 = out_start_frame
        self.n = n_frames
        a = np.array(spec['anchors'], float)
        self.map = PchipInterpolator(a[:, 0], a[:, 1], extrapolate=True)
        self.dmap = self.map.derivative()
        tt = (self.f0 + np.arange(n_frames)) / FPS
        self.t_out = tt
        self.src = self.map(tt)
        self.speed = np.maximum(self.dmap(tt), 0.02)
        self.kills_out = [k[0] for k in spec.get('kills', [])]
        # stepped cadence (shutter drag) where fast: hold the frame at the start of each 5-frame step
        ff = self.speed > FF_T
        self.hold = np.arange(n_frames)
        i = 0
        while i < n_frames:
            if ff[i]:
                j = i
                while j < n_frames and ff[j] and j - i < STEP:
                    j += 1
                self.hold[i:j] = i
                i = j
            else:
                i += 1
        # exposure window (source seconds) per output frame
        win = SHUTTER * self.speed / FPS
        step_len = np.array([np.sum(self.hold == self.hold[k]) for k in range(n_frames)])
        span = np.array([self.src[min(n_frames - 1, self.hold[k] + step_len[k] - 1)] - self.src[self.hold[k]] + self.speed[k] / FPS
                         for k in range(n_frames)])
        self.window = np.where(ff, span, win)
        self.src_start = np.where(ff, self.src[self.hold], self.src - win / 2)

    def src_range(self):
        lo = float(np.min(self.src_start)) - 0.1
        hi = float(np.max(self.src_start + self.window)) + 0.15
        return max(0.0, lo), hi


# ----------------------------------------------------------------------------------------------
# Source access
# ----------------------------------------------------------------------------------------------
class Source:
    """Raw 60fps frames (memmap) for [t0, t1] of a clip, plus RIFE sub-frames for slow ranges."""

    def __init__(self, clip, t0, t1, work):
        work = os.path.abspath(work)
        self.clip, self.t0, self.t1, self.work = clip, t0, t1, work
        os.makedirs(work, exist_ok=True)
        self.raw_path = f'{work}/raw.u8'
        path = f'{CLIPS}/{INDEX[clip]}'
        n = int(math.ceil((t1 - t0) * FPS)) + 1
        if not os.path.exists(self.raw_path + '.done'):
            with open(self.raw_path, 'wb') as fh:
                subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t0:.4f}', '-i', path, '-t', f'{t1 - t0 + 0.05:.4f}',
                                '-vf', f'fps={FPS},scale={W}:{H}:flags=lanczos,setsar=1{DELOGO.get(clip, "")},format=rgb24',
                                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=fh, check=True)
            open(self.raw_path + '.done', 'w').close()
        sz = os.path.getsize(self.raw_path) // (W * H * 3)
        self.n = sz
        self.raw = np.memmap(self.raw_path, np.uint8, 'r', shape=(sz, H, W, 3))
        self.dense = []   # list of (i0, i1, U, dir)
        self._flow_cache = {}

    def raw_frame(self, i):
        i = int(np.clip(i, 0, self.n - 1))
        return self.raw[i]

    def add_dense(self, i0, i1, U):
        """Interpolate raw frames i0..i1 (inclusive) U-times with RIFE."""
        i0 = max(0, i0); i1 = min(self.n - 1, i1)
        d = f'{self.work}/dense_{i0}_{i1}_{U}'
        if not os.path.exists(d + '/.done'):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d + '/in'); os.makedirs(d + '/out')
            for k in range(i0, i1 + 1):
                cv2.imwrite(f'{d}/in/{k - i0:06d}.png', cv2.cvtColor(np.asarray(self.raw[k]), cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_PNG_COMPRESSION, 1])
            n_in = i1 - i0 + 1
            import fcntl, time
            # one RIFE at a time on the 4 GB laptop GPU; retry on driver hiccups
            with open(f'{ROOT}/render/rife.lock', 'w') as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                for attempt in range(4):
                    shutil.rmtree(d + '/out', ignore_errors=True); os.makedirs(d + '/out')
                    r = subprocess.run([RIFE, '-i', d + '/in', '-o', d + '/out', '-n', str(n_in * U), '-m', RIFE_MODEL,
                                        '-g', RIFE_GPU, '-j', '1:2:2', '-f', '%06d.png'], cwd=os.path.dirname(RIFE),
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    if r.returncode == 0 and len(os.listdir(d + '/out')) == n_in * U and all(os.path.getsize(f'{d}/out/{q}') > 1000 for q in os.listdir(d + '/out')):
                        break
                    sys.stderr.write(f'RIFE attempt {attempt} failed: {r.stderr[-400:]!r}\n'); time.sleep(3)
                else:
                    raise RuntimeError('RIFE failed')
            shutil.rmtree(d + '/in')
            open(d + '/.done', 'w').close()
        self.dense.append((i0, i1, U, d))

    @lru_cache(maxsize=96)
    def dense_frame(self, d, j):
        im = cv2.imread(f'{d}/out/{j + 1:06d}.png')
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    def frame_at(self, t):
        """Frame at fractional source time t (seconds, absolute clip time)."""
        x = (t - self.t0) * FPS
        for i0, i1, U, d in self.dense:
            if i0 <= x <= i1:
                j = int(round((x - i0) * U))
                j = min(j, (i1 - i0 + 1) * U - 1)
                return self.dense_frame(d, j)
        return self.raw_frame(round(x))

    def flow(self, i):
        """Optical flow raw frame i -> i+1 at quarter res, returned at quarter res (pixels of full res)."""
        i = int(np.clip(i, 0, self.n - 2))
        if i in self._flow_cache:
            return self._flow_cache[i]
        a = cv2.cvtColor(cv2.resize(np.asarray(self.raw[i]), (W // 4, H // 4), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(cv2.resize(np.asarray(self.raw[i + 1]), (W // 4, H // 4), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        f = DIS.calc(a, b, None) * 4.0
        if len(self._flow_cache) > 64:
            self._flow_cache.pop(next(iter(self._flow_cache)))
        self._flow_cache[i] = f
        return f


DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
GX, GY = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))


def vector_blur(img, flow_q, length, max_taps=14):
    """RSMB-style blur: average samples along the per-pixel motion vector (flow * length), centred."""
    if length <= 0.02:
        return img.astype(np.float32)
    f = cv2.resize(flow_q, (W, H), interpolation=cv2.INTER_LINEAR) * np.float32(length)
    mag = float(np.percentile(np.abs(f).max(axis=2), 99))
    taps = int(np.clip(mag / 3.0, 1, max_taps))
    if taps <= 1:
        return img.astype(np.float32)
    acc = np.zeros((H, W, 3), np.float32)
    fx, fy = f[..., 0], f[..., 1]
    for k in range(taps):
        u = np.float32((k + 0.5) / taps - 0.5)
        acc += cv2.remap(img, GX + fx * u, GY + fy * u, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return acc / taps


# ----------------------------------------------------------------------------------------------
# Look
# ----------------------------------------------------------------------------------------------
def build_luts():
    """Cool, slightly crushed contrast grade: S-curve, blue/teal shadows, neutral-warm highlights."""
    x = np.arange(256, dtype=np.float32) / 255
    s = x * 0.55 + (x * x * (3 - 2 * x)) * 0.45
    s = np.clip((s - 0.03) / 0.97, 0, 1)
    sh = (1 - x) ** 3
    hi = x ** 3
    r = np.clip(s - 0.030 * sh + 0.010 * hi, 0, 1)
    g = np.clip(s + 0.006 * sh, 0, 1)
    b = np.clip(s + 0.050 * sh - 0.012 * hi, 0, 1)
    return [(c * 255).astype(np.uint8) for c in (r, g, b)]


LUT_R, LUT_G, LUT_B = build_luts()
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
RN = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / math.sqrt(2)
VIGNETTE = (1 - 0.32 * smoothstep(0.45, 1.0, RN))[..., None].astype(np.float32)
EDGE = smoothstep(0.38, 1.0, RN)[..., None].astype(np.float32)        # peripheral flash mask
EDGE_COL = np.array([0.80, 0.90, 1.0], np.float32)                     # cool white


def radial_ca(img, amt):
    """Chromatic aberration growing towards the edges: R scaled out, B scaled in."""
    if amt < 0.0005:
        return img
    out = img.copy()
    for ch, k in ((0, 1 + amt), (2, 1 - amt)):
        M = np.float32([[k, 0, (1 - k) * W / 2], [0, k, (1 - k) * H / 2]])
        out[..., ch] = cv2.warpAffine(img[..., ch], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


def grade(img_u8, sat):
    r, g, b = cv2.split(img_u8)
    img = cv2.merge([cv2.LUT(r, LUT_R), cv2.LUT(g, LUT_G), cv2.LUT(b, LUT_B)]).astype(np.float32)
    gray = img @ np.array([0.299, 0.587, 0.114], np.float32)
    sat = float(sat)
    if abs(sat - 1) > 1e-3:
        img = gray[..., None] + (img - gray[..., None]) * sat
    return img


# ----------------------------------------------------------------------------------------------
# Rendering one play
# ----------------------------------------------------------------------------------------------
def envelopes(play):
    """Per-frame effect curves for a play."""
    n = play.n
    t = play.t_out
    s = play.speed
    near = np.zeros(n, np.float32)
    flash = np.zeros(n, np.float32)
    punch = np.zeros(n, np.float32)
    ca = 0.0025 * smoothstep(1.4, 4.0, s)
    shake = np.zeros((n, 2), np.float32)
    rng = np.random.default_rng(int(play.spec['clip']) * 7 + play.f0)
    strength = play.spec.get('kill_strength', [1.0] * len(play.kills_out))
    for K, st in zip(play.kills_out, strength):
        d = t - K
        pre = smoothstep(-0.60, -0.14, d) * (d < 0)
        post = np.where(d >= 0, 1 - smoothstep(0.55, 1.1, d), 0)
        near = np.maximum(near, np.maximum(pre, post))
        e = np.where(d >= -1e-6, np.exp(-np.maximum(d, 0) / 0.11), 0)
        flash = np.maximum(flash, st * e)
        punch = np.maximum(punch, st * np.where(d >= -1e-6, np.exp(-np.maximum(d, 0) / 0.14), 0)
                           + 0.25 * st * smoothstep(-0.12, 0, d) * (d < 0))
        jitter = rng.normal(0, 1, (n, 2))
        shake += (st * 7.0 * np.where(d >= 0, np.exp(-np.maximum(d, 0) / 0.07), 0))[:, None] * jitter
        ca = ca + st * 0.009 * np.where(d >= 0, np.exp(-np.maximum(d, 0) / 0.10), 0)
    tr = smoothstep(1.5, 2.8, s)
    bwf = np.zeros(n, np.float32)
    for a, b in play.spec.get('bw', []):            # forced B&W spans (output seconds)
        bwf = np.maximum(bwf, smoothstep(a - 0.15, a, t) * (1 - smoothstep(b, b + 0.12, t)))
    bw = np.maximum(tr, bwf) * (1 - near)
    sat = (1 - bw) * 0.86 + 0.25 * flash
    # transitions at the edges of the play
    white = np.zeros(n, np.float32)
    whip = np.zeros(n, np.float32)
    fin, fout = play.spec.get('tin', 'cut'), play.spec.get('tout', 'cut')
    k = np.arange(n)
    if fin == 'flash':
        white = np.maximum(white, 0.95 * np.exp(-k / 4.5))
    if fout == 'flash':
        white = np.maximum(white, 0.95 * smoothstep(n - 7, n - 1, k))
    if fin == 'whip':
        whip = np.maximum(whip, 1 - smoothstep(0, 6, k))
    if fout == 'whip':
        whip = np.maximum(whip, smoothstep(n - 7, n - 1, k))
    if fin == 'fade':
        white = np.maximum(white, 0)  # handled as fade from black below
    return dict(sat=sat, flash=flash, punch=punch, ca=ca + 0.010 * whip, shake=shake, white=white, whip=whip,
                fade_in=(1 - smoothstep(0, 18, k)) if fin == 'fade' else np.zeros(n),
                fade_out=smoothstep(n - play.spec.get('fade_frames', 30), n - 1, k) if fout == 'fade' else np.zeros(n))


def base_frame(src, play, k):
    """Motion-blurred source image for output frame k (float32 RGB)."""
    s = play.speed[k]
    if s > FF_T:
        a = play.src_start[k]
        b = a + play.window[k]
        i0 = (a - src.t0) * FPS
        i1 = (b - src.t0) * FPS
        idx = np.arange(math.floor(i0), math.ceil(i1) + 1)
        stride = max(1, int(math.ceil(len(idx) / 16)))
        idx = idx[::stride]
        acc = np.zeros((H, W, 3), np.float32)
        for i in idx:
            acc += vector_blur(np.asarray(src.raw_frame(i)), src.flow(i), float(stride), max_taps=10)
        return acc / len(idx)
    t = play.src[k]
    img = np.asarray(src.frame_at(t))
    i = int(math.floor((t - src.t0) * FPS))
    return vector_blur(img, src.flow(i), SHUTTER * s)


def render_play(spec, f0, n, out_path, work):
    play = Play(spec, f0, n)
    lo, hi = play.src_range()
    src = Source(spec['clip'], lo, hi, work)
    # RIFE ranges where the play is slow
    slow = play.speed < SLOW_T
    if slow.any():
        xs = (play.src[slow] - src.t0) * FPS
        sp = play.speed[slow]
        # group contiguous runs
        runs = []
        idxs = np.flatnonzero(slow)
        start = idxs[0]; prev = idxs[0]
        for q in list(idxs[1:]) + [None]:
            if q is None or q != prev + 1:
                seg = slice(start, prev + 1)
                x = (play.src[seg] - src.t0) * FPS
                U = 8 if play.speed[seg].min() < 0.45 else 4
                runs.append((int(math.floor(x.min())) - 1, int(math.ceil(x.max())) + 1, U))
                if q is not None:
                    start = q
            if q is not None:
                prev = q
        for i0, i1, U in runs:
            src.add_dense(i0, i1, U)
    env = envelopes(play)
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                            '-r', str(FPS), '-i', '-'] + encoder_args() +
                           ['-pix_fmt', 'yuv420p', '-g', '60', out_path], stdin=subprocess.PIPE)
    zcurve = np.ones(n, np.float32)
    zk = spec.get('zoomk')
    if zk:
        for (ta, za), (tb, zb) in zip(zk, zk[1:]):
            m = (play.t_out >= ta) & (play.t_out <= tb)
            zcurve[m] = za + (zb - za) * smoothstep(ta, tb, play.t_out[m])
    cache_k, cache_img = -1, None
    for k in range(n):
        hk = play.hold[k] if play.speed[k] > FF_T else k
        if hk == cache_k:
            img = cache_img
        else:
            img = base_frame(src, play, hk)
            cache_k, cache_img = hk, img
        # zoom punch + shake
        z = (1 + 0.06 * env['punch'][k] + spec.get('zoom', 0.0)) * zcurve[k]
        dx, dy = env['shake'][k]
        if z > 1.0005 or abs(dx) + abs(dy) > 0.2:
            M = np.float32([[z, 0, (1 - z) * W / 2 + dx], [0, z, (1 - z) * H / 2 + dy]])
            img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img = radial_ca(img, float(env['ca'][k]))
        img = grade(np.clip(img, 0, 255).astype(np.uint8), env['sat'][k])
        fl = float(env['flash'][k])
        if fl > 0.01:
            img = img * (1 + 0.18 * fl) + (EDGE * EDGE_COL) * (255 * 0.75 * fl)
        img = img * VIGNETTE
        wp = float(env['whip'][k])
        if wp > 0.02:
            kx = int(8 + 150 * wp) | 1
            img = cv2.blur(img, (kx, 1 + int(6 * wp)))
            img = img * (1 + 0.25 * wp)
        wt = float(env['white'][k])
        if wt > 0.01:
            img = img * (1 - wt) + 245 * wt
        fd = max(float(env['fade_in'][k]), float(env['fade_out'][k]))
        if fd > 0:
            img = img * (1 - fd)
        enc.stdin.write(np.clip(img, 0, 255).astype(np.uint8).tobytes())
    enc.stdin.close()
    returncode = enc.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, enc.args)
    return out_path


if __name__ == '__main__':
    # usage: engine.py edit.json play_index
    edit = json.load(open(sys.argv[1]))
    i = int(sys.argv[2])
    plays = edit['plays']
    f0 = round(plays[i]['out'][0] * FPS)
    f1 = round(plays[i]['out'][1] * FPS)
    name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    base = f'{ROOT}/render' if name == 'edit' else f'{ROOT}/render/{name}'
    os.makedirs(f'{base}/pieces', exist_ok=True)
    out = f'{base}/pieces/{i:02d}.mp4'
    render_play(plays[i], f0, f1 - f0, out, f'{base}/work/{i:02d}')
    shutil.rmtree(f'{base}/work/{i:02d}', ignore_errors=True)
    print('done', i, out)
