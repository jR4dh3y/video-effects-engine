"""Concatenate rendered pieces, mux mastered audio, final encode + checks."""
import json, os, subprocess
import engine
ROOT = engine.ROOT
import os
EDIT = os.environ.get('EDIT', 'edit')
BASE = f'{ROOT}/render' if EDIT == 'edit' else f'{ROOT}/render/{EDIT}'
edit = json.load(open(f'{ROOT}/{EDIT}.json'))
n = len(edit['plays'])
pieces = [f'{BASE}/pieces/{i:02d}.mp4' for i in range(n)]
total = 0
durs = []
for i, p in enumerate(pieces):
    exp = round(edit['plays'][i]['out'][1] * 60) - round(edit['plays'][i]['out'][0] * 60)
    got = int(subprocess.check_output(['ffprobe', '-v', 'error', '-count_packets', '-select_streams', 'v:0', '-show_entries',
                                       'stream=nb_read_packets', '-of', 'csv=p=0', p]).decode().strip())
    assert got == exp, (p, got, exp)
    durs.append(got)
# chain: hard cuts via concat, 'blend' boundaries via a cross-dissolve over the overlapping frames
inputs, fc = [], []
for p in pieces:
    inputs += ['-i', p]
for i in range(len(pieces)):
    fc.append(f'[{i}:v]settb=1/60,setpts=N,fps=60,setsar=1,format=yuv420p[s{i}]')
acc, acc_n = '[s0]', durs[0]
for i in range(1, len(pieces)):
    prev, cur = edit['plays'][i - 1], edit['plays'][i]
    lab = f'[v{i}]'
    if prev['tout'] == 'blend':
        ov = round(prev['out'][1] * 60) - round(cur['out'][0] * 60)
        fc.append(f"{acc}[s{i}]xfade=transition=fade:duration={ov / 60:.6f}:offset={(acc_n - ov) / 60:.6f},settb=1/60{lab}")
        acc_n += durs[i] - ov
    else:
        fc.append(f"{acc}[s{i}]concat=n=2:v=1:a=0,settb=1/60{lab}")
        acc_n += durs[i]
    acc = lab
total = acc_n
assert total == round(edit['duration'] * 60), total
out = f'{BASE}/' + os.environ.get('OUTNAME', 'cs2-soulless-montage-80s.mp4')
subprocess.run(['ffmpeg', '-v', 'error', '-y'] + inputs + ['-i', f'{BASE}/mix.wav', '-filter_complex', ';'.join(fc) + f';{acc}fps=60,settb=1/60[vout]',
                '-map', '[vout]', '-map', f'{len(pieces)}:a'] + engine.encoder_args(final=True) + [
                '-profile:v', 'high', '-pix_fmt', 'yuv420p',
                '-g', '120', '-bf', '3', '-r', '60', '-c:a', 'aac', '-b:a', '320k', '-ar', '48000',
                '-movflags', '+faststart', '-t', str(edit['duration']), out], check=True)
print(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration,size,bit_rate:stream=codec_name,width,height,r_frame_rate,nb_frames',
                               '-of', 'compact', out]).decode())
print('frames:', total, 'out:', out)
