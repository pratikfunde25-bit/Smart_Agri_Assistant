import subprocess
import sys

out = subprocess.check_output('git rev-list --objects --all', shell=True).decode('utf-8')
lines = out.strip().split('\n')
objects = [l.split(' ', 1) for l in lines if ' ' in l]

hashes = '\n'.join([o[0] for o in objects]).encode('utf-8')
result = subprocess.run(['git', 'cat-file', '--batch-check=%(objectname) %(objectsize)'], input=hashes, capture_output=True)
out2 = result.stdout.decode('utf-8')

sizes = {}
for l in out2.strip().split('\n'):
    p = l.split(' ')
    if len(p) == 2:
        sizes[p[0]] = int(p[1])

largest = sorted(objects, key=lambda x: sizes.get(x[0], 0), reverse=True)[:20]
for h, n in largest:
    mb = sizes.get(h, 0) / (1024 * 1024)
    if mb > 1:
        print(f'{mb:.2f} MB - {n}')
