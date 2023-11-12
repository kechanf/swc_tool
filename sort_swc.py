"""*================================================================
*   Copyright (C) 2023 Gaoyu Wang (Braintell, Southeast University). All rights reserved.
*
*   Filename    : sort_swc.py
*   Author      : Gaoyu Wang
*   Date        : 2023-01-11
*   Description :
*
================================================================*"""

from collections import defaultdict
import subprocess
import os
import glob


def parse_swc(swc_file):
    tree = []
    with open(swc_file) as fp:
        for line in fp.readlines():
            line = line.strip()
            if not line: continue
            if line[0] == '#': continue
            idx, type_, x, y, z, r, p = line.split()[:7]
            idx = int(idx)
            type_ = int(type_)
            x = float(x)
            y = float(y)
            z = float(z)
            r = float(r)
            p = int(p)
            tree.append((idx, type_, x, y, z, r, p))
    return tree


def write_swc(tree, swc_file, header=tuple()):
    if header is None:
        header = []
    with open(swc_file, 'w') as fp:
        for s in header:
            if not s.startswith("#"):
                s = "#" + s
            if not s.endswith("\n") or not s.endswith("\r"):
                s += "\n"
            fp.write(s)
        fp.write(f'##n type x y z r parent\n')
        for leaf in tree:
            idx, type_, x, y, z, r, p = leaf
            fp.write(f'{idx:d} {type_:d} {x:.2f} {y:.2f} {z:.2f} {r:.1f} {p:d}\n')


def runcmd(command, timeout=400):
    ret = subprocess.run(command, shell=True, encoding="utf-8", timeout=timeout)
    if ret.returncode == 0:
        print("success:", ret)
    else:
        print("error:", ret)


def sort_swc(input_swc, output_swc, root, vaa3d_path, plugin='sort_neuron_swc'):
    command = f'{vaa3d_path} /x {plugin} /f sort_swc /i {input_swc} /o {output_swc} /p 0 {root}'
    runcmd(command)


def get_root(tree):
    pos_idx = defaultdict(int)
    pos_count = defaultdict(int)
    for line in tree:
        idx, _, x, y, z, *_ = line
        pos = (x, y, z)
        pos_idx[pos] = idx
        pos_count[pos] += 1
    pos_count = sorted(pos_count.items(), key=lambda x: x[1], reverse=True)
    return pos_idx[pos_count[0][0]]


def batch(swcdir, outdir, vaa3d):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for swcfile in glob.glob(os.path.join(swcdir, '*.eswc')):
        tree = parse_swc(swcfile)
        root = get_root(tree)
        filename = os.path.split(swcfile)[-1]
        outfile = os.path.join(outdir, filename)
        sort_swc(swcfile, outfile, root, vaa3d)
        tree_new = parse_swc(outfile)
        idx, type_, x, y, z, r, p = tree_new[0]
        type_ = 1
        tree_new[0] = (idx, type_, x, y, z, r, p)
        write_swc(tree_new, outfile)


if __name__ == '__main__':
    swcdir = 'C:/Users/BrainCenter/Desktop/cannot_open_brain_for_sort'
    outdir = 'C:/Users/BrainCenter/Desktop/cannot_open_brain_sorted'
    vaa3d = 'D:/ChenXin/outreach/Vaa3D/Vaa3D_V3.601_Windows_MSVC_64bit/Vaa3D_V3.601_Windows_MSVC_64bit/vaa3d_msvc.exe'
    batch(swcdir, outdir, vaa3d)