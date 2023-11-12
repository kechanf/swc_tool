import queue

from seg_prune_v1 import *
from swc_base import *
import file_io
import platform
from shutil import copyfile
import re

v3d_path = r"D:/Vaa3D-x.1.1.2_Windows_64bit/Vaa3D-x"

def RebuildPointNearSoma_1891(point_l, dis_tho = 10):
    # soma_r = point_l.p[1].r
    # print(soma_r * alpha)
    # print(soma_r * alpha)
    flag = False
    for p in point_l.p:
        if (p.n == 0): continue
        if(p.n == 1):continue
        if (CalcswcPointDist(point_l.p[1], p) >= dis_tho): continue
        # if(len(p.s) == 1):continue
        if(not p.p == 1):
            point_l.p[p.p].s.remove(p.n)
            point_l.p[p.n].p = 1
            point_l.p[1].s.append(p.n)
        ps = p.s.copy()
        for s in ps:
            point_l.p[p.n].s.remove(s)
            point_l.p[s].p = 1
            point_l.p[1].s.append(s)
    # for i in range(10):
    #     print(i, point_l.p[i].n, point_l.p[i].r, point_l.p[i].p, point_l.p[i].s)
    return point_l

def PruneFiberNearSoma_1891(point_l, fiber_l, dis_tho = 10):
    fibers_pruned = 0
    # dis_tho = 40 # dis_pro * point_l.p[1].r
    # print(dis_tho)
    f_que = queue.Queue()
    for f in fiber_l.f:
        if (f.p[0] == 1): continue  # soma
        if (f.pruned): continue
        # print(f.dist2soma_max)
        if(len(f.sf) == 0 and f.dist2soma_max < dis_tho):
            # print("!!!!!!!!")
            f_que.put(f)

    while(not f_que.empty()):
        now_f = f_que.get()
        fiber_l = PruneOneFiber(fiber_l, now_f)
        # fiber_l.f[now_f.fn - 1].pruned = True
        # fiber_l.f[now_f.pf - 1].sf.remove(now_f.fn)
        fibers_pruned = fibers_pruned + 1
        if(len(fiber_l.f[now_f.pf - 1].sf) == 0 and fiber_l.f[now_f.pf - 1].dist2soma_max < dis_tho):
            f_que.put(fiber_l.f[now_f.pf - 1])
    if(fibers_pruned):
        print(f"Pruned {fibers_pruned} fibers in func PruneFiberNearSoma")
    return point_l, fiber_l

def main_soma_prune_1891(swc_path):
    point_l = Readswc_v2(swc_path)
    point_l = RebuildPointNearSoma_1891(point_l)
    point_l, fiber_l = BuildFiberList_v2(point_l)
    point_l, fiber_l = PruneFiberNearSoma_1891(point_l, fiber_l)
    stage3_path = r"D:\tracing_ws\1891\stage3"
    file_name, extension = os.path.splitext(os.path.basename(swc_path))
    out_path = stage3_path + "//" + file_name + ".swc"
    Writeswc_v2(out_path, point_l, fiber_l)

def Sortswc_1891(swc_path, soma_num=-1, out_path=""):
    def get_soma_num(swc_path):
        point_l = Readswc_v2(swc_path)
        # print(point_l.p[1].s)
        if(len(point_l.p[1].s) >= 3):return 1
        true_soma_num = 1
        true_soma_degree = len(point_l.p[1].s)
        q = queue.Queue()
        for s in point_l.p[1].s:
            q.put(s)
        while(not q.empty()):
            now_p = point_l.p[q.get()]
            if(1 + len(now_p.s) > true_soma_degree):
                true_soma_num = now_p.n
                true_soma_degree = 1 + len(now_p.s)
            for s in now_p.s:
                q.put(s)

        return true_soma_num

    def get_soma_num_v2(swc_path):
        file_name, extension = os.path.splitext(os.path.basename(swc_path))
        file_split = re.split("_", file_name)
        if(len(file_split) == 4):
            if(not (file_split[2][0] == "x") and file_split[2][0] == "y"):
                print(swc_path)
                exit(0)
            soma_x = float(file_split[2][1:])
            soma_y = float(file_split[3][1:])
            soma_z = float(file_split[1])
            print(soma_x, soma_y, soma_z)

            point_l = Readswc_v2(swc_path)
            print(point_l.p[1].n, point_l.p[1].x, point_l.p[1].y, point_l.p[1].z)
            print(abs(point_l.p[1].x - soma_x), abs(point_l.p[1].y - soma_y), abs(point_l.p[1].z - soma_z))
            if(abs(point_l.p[1].x - soma_x) <= 1
                    and abs(point_l.p[1].y - soma_y) <= 1
                    and abs(point_l.p[1].z - soma_z) <= 1):
                print("!!!!!!!!")
                return 1
            else:
                min_dis = 10000
                soma_num = 1
                for p in point_l.p:
                    temp_dis = (p.x - soma_x)**2 + (p.y - soma_y)**2 + (p.z - soma_z)**2
                    if(temp_dis < min_dis):
                        min_dis = 10000
                        soma_num = p.n
                return soma_num

    stage2_path = r"D:\tracing_ws\1891\stage2"
    file_name, extension = os.path.splitext(os.path.basename(swc_path))
    if (soma_num == -1):
        soma_num = get_soma_num_v2(swc_path)
        print(f"soma num {soma_num}")

    if(soma_num == 1):
        copyfile(swc_path, stage2_path + "//" + file_name + ".swc")
        return False
    else:
        print(f"{swc_path} processeed in sort_soma")
        sort_soma_path = r"D:\tracing_ws\1891\sort_soma"


        out_path = sort_soma_path + "//" + file_name + ".swc"
        copyfile(swc_path, sort_soma_path + "//" + file_name + "_origin.swc")
        if (platform.system() == "Windows"):
            subprocess.run(
                f'{v3d_path} /x sort /f sort_swc /i {swc_path} /o {out_path} /p 0 {soma_num}',
                stdout=subprocess.DEVNULL)  # 全路径
            copyfile(out_path, stage2_path + "//" + file_name + ".swc")
        else:
            print("failed for linux")
            exit(0)

    # out_path = check_0(swc_path, out_path)
    # out_path = check_soma(swc_path, out_path, soma_x, soma_y, soma_z)
    return True

from segmask import findAllFile
from multiprocessing import Pool
from tqdm import tqdm
if __name__ == '__main__':
    processed_num = 0
    # refine_path = r"D:\tracing_ws\1891\refinement"
    refine_path = r"D:\tracing_ws\1891\test"
    arglist = findAllFile(refine_path, ".swc")
    # for i in range(0, len(arglist)):
    #     # print(i)
    #     if(Sortswc_1891(arglist[i])):
    #         processed_num = processed_num + 1
    #     if(i % 10 == 0):
    #         print(f"{i} / {len(arglist)} checked in sort_soma")
    with Pool(12) as p:
        for res in tqdm(p.imap(Sortswc_1891, arglist), total=len(arglist)): pass
    print(f"{processed_num} files processed in sort_soma")
    with Pool(12) as p:
        for res in tqdm(p.imap(main_soma_prune_1891, arglist), total=len(arglist)): pass