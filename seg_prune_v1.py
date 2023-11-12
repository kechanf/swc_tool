import math
import time

import numpy as np

from swc_tool_lib import *
import queue
import seaborn as sns
from swc_base import *
def Readswc_v2(swc_name):
    point_l = swcP_list()
    with open(swc_name, 'r' ) as f:
        lines = f.readlines()

    swcPoint_number = -1
    # swcPoint_list = []
    point_list = []
    list_map = np.zeros(500000)

    for line in lines:
        if(line[0] == '#'):
            continue

        temp_line = line.split()
        # print(temp_line)
        point_list.append(temp_line)

        swcPoint_number = swcPoint_number + 1
        list_map[int(temp_line[0])] = swcPoint_number

    # print(point_list)
    swcPoint_number = 0
    for point in point_list:
        swcPoint_number = swcPoint_number + 1
        point[0] = swcPoint_number # int(point[0])
        point[1] = int(point[1])
        point[2] = float(point[2])
        point[3] = float(point[3])
        point[4] = float(point[4])
        point[5] = float(point[5])
        point[6] = int(point[6])
        if(point[6] == -1):
            pass
        else:
            point[6] = int(list_map[int(point[6])]) + 1

    # swcPoint_list.append(swcPoint(0,0,0,0,0,0,0)) # an empty point numbered 0
    point_l.p.append(swcPoint(0,0,0,0,0,0,0))

    for point in point_list:
        temp_swcPoint = swcPoint(point[0], point[1], point[2], point[3], point[4], point[5], point[6])
        point_l.p.append(temp_swcPoint)
    for point in point_list:
        temp_swcPoint = swcPoint(point[0], point[1], point[2], point[3], point[4], point[5], point[6])
        if not temp_swcPoint.p == -1:
            # parent = swcPoint_list[int(temp_swcPoint.p)]
            parent = point_l.p[int(temp_swcPoint.p)]
            parent.s.append(temp_swcPoint.n)
        if(point[0] == 1):
            point_l.p[int(point[0])].depth = 0
        else:
            point_l.p[int(point[0])].depth = parent.depth + 1
        # point_l.p.append(temp_swcPoint)
    # for i in range(1, 10):
    #     print(point_l.p[i].s)

    return point_l # (swcPoint_list)

def GetFiberDepth(point_l, fiber_l, n):
    # print(point_l.p[n].fn)
    # print(fiber_l.count)
    return fiber_l.f[point_l.p[n].fn - 1].depth

def AddNewFiber_v2(point_l, fiber_l, temp_point_l):
    fiber_l.count = fiber_l.count + 1
    if (fiber_l.count == 1):
        temp_fiber = swcFiber(fiber_l.count, 0)
    else:
        temp_fiber = swcFiber(fiber_l.count, 1 + GetFiberDepth(point_l, fiber_l, temp_point_l[0].p))
    for p in temp_point_l:
        temp_fiber.p.append(p.n)
        point_l.p[p.n].fn = temp_fiber.fn

    temp_fiber.l = len(temp_fiber.p)

    temp_fiber.UpdateParm(point_l)
    point_l.p[temp_point_l[0].n].ishead = True
    point_l.p[temp_point_l[-1].n].istail = True
    # print(f"head = {temp_point_l[0].n} tail = {temp_point_l[-1].n}")
    if(not temp_point_l[0].p == -1):
        # print(temp_point_l[0].p)
        temp_fiber.pf = point_l.p[temp_point_l[0].p].fn
        fiber_l.f[temp_fiber.pf - 1].sf.append(temp_fiber.fn)
    fiber_l.f.append(temp_fiber)
    return point_l, fiber_l


    # while (True):
    #     temp_swcFiber.sp.append(temp_swcPoint.n)
    #
    #     swcPoint_list[temp_swcPoint.n].fn = fiber_count
    #     temp_swcFiber.l = temp_swcFiber.l + 1
    #     # temp_swcFiber.Printswc()
    #
    #     if (not len(temp_swcPoint.s) == 1):  # no son or have sons
    #         break
    #     temp_swcPoint = swcPoint_list[temp_swcPoint.s[0]]

    # swcPoint_list = temp_swcFiber.UpdateEnd(swcPoint_list)
    # fiber_list.append(temp_swcFiber)
    # if(debug_mode == 1):
    #     temp_swcFiber.Printswc()
    # return swcPoint_list, fiber_list, fiber_count

def BuildFiberList_v2(point_l):
    fiber_l = swcF_list()

    p_que = queue.Queue()
    p_que.put(point_l.p[1]) # soma
    temp_point_l = []
    temp_point_l.append(point_l.p[1])
    point_l, fiber_l = AddNewFiber_v2(point_l, fiber_l, temp_point_l)

    while(not p_que.empty()):
        front_p = p_que.get()
        for s in front_p.s:
            now_p = point_l.p[s]
            temp_point_l = []
            while(True):
                temp_point_l.append(now_p)
                if(len(now_p.s) > 1):
                    p_que.put(now_p)
                    point_l, fiber_l = AddNewFiber_v2(point_l, fiber_l, temp_point_l)
                    break
                elif(now_p.s == []):
                    point_l, fiber_l = AddNewFiber_v2(point_l, fiber_l, temp_point_l)
                    break
                else:
                    # print(now_p.s)
                    now_p = point_l.p[now_p.s[0]]

    print(f"{len(fiber_l.f)} fibers build")
    fiber_l = RecalcDepthFiberList(point_l, fiber_l)
    # now_f = fiber_l.f[1]
    # while(True):
    #     print(now_f.fn, now_f.p, now_f.sf, now_f.depth, now_f.s_l, now_f.s_al)
    #     if(not now_f.sf):break
    #     now_f = fiber_l.f[now_f.sf[0] - 1]
    return point_l, fiber_l

def RecalcDepthFiberList(point_l, fiber_l, redepth = False):
    def getdepth(elem):
        return elem.depth
    def getal(elem):
        return fiber_l.f[elem - 1].s_al
    def getl(elem):
        return fiber_l.f[elem - 1].s_l

    temp_fiber_l = fiber_l.f.copy()
    temp_fiber_l.sort(key = getdepth,reverse = True)
    # for f in temp_fiber_l:
    #     print(f.depth)
    for f in temp_fiber_l:
        if(f.p[0] == 1):continue # soma
        if(f.sf):
            f.sf.sort(key = getal, reverse = True)
            f.sf.sort(key = getl, reverse = True)
            f.s_l = f.s_l + fiber_l.f[f.sf[0] - 1].s_l
            f.s_al = f.s_al + fiber_l.f[f.sf[0] - 1].s_al \
                     + CalcswcPointDist(point_l.p[f.p[-1]], point_l.p[fiber_l.f[f.sf[0] - 1].p[0]])
            fiber_l.f[f.fn - 1] = f
    if(redepth):
        temp_fiber_l.sort(key=getdepth, reverse=False)
        for f in temp_fiber_l:
            if(f.p[0] == 1):continue # soma
            if(f.sf):
                fiber_l.f[f.sf[0] - 1].depth = fiber_l.f[f.fn - 1].depth
    return fiber_l

def PruneOneFiber(fiber_l, f):
    if(not fiber_l.f[f.fn - 1].pruned):
        fiber_l.f[f.fn - 1].pruned = True
    if (f.fn in fiber_l.f[fiber_l.f[f.fn - 1].pf - 1].sf):
        fiber_l.f[fiber_l.f[f.fn - 1].pf - 1].sf.remove(f.fn)
    return fiber_l

def PruneFiberAndSon(fiber_l, f, fibers_pruned):
    temp_num = fibers_pruned
    f_que = queue.Queue()
    for sf in f.sf:
        f_que.put(fiber_l.f[sf - 1])
    while (not f_que.empty()):
        now_f = f_que.get()
        if (not now_f.pruned):
            fiber_l = PruneOneFiber(fiber_l, now_f)
            temp_num = temp_num + 1
            for sf in now_f.sf:
                f_que.put(fiber_l.f[sf - 1])
    return fiber_l, temp_num

def PruneFiberHead(point_l, fiber_l, r_tho = 2, l_tho = 1):
    fibers_pruned = 0
    for f in fiber_l.f:
        if(f.p[0] == 1):continue # soma
        if(f.pruned):continue
        if(not len(f.sf) == 0):continue

        if(f.avg_r > r_tho * point_l.p[point_l.p[f.p[0]].p].r):
            # fiber_l.f[f.fn - 1].pruned = True
            # print("??????????")
            fiber_l = PruneOneFiber(fiber_l, f)
            fibers_pruned = fibers_pruned + 1
            # fiber_l, fibers_pruned = PruneFiberAndSon(fiber_l, f, fibers_pruned)
        elif(f.l <= l_tho):
            # fiber_l.f[f.fn - 1].pruned = True
            # print("!!!!!!!!!")
            fiber_l = PruneOneFiber(fiber_l, f)
            fibers_pruned = fibers_pruned + 1
            # fiber_l, fibers_pruned = PruneFiberAndSon(fiber_l, f, fibers_pruned)

    print(f"Pruned {fibers_pruned} fibers in func PruneFiberHead")
    return point_l, fiber_l


def PruneFiberHead_v2(point_l, fiber_l, r_tho = 2, l_tho = 3, al_tho = 10):
    fibers_pruned = 0

    def getdepth(elem):
        return elem.depth

    temp_fiber_l = fiber_l.f.copy()
    temp_fiber_l.sort(key = getdepth,reverse = True)
    # for f in temp_fiber_l:
    #     print(f.depth)
    pre_d = 0
    for f in temp_fiber_l:
        if(f.p[0] == 1):continue # soma
        if(f.pruned):continue
        if(not len(f.sf) == 0):continue
        f = fiber_l.f[f.fn - 1]
        #
        # if(566 in f.p or 640 in f.p or 567 in f.p or 615 in f.p):
        #     print(f.p, f.sf)

        # if(f.avg_r > r_tho * point_l.p[point_l.p[f.p[0]].p].r):
        #     if(f.l >= l_tho*3 or f.al >= al_tho*3):
        #         continue
        #     # fiber_l.f[f.fn - 1].pruned = True
        #     # print("??????????")
        #     fiber_l = PruneOneFiber(fiber_l, f)
        #     fibers_pruned = fibers_pruned + 1
        #     # fiber_l, fibers_pruned = PruneFiberAndSon(fiber_l, f, fibers_pruned)
        if(f.l <= l_tho or f.al <= al_tho):
            # fiber_l.f[f.fn - 1].pruned = True
            # print("!!!!!!!!!")
            fiber_l = PruneOneFiber(fiber_l, f)
            fibers_pruned = fibers_pruned + 1
            if(not pre_d == f.depth):
                pre_d = f.depth
                point_l, fiber_l = MergeFiber(point_l, fiber_l)
                for ff in fiber_l.f: ff.UpdateParm(point_l)
            # fiber_l, fibers_pruned = PruneFiberAndSon(fiber_l, f, fibers_pruned)
    if(fibers_pruned):
        print(f"Pruned {fibers_pruned} fibers in func PruneFiberHead")
    return point_l, fiber_l

def PruneFiberTail(point_l, fiber_l, r_tho1 = 2, r_tho2 = 3, strict_prune = True):
    points_pruned = 0
    fiber_pruned = 0
    for f in fiber_l.f:
        p_list = f.p.copy()
        if (f.p[0] == 1): continue  # soma
        if(f.pruned):continue
        if((not strict_prune) and len(f.sf)):continue

        if(len(f.sf)): r_tho = r_tho1
        else: r_tho = r_tho2

        pruned_flag = False

        temp_l = f.l
        for i in range(f.l):

            inx = f.l - i
            now_p = point_l.p[f.p[inx - 1]]
            if(inx == 1):
                pass
            elif (inx == temp_l):
                if(now_p.r > r_tho * f.sum_r[inx - 2] / (temp_l - 1)):
                    print(f"now_p.r {now_p.r}")
                    print(f"f.sum_r[inx - 2] / (temp_l - 1) {f.sum_r[inx - 2] / (temp_l - 1)}")
                    print(f"r_tho {r_tho}")
                    temp_l = temp_l - 1
                    f.p = f.p[0: temp_l]
                    f.sum_r = f.sum_r[0: temp_l]
                    pruned_flag = True
                    # print(temp_l, f.l)
            else:
                acc_sum_r = f.sum_r[inx - 2] / (inx - 1)
                son_sum_r = (f.sum_r[temp_l - 1] - f.sum_r[inx - 1]) / (temp_l - inx)
                # print(inx, temp_l)
                if(acc_sum_r  * r_tho < now_p.r and son_sum_r * r_tho < now_p.r):

                    # print("case2")
                    # print(len(f.sf))
                    # print(f"acc_sum_r {acc_sum_r}")
                    # print(f"son_sum_r {son_sum_r}")
                    # print(f"now_p.r {now_p.r}")

                    temp_l = inx - 1
                    f.p = f.p[0: temp_l]
                    f.sum_r = f.sum_r[0: temp_l]
                    pruned_flag = True
                    # print(temp_l, f.l)

        f.l = temp_l
        for p in p_list:
            if(p not in f.p):
                point_l.p[p].pruned = True
                points_pruned = points_pruned + 1
        fiber_l.f[f.fn - 1] = f

        if(pruned_flag and strict_prune):
            f_que = queue.Queue()
            for sf in f.sf:
                f_que.put(fiber_l.f[sf - 1])
            while(not f_que.empty()):
                now_f = f_que.get()
                if(not now_f.pruned):
                    fiber_l.f[now_f.fn - 1].pruned = True
                    fiber_pruned = fiber_pruned + 1
                for sf in now_f.sf:
                    f_que.put(fiber_l.f[sf - 1])

    print(f"Pruned {points_pruned} points and {fiber_pruned} fibers in func PruneFiberTail")
    return point_l, fiber_l



def PruneFiberTail_v2(point_l, fiber_l, r_tho1 = 2, r_tho2 = 1.5):
    points_pruned = 0
    fiber_pruned = 0

    def getdepth(elem):
        return elem.depth

    temp_fiber_l = fiber_l.f.copy()
    temp_fiber_l.sort(key = getdepth,reverse = True)
    # for f in temp_fiber_l:
    #     print(f.depth)

    for f in temp_fiber_l:
        if (f.p[0] == 1): continue  # soma
        if(f.pruned):continue

        if(len(f.sf)): r_tho = r_tho1
        else: r_tho = r_tho2

        pruned_flag = False
        p_list = f.p.copy()
        temp_l = f.l
        for i in range(f.l):

            inx = f.l - i
            now_p = point_l.p[f.p[inx - 1]]
            if(inx == 1):
                pass
            elif (inx == temp_l):
                if(now_p.r > r_tho * f.sum_r[inx - 2] / (temp_l - 1)):
                    # print(f"now_p.r {now_p.r}")
                    # print(f"f.sum_r[inx - 2] / (temp_l - 1) {f.sum_r[inx - 2] / (temp_l - 1)}")
                    # print(f"r_tho {r_tho}")
                    temp_l = temp_l - 1
                    f.p = f.p[0: temp_l]
                    f.sum_r = f.sum_r[0: temp_l]
                    pruned_flag = True
                    # print(temp_l, f.l)
            else:
                acc_sum_r = f.sum_r[inx - 2] / (inx - 1)
                son_sum_r = (f.sum_r[temp_l - 1] - f.sum_r[inx - 1]) / (temp_l - inx)
                # print(inx, temp_l)
                if(acc_sum_r  * r_tho < now_p.r and son_sum_r * r_tho < now_p.r):

                    # print("case2")
                    # print(len(f.sf))
                    # print(f"acc_sum_r {acc_sum_r}")
                    # print(f"son_sum_r {son_sum_r}")
                    # print(f"now_p.r {now_p.r}")

                    temp_l = inx - 1
                    f.p = f.p[0: temp_l]
                    f.sum_r = f.sum_r[0: temp_l]
                    pruned_flag = True
                    # print(temp_l, f.l)

        f.l = temp_l
        for p in p_list:
            if(p not in f.p):
                point_l.p[p].pruned = True
                points_pruned = points_pruned + 1
        fiber_l.f[f.fn - 1] = f

        if(pruned_flag):
            f_que = queue.Queue()
            for sf in f.sf:
                f_que.put(fiber_l.f[sf - 1])
            while(not f_que.empty()):
                now_f = f_que.get()
                if(not now_f.pruned):
                    fiber_l.f[now_f.fn - 1].pruned = True
                    fiber_pruned = fiber_pruned + 1
                for sf in now_f.sf:
                    f_que.put(fiber_l.f[sf - 1])

    print(f"Pruned {points_pruned} points and {fiber_pruned} fibers in func PruneFiberTail")
    return point_l, fiber_l

def PruneFiberTail_v3(point_l, fiber_l, r_tho1 = 2.1, r_tho2 = 2.1):
    points_pruned = 0
    fiber_pruned = 0

    def getdepth(elem):
        return elem.depth

    temp_fiber_l = fiber_l.f.copy()
    temp_fiber_l.sort(key = getdepth,reverse = True)
    # for f in temp_fiber_l:
    #     print(f.depth)

    for f in temp_fiber_l:
        if (f.p[0] == 1): continue  # soma
        if(f.pruned):continue
        # if(264 in f.p):
        #     print(f.p, f.sf)

        if(len(f.sf)): r_tho = r_tho1
        else: r_tho = r_tho2

        pruned_flag = False
        p_list = f.p.copy()
        temp_l = f.l
        for i in range(f.l):
            # if(1424 in f.p):
            #     print(f"f.p {f.p}")
            inx = f.l - i
            now_p = point_l.p[f.p[inx - 1]]
            if(inx == 1):
                # if (743 in f.p):
                #     print(f"!!!!!")
                #     print(f.p)
                #     print(f"sum {f.sum_r[temp_l - 1] / temp_l}")
                #     print(f"r {point_l.p[point_l.p[f.p[0]].p].r}")
                #     print(f"sum {(f.sum_r[math.floor((temp_l - 1)/2) - 1]) / max(1, math.floor((temp_l - 1)/2))}")
                #     print((f.sum_r[math.floor((temp_l - 1)/2)] - 1), math.floor((temp_l - 1)/2))
                #     print(f"r {point_l.p[point_l.p[f.p[0]].p].r}")
                if(f.sum_r[temp_l - 1] / temp_l > r_tho * point_l.p[point_l.p[f.p[0]].p].r
                        or (f.sum_r[math.floor((temp_l - 1)/2) - 1]) /
                        max(1, math.floor((temp_l - 1)/2)) > r_tho * point_l.p[point_l.p[f.p[0]].p].r):
                    if(f.s_al < 40 and f.s_l < 15):
                        # if (743 in f.p):
                        #     print("???")
                        fiber_l = PruneOneFiber(fiber_l, f)
                        pruned_flag = True
                        break
            elif (inx == temp_l):
                if(now_p.r > r_tho * f.sum_r[inx - 2] / (temp_l - 1)):
                    temp_l = temp_l - 1
                    f.p = f.p[0: temp_l]
                    f.sum_r = f.sum_r[0: temp_l]
                    pruned_flag = True
            else:
                acc_sum_r = f.sum_r[inx - 2] / (inx - 1)
                son_sum_r = (f.sum_r[temp_l - 1] - f.sum_r[inx - 1]) / (temp_l - inx)
                # print(inx, temp_l)
                if(acc_sum_r  * r_tho < now_p.r and son_sum_r * r_tho < now_p.r): # overprune??
                    if(i < 5):
                        temp_l = inx - 1
                        f.p = f.p[0: temp_l]
                        f.sum_r = f.sum_r[0: temp_l]
                        pruned_flag = True
                    # print(temp_l, f.l)
        f.l = temp_l
        for p in p_list:
            if(p not in f.p):
                point_l.p[p].pruned = True
                points_pruned = points_pruned + 1
        f.UpdateParm(point_l)
        fiber_l.f[f.fn - 1] = f
        if(pruned_flag):
            f_que = queue.Queue()
            for sf in f.sf:
                f_que.put(fiber_l.f[sf - 1])
            while(not f_que.empty()):
                now_f = f_que.get()
                if(not now_f.pruned):
                    fiber_l.f[now_f.fn - 1].pruned = True
                    fiber_pruned = fiber_pruned + 1
                for sf in now_f.sf:
                    f_que.put(fiber_l.f[sf - 1])
    if(points_pruned or fiber_pruned):
        print(f"Pruned {points_pruned} points and {fiber_pruned} fibers in func PruneFiberTail")
    return point_l, fiber_l

def PruneFiberNearSoma(point_l, fiber_l, dis_pro = 5):
    fibers_pruned = 0
    dis_tho = dis_pro * point_l.p[1].r
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

    print(f"Pruned {fibers_pruned} fibers in func PruneFiberNearSoma")
    return point_l, fiber_l

def PruneFiberNearSoma_v2(point_l, fiber_l, dis_tho = 10):
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

def PruneFiberAngle(point_l, fiber_l, angle_tho = math.pi * 70 / 180):
    points_pruned = 0
    for f in fiber_l.f:
        # print(f.fn, f.p)
        if (f.p[0] == 1): continue  # soma
        if (f.pruned): continue
        if (not len(f.sf) == 0):continue
        pruned_flag = False
        p_list = f.p.copy()
        for p in p_list:
            if(pruned_flag):
                # print(p_list)
                # print(p)
                point_l.p[p].pruned = True
                fiber_l.f[f.fn - 1].p.remove(p)
                points_pruned = points_pruned + 1
                continue
            now_p = point_l.p[p]
            if(now_p.pruned):continue
            if(not now_p.s):continue
            va = [now_p.x - point_l.p[now_p.p].x,
                  now_p.y - point_l.p[now_p.p].y,
                  now_p.z - point_l.p[now_p.p].z]
            vb = [point_l.p[now_p.s[0]].x - now_p.x,
                  point_l.p[now_p.s[0]].y - now_p.y,
                  point_l.p[now_p.s[0]].z - now_p.z]
            temp_angle = CalcVectorAngle(va, vb)
            if(temp_angle > angle_tho):
                pruned_flag = True
        if(pruned_flag):
            fiber_l.f[f.fn - 1].UpdateParm(point_l)

    if(points_pruned):
        print(f"Pruned {points_pruned} points in func PruneFiberAngle")
    return point_l, fiber_l

def PruneFiberParallel(point_l, fiber_l, angle_tho = math.pi * 30 / 180):
    fiber_pruned = 0
    def getdepth(elem):
        return elem.depth

    temp_fiber_l = fiber_l.f.copy()
    temp_fiber_l.sort(key=getdepth, reverse=False)

    for f in temp_fiber_l:
        if (f.p[0] == 1): continue  # soma
        if(f.pruned):continue
        if(len(f.sf) <= 1):continue
        flag = False
        for i in range(1, len(f.sf)):
            if(fiber_l.f[f.sf[i] - 1].sf):
                flag = True
                break
        if(flag == False and fiber_l.f[f.sf[0] - 1].sf):
            # if (482 in f.p or 483 in f.p):
            #     print("!!!")
            #     print(f.p)
            #     for sf in f.sf:
            #         print(fiber_l.f[sf - 1].p)
            v_head = [point_l.p[f.p[-1]].x - point_l.p[fiber_l.f[f.sf[0] - 1].p[0]].x,
                      point_l.p[f.p[-1]].y - point_l.p[fiber_l.f[f.sf[0] - 1].p[0]].y,
                      point_l.p[f.p[-1]].z - point_l.p[fiber_l.f[f.sf[0] - 1].p[0]].z]
            v_tail = [point_l.p[f.p[-1]].x - point_l.p[fiber_l.f[f.sf[0] - 1].p[-1]].x,
                      point_l.p[f.p[-1]].y - point_l.p[fiber_l.f[f.sf[0] - 1].p[-1]].y,
                      point_l.p[f.p[-1]].z - point_l.p[fiber_l.f[f.sf[0] - 1].p[-1]].z]
            sf = f.sf.copy()
            for i in range(1, len(sf)):
                if(fiber_l.f[sf[i] - 1].l >= 8):continue
                v_a = [point_l.p[f.p[-1]].x - point_l.p[fiber_l.f[sf[i] - 1].p[0]].x,
                       point_l.p[f.p[-1]].y - point_l.p[fiber_l.f[sf[i] - 1].p[0]].y,
                       point_l.p[f.p[-1]].z - point_l.p[fiber_l.f[sf[i] - 1].p[0]].z]
                v_b = [point_l.p[f.p[-1]].x - point_l.p[fiber_l.f[sf[i] - 1].p[-1]].x,
                       point_l.p[f.p[-1]].y - point_l.p[fiber_l.f[sf[i] - 1].p[-1]].y,
                       point_l.p[f.p[-1]].z - point_l.p[fiber_l.f[sf[i] - 1].p[-1]].z]
                # if (482 in f.p or 483 in f.p):
                #     print(CalcVectorAngle(v_head, v_a)/math.pi*180)
                #     print(CalcVectorAngle(v_tail, v_b)/math.pi*180)
                if(CalcVectorAngle(v_head, v_a) < angle_tho
                        and CalcVectorAngle(v_tail, v_b) < angle_tho):
                    fiber_l = PruneOneFiber(fiber_l, fiber_l.f[sf[i] - 1])
                    # point_l, fiber_l = MergeFiber(point_l, fiber_l)
                    fiber_pruned = fiber_pruned + 1
                    # if (482 in f.p or 483 in f.p):
                    #     print("done")
    if(fiber_pruned):
        print(f"Pruned {fiber_pruned} fibers in func PruneFiberParallel")
    return point_l, fiber_l

def PruneFiberrstd(point_l, fiber_l, i_std_limit):
    point_l, fiber_l = MergeFiber(point_l, fiber_l)
    for ff in fiber_l.f: ff.UpdateParm(point_l)

    fiber_pruned = 0
    def getdepth(elem):
        return elem.depth

    temp_fiber_l = fiber_l.f.copy()
    temp_fiber_l.sort(key=getdepth, reverse=True)

    for f in temp_fiber_l:
        f = fiber_l.f[f.fn - 1]
        if(f.pruned):continue
        # if (324 in f.p or 240 in f.p):
        #     print(f.r_mean, f.r_std)
        #     print(f.p, f.pruned, f.sf)
        #     for sf in f.sf:
        #         print(fiber_l.f[sf - 1].p)
        if (f.p[0] == 1): continue  # soma
        if(len(f.sf)):continue

        #
        # if (324 in f.p):
        #     print("!!!!!!!!")
        #     print(f.r_mean, f.r_std)
        #     print(f.p)
        #     print("!!!!!!!!")
        prune_flag = False
        outline_p_count = 0
        for p in f.p:
            # if(point_l.p[p].r >= f.r_std*2 + f.r_mean):
            #     prune_flag = True
            #     break
            if(point_l.p[p].i >= f.i_mean + f.i_std*2 or point_l.p[p].i <= f.i_mean - f.i_std*2 or
                    point_l.p[p].i <= i_std_limit[0] or point_l.p[p].i >= i_std_limit[1]):
                outline_p_count = outline_p_count + 1
        if(outline_p_count > len(f.p) * 0.5):
            prune_flag = True
        if(prune_flag):
            # if(324 in f.p):
            #     print("!!!!!!!!!!!")
            fiber_l = PruneOneFiber(fiber_l, f)
            point_l, fiber_l = MergeFiber(point_l, fiber_l)
            for ff in fiber_l.f:
                if(not f.pruned): ff.UpdateParm(point_l)
            fiber_pruned = fiber_pruned + 1


    if(fiber_pruned):
        print(f"Pruned {fiber_pruned} fibers in func PruneFiberrstd")
    return point_l, fiber_l


def PruneFiber_v1(point_l, fiber_l):
    # point_l, fiber_l = PruneFiberTail_v3(point_l, fiber_l)
    # point_l, fiber_l = MergeFiber(point_l, fiber_l)
    #
    # point_l, fiber_l = PruneFiberAngle(point_l, fiber_l)
    # point_l, fiber_l = MergeFiber(point_l, fiber_l)
    #
    # point_l, fiber_l = PruneFiberNearSoma_v2(point_l, fiber_l)
    # point_l, fiber_l = MergeFiber(point_l, fiber_l)
    # #
    # fiber_l = RecalcDepthFiberList(point_l, fiber_l, True)
    # #
    # point_l, fiber_l = PruneFiberHead_v2(point_l, fiber_l, l_tho=2, al_tho=20)
    # point_l, fiber_l = MergeFiber(point_l, fiber_l)
    #
    # fiber_l = RecalcDepthFiberList(point_l, fiber_l, True)

    return point_l, fiber_l

def PruneFiber_v2(point_l, fiber_l):
    point_l, fiber_l = PruneFiberTail_v3(point_l, fiber_l)
    point_l, fiber_l = MergeFiber(point_l, fiber_l)

    point_l, fiber_l = PruneFiberAngle(point_l, fiber_l)
    point_l, fiber_l = MergeFiber(point_l, fiber_l)

    point_l, fiber_l = PruneFiberNearSoma_v2(point_l, fiber_l)
    point_l, fiber_l = MergeFiber(point_l, fiber_l)
    #
    fiber_l = RecalcDepthFiberList(point_l, fiber_l, True)

    point_l, fiber_l = PruneFiberHead_v2(point_l, fiber_l, l_tho=2, al_tho=20)
    point_l, fiber_l = MergeFiber(point_l, fiber_l)

    fiber_l = RecalcDepthFiberList(point_l, fiber_l, True)
    # #
    point_l, fiber_l = PruneFiberParallel(point_l, fiber_l)
    point_l, fiber_l = MergeFiber(point_l, fiber_l)

    return point_l, fiber_l
def MergeFiber(point_l, fiber_l):
    merge_count = 0
    for f in fiber_l.f:
        if(f.pruned):continue
        if (f.p[0] == 1): continue  # soma
        if(len(f.sf) == 1):
            f_que = queue.Queue()
            f_que.put(fiber_l.f[f.sf[0] - 1])
            while(not f_que.empty()):
                now_f = f_que.get()
                fiber_l.f[now_f.fn-1].pruned = True
                point_l.p[f.p[-1]].istail = False
                f.p = f.p + now_f.p
                f.sf = now_f.sf
                f.UpdateParm(point_l)
                point_l.p[f.p[-1]].istail = True
                for p in now_f.p:
                    point_l.p[p].fn = f.fn
                merge_count = merge_count + 1
                if(len(now_f.sf) == 1):
                    f_que.put(fiber_l.f[now_f.sf[0] - 1])
                fiber_l.f[f.fn - 1] = f


    # if(merge_count):
    #     print(f"merge {merge_count} fibers")
    return point_l, fiber_l

def Writeswc_v2(filepath, point_l, fiber_l, reversal=False, limit=[1000, 1000, 1000], overlay=False, number_offset=0):
    lines = []
    for temp_p in point_l.p:
        if(temp_p.n == 0):continue
        if(temp_p.fn == -1):continue
        if(fiber_l.f[temp_p.fn - 1].pruned):continue
        if(temp_p.pruned):continue
        # if(temp_p.n not in fiber_l.f[temp_p.fn - 1].p):continue
        # if(temp_p.ishead): continue
        # print(temp_p.n)
        if (reversal):
            line = "%d %d %f %f %f %f %d\n" % (
                temp_p.n + number_offset, temp_p.si, temp_p.x,
                limit[1] - temp_p.y,
                temp_p.z, temp_p.r, temp_p.p + number_offset
            )
        else:
            line = "%d %d %f %f %f %f %d\n" % (
                temp_p.n + number_offset, temp_p.si, temp_p.x,
                temp_p.y,
                temp_p.z, temp_p.r, temp_p.p + number_offset
            )
        lines.append(line)
    if (overlay and os.path.exists(filepath)):
        # print("!!!!!!")
        os.remove(filepath)
    file_handle = open(filepath, mode="a")
    file_handle.writelines(lines)
    file_handle.close()

def main():
    processed_num = 0
    parm_list = []
    home_path = r"D:\tracing_ws\hunam_brain\recons"
    for filepath, dirnames, filenames in os.walk(home_path):
        for filename in filenames:
            if("test" in filename):continue
            if (".swc" not in filename): continue
            if("ini" in filename):continue
            if("prunedv1" in filename):continue

            file_AbsolutePath = os.path.join(filepath, filename)
            print("Processing file %s ..." % (file_AbsolutePath))

            point_l = Readswc_v2(os.path.join(filepath, filename))
            point_l, fiber_l = BuildFiberList_v2(point_l)
            # print(len(fiber_l.f))


            point_l, fiber_l = PruneFiber(point_l, fiber_l)

            # test_file_path = r"D:\tracing_ws\human_brain\test\test.swc"

            Writeswc_v2(file_AbsolutePath[0: -4] + "_prunedv1.swc", point_l, fiber_l)

            processed_num = processed_num + 1
            print(str(processed_num) + "/" + str(len(filenames)) + " files done!\n")

def RebuildPointNearSoma(point_l, alpha = 2):
    soma_r = point_l.p[1].r
    # print(soma_r * alpha)
    # for i in range(1, 10):

    for p in point_l.p:
        # if(p.p == 1):continue
        if(p.n == 0):continue
        if(p.n == 1):continue
        if(not len(p.s)):continue
        if(CalcswcPointDist(point_l.p[1], p) >= soma_r * alpha): continue

        for i in range(1, len(p.s)):
            point_l.p[p.s[i]].p = 1
            point_l.p[1].s.append(p.s[i])
        p.s = [p.s[0]]
        point_l.p[point_l.p[p.n].p].s.remove(p.n)
        point_l.p[p.n] = p
    # for i in range(10):
    #     print(i, point_l.p[i].n, point_l.p[i].r, point_l.p[i].p, point_l.p[i].s)
    return point_l


def RebuildPointNearSoma_v2(point_l, soma_r, alpha = 2):
    # soma_r = point_l.p[1].r
    # print(soma_r * alpha)
    # print(soma_r * alpha)
    for p in point_l.p:
        if (p.n == 0): continue
        if(p.n == 1):continue
        if (CalcswcPointDist(point_l.p[1], p) >= soma_r * alpha): continue
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


def PruneFiberMain_v1(in_path, outpath = None):
    point_l = Readswc_v2(in_path)
    point_l = RebuildPointNearSoma_v2(point_l)
    point_l, fiber_l = BuildFiberList_v2(point_l)
    if(not (len(point_l.p) == 1 and len(fiber_l.f) == 1)):
        point_l, fiber_l = PruneFiber_v1(point_l, fiber_l)
    if(outpath):
        Writeswc_v2(outpath, point_l, fiber_l)
    else:
        Writeswc_v2(in_path[0: -4] + "_prunedv1.swc", point_l, fiber_l)

def PruneFiberMain_v2(in_path, soma_r, img, outpath = None):
    point_l = Readswc_v2(in_path)
    point_l = RebuildPointNearSoma_v2(point_l, soma_r)
    point_l = GetPointI(point_l, img)
    point_l, fiber_l = BuildFiberList_v2(point_l)
    if(not (len(point_l.p) == 1 and len(fiber_l.f) == 1)):
        point_l, fiber_l = PruneFiber_v2(point_l, fiber_l)
    if(outpath):
        Writeswc_v2(outpath, point_l, fiber_l)
    else:
        Writeswc_v2(in_path[0: -4] + "_prunedv2.swc", point_l, fiber_l)

def check_resample(resample_swc_path):
    with open(resample_swc_path, "r") as f:
        lines = f.readlines()
        data = []
        if(lines[0][0] == '#' and lines[1][0] == '2'):
            lines[0] = lines[0][lines[0].find(",pid")+4:]
        for i in lines:
            data.append(i)  # 记录每一行
    # write
    with open(resample_swc_path, "w") as f:
        for i in data:
            f.writelines(i)

import subprocess
def Prune2ndMain(in_path, origin_in_path, soma_r, img, outpath = None):
    in_resample_swc_path = f"{in_path[:-4]}_resample.swc"
    if (os.path.exists(in_resample_swc_path)): os.remove(in_resample_swc_path)
    resample_step = 2
    subprocess.run(
        f'D:/Vaa3D-x.1.1.2_Windows_64bit/Vaa3D-x /x resample_swc /f resample_swc /i {in_path} /o {in_resample_swc_path} /p {resample_step}',
        stdout=subprocess.DEVNULL)  # 全路径
    check_resample(in_resample_swc_path)

    origin_resample_swc_path = f"{origin_in_path[:-4]}_origin_resample.swc"
    if (os.path.exists(origin_resample_swc_path)): os.remove(origin_resample_swc_path)
    resample_step = 2
    subprocess.run(
        f'D:/Vaa3D-x.1.1.2_Windows_64bit/Vaa3D-x /x resample_swc /f resample_swc /i {origin_in_path} /o {origin_resample_swc_path} /p {resample_step}',
        stdout=subprocess.DEVNULL)  # 全路径
    check_resample(origin_resample_swc_path)

    point_l = Readswc_v2(in_resample_swc_path)
    point_l = RebuildPointNearSoma_v2(point_l, soma_r)
    point_l = GetPointI(point_l, img)
    point_l, fiber_l = BuildFiberList_v2(point_l)
    r_std_limit, i_std_limit, angle_limit, r_diff_limit, i_diff_limit = CalcFiberParm(point_l, fiber_l, soma_r)


    point_l = Readswc_v2(origin_resample_swc_path)
    point_l = RebuildPointNearSoma_v2(point_l, soma_r)
    point_l = GetPointI(point_l, img)
    point_l, fiber_l = BuildFiberList_v2(point_l)
    if (not (len(point_l.p) == 1 and len(fiber_l.f) == 1)):
        point_l, fiber_l = Prune2nd(point_l, fiber_l, soma_r, r_std_limit, i_std_limit, angle_limit, r_diff_limit, i_diff_limit)
        point_l, fiber_l = MergeFiber(point_l, fiber_l)
        fiber_l = RecalcDepthFiberList(point_l, fiber_l, True)

        point_l, fiber_l = PruneFiberTail_v3(point_l, fiber_l)
        point_l, fiber_l = MergeFiber(point_l, fiber_l)

        point_l, fiber_l = PruneFiberAngle(point_l, fiber_l)
        point_l, fiber_l = MergeFiber(point_l, fiber_l)

        point_l, fiber_l = PruneFiberNearSoma_v2(point_l, fiber_l)
        point_l, fiber_l = MergeFiber(point_l, fiber_l)
        #
        fiber_l = RecalcDepthFiberList(point_l, fiber_l, True)

        point_l, fiber_l = PruneFiberHead_v2(point_l, fiber_l, l_tho=2, al_tho=20)
        point_l, fiber_l = MergeFiber(point_l, fiber_l)

        fiber_l = RecalcDepthFiberList(point_l, fiber_l, True)
        # #
        point_l, fiber_l = PruneFiberParallel(point_l, fiber_l)
        point_l, fiber_l = MergeFiber(point_l, fiber_l)
        fiber_l = RecalcDepthFiberList(point_l, fiber_l, True)

        # point_l, fiber_l = PruneFiberNearSoma_v2(point_l, fiber_l)
        point_l, fiber_l = PruneFiberrstd(point_l, fiber_l, i_std_limit)
        Writeswc_v2(in_path[0: -4] + "_prunedv2.2.swc", point_l, fiber_l)

    resample_swc_path = f"{in_path[0: -4]}_prunedv2.2_resample.swc"
    if(os.path.exists(resample_swc_path)):os.remove(resample_swc_path)
    resample_step = 2
    import platform
    if(platform.system() == "Windows"):
        subprocess.run(
            f'D:/Vaa3D-x.1.1.2_Windows_64bit/Vaa3D-x /x resample_swc /f resample_swc /i {in_path[0: -4] + "_prunedv2.2.swc"} /o {resample_swc_path} /p {resample_step}',
            stdout=subprocess.DEVNULL)  # 全路径
    else:
        print("failed for linux")
        exit(0)
    check_resample(resample_swc_path)

    # if (os.path.exists(in_resample_swc_path)): os.remove(in_resample_swc_path)
    if(os.path.exists(in_path[0: -4] + "_prunedv2.2.swc")):os.remove(in_path[0: -4] + "_prunedv2.2.swc")
    if (os.path.exists(origin_resample_swc_path)): os.remove(origin_resample_swc_path)

def Prune2nd(point_l, fiber_l, soma_r, r_std_limit, i_std_limit, angle_limit, r_diff_limit, i_diff_limit):
    fiber_pruned = 0
    def getdepth(elem):
        return elem.depth

    temp_fiber_l = fiber_l.f.copy()
    temp_fiber_l.sort(key=getdepth, reverse=True)

    for f in temp_fiber_l:
        f = fiber_l.f[f.fn - 1]
        if (f.p[0] == 1): continue
        # if(len(f.sf)):continue
        temp_r = []
        temp_i = []
        temp_vector = [0, 0, 0]
        for p in f.p:
            temp_vector = temp_vector + np.array([point_l.p[p].x, point_l.p[p].y, point_l.p[p].z]) \
                          - np.array([point_l.p[point_l.p[f.p[0]].p].x,
                                      point_l.p[point_l.p[f.p[0]].p].y,
                                      point_l.p[point_l.p[f.p[0]].p].z])
            if(CalcswcPointDist(point_l.p[p], point_l.p[1]) < soma_r * 2):continue
            temp_r.append(point_l.p[p].r)
            temp_i.append(point_l.p[p].i)
        if (temp_r and temp_i):
            fiber_l.f[f.fn - 1].r_std = np.std(temp_r)
            fiber_l.f[f.fn - 1].r_mean = np.mean(temp_r)
            fiber_l.f[f.fn - 1].i_std = np.std(temp_i)
            fiber_l.f[f.fn - 1].i_mean = np.mean(temp_i)
            fiber_l.f[f.fn - 1].vector = list(temp_vector)

    for f in temp_fiber_l:
        f = fiber_l.f[f.fn - 1]
        if (f.p[0] == 1): continue
        if (len(f.sf)): continue
        prune_flag = False
        # print(r_std_limit, i_std_limit, angle_limit, r_diff_limit, i_diff_limit)
        # if(f.r_std >= r_std_limit[1] or f.i_std >= i_std_limit[1]):prune_flag = True
        if(not fiber_l.f[f.pf - 1].p[0]==1):
            # if(CalcVectorAngle(f.vector, fiber_l.f[f.pf - 1].vector) >= angle_limit[1]):prune_flag = True
            # if(f.r_mean - fiber_l.f[f.pf - 1].r_mean <= r_diff_limit[0] or
            #     f.r_mean - fiber_l.f[f.pf - 1].r_mean >= r_diff_limit[1] or
            #     f.i_mean - fiber_l.f[f.pf - 1].i_mean <= i_diff_limit[0] or
            #     f.i_mean - fiber_l.f[f.pf - 1].i_mean >= i_diff_limit[1]):prune_flag = True
            # print(r_diff_limit, r_diff_limit)
            if(
                    f.r_mean - fiber_l.f[f.pf - 1].r_mean <= r_diff_limit[0] or
                    f.r_mean - fiber_l.f[f.pf - 1].r_mean >= r_diff_limit[1] or
                    f.i_mean - fiber_l.f[f.pf - 1].i_mean <= i_diff_limit[0] or
                    f.i_mean - fiber_l.f[f.pf - 1].i_mean >= i_diff_limit[1]
            ):
                # print(f.r_mean, fiber_l.f[f.pf - 1].r_mean, r_diff_limit)
                prune_flag = True
            # pass
        if(prune_flag):
            fiber_l = PruneOneFiber(fiber_l, f)
            point_l, fiber_l = MergeFiber(point_l, fiber_l)
            for ff in fiber_l.f: ff.UpdateParm(point_l)
            fiber_pruned = fiber_pruned + 1
            # else:
            #     print(np.std(temp_r), r_std_limit, np.std(temp_i), i_std_limit)
    if(fiber_pruned):
        print(f"Pruned {fiber_pruned} fibers in func Prune2nd")


    # point_l, fiber_l = PruneFiberTail_v3(point_l, fiber_l)
    # point_l, fiber_l = MergeFiber(point_l, fiber_l)
    # point_l, fiber_l = PruneFiberAngle(point_l, fiber_l)
    # point_l, fiber_l = MergeFiber(point_l, fiber_l)
    return point_l, fiber_l


def GetPointI(point_l, img):
    for p in point_l.p:
        x = round(p.x)
        y = round(p.y)
        z = round(p.z)
        r = math.ceil(p.r)
        temp_i = []
        for i in range(x-r, x+r):
            for j in range(y-r, y+r):
                for k in range(z-r, z+r):
                    if(i >= img.shape[2] or j >= img.shape[1] or k >= img.shape[0]):continue
                    if((i-x)**2 + (j-y)**2 + (k-z)**2 <= r**2):
                        temp_i.append(img[k][j][i])
        if(temp_i):
            point_l.p[p.n].i = np.mean(temp_i)
        else:
            point_l.p[p.n].i = 0
    return point_l



def CalcFiberParm(point_l, fiber_l, soma_r):
    r_std_list_positive = []
    # r_std_list_negative = []
    r_mean_list_positive = []
    # r_mean_list_negative = []
    i_std_list_positive = []
    # i_std_list_negative = []
    i_mean_list_positive = []
    # i_mean_list_negative = []
    angle_list = []
    r_diff_list = []
    i_diff_list = []

    for f in fiber_l.f:
        if (f.p[0] == 1): continue  # soma
        # print("!!!")
        temp_r = []
        temp_i = []
        temp_vector = np.array([0,0,0])
        for p in f.p:
            temp_vector = temp_vector + np.array([point_l.p[p].x, point_l.p[p].y, point_l.p[p].z])\
                          - np.array([point_l.p[point_l.p[f.p[0]].p].x,
                                      point_l.p[point_l.p[f.p[0]].p].y,
                                      point_l.p[point_l.p[f.p[0]].p].z])
            if(CalcswcPointDist(point_l.p[p], point_l.p[1]) < soma_r * 2):continue
            temp_r.append(point_l.p[p].r)
            temp_i.append(point_l.p[p].i)
        # print(temp_r, temp_i)
        # print(np.std(temp_r), np.std(temp_i))
        if(temp_r and temp_i):
            # if(not f.pruned):
            r_std_list_positive.append(np.std(temp_r))
            fiber_l.f[f.fn - 1].r_std = np.std(temp_r)
            r_mean_list_positive.append(np.mean(temp_r))
            fiber_l.f[f.fn - 1].r_mean = np.mean(temp_r)
            i_std_list_positive.append(np.std(temp_i))
            fiber_l.f[f.fn - 1].i_std = np.std(temp_i)
            i_mean_list_positive.append(np.mean(temp_i))
            fiber_l.f[f.fn - 1].i_mean = np.mean(temp_i)

            fiber_l.f[f.fn - 1].vector = list(temp_vector)
            # print(temp_vector)

            # else:
            #     r_std_list_negative.append(np.std(temp_r))
            #     r_mean_list_negative.append(np.mean(temp_r))
            #     i_std_list_negative.append(np.std(temp_i))
            #     i_mean_list_negative.append(np.mean(temp_i))



    for f in fiber_l.f:
        if (f.p[0] == 1): continue  # soma
        for sf in f.sf:
            # print(f.vector, fiber_l.f[sf - 1].vector)
            angle_list.append(CalcVectorAngle(f.vector, fiber_l.f[sf - 1].vector))
            r_diff_list.append(fiber_l.f[sf - 1].r_mean - f.r_mean)
            i_diff_list.append(fiber_l.f[sf - 1].i_mean - f.i_mean)

    #array

    # print("!!!!!!!!!")
    # print(r_std_list_positive)
    # print(np.mean(r_std_list_positive), np.std(r_std_list_positive))
    # print(i_std_list_positive)
    # print(np.mean(i_std_list_positive),np.std(i_std_list_positive))
    #
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    plt.subplot(2, 3, 1)
    p = sns.histplot(r_std_list_positive, bins=100)
    p.set_title("r_std_list_positive")
    # # # plt.subplot(4, 2, 2)
    # # # # sns.histplot(r_std_list_negative)
    # plt.subplot(2, 3, 2)
    # p = sns.histplot(r_mean_list_positive, bins=100)
    # p.set_title("r_mean_list_positive")
    # # # plt.subplot(4, 2, 4)
    # # sns.histplot(r_mean_list_negative)
    plt.subplot(2, 3, 2)
    p = sns.histplot(i_std_list_positive, bins=100)
    p.set_title("i_std_list_positive")
    # # # plt.subplot(4, 2, 6)
    # # # sns.histplot(i_std_list_negative)
    # plt.subplot(2, 3, 4)
    # p = sns.histplot(i_mean_list_positive, bins=100)
    # p.set_title("i_mean_list_positive")
    # # # plt.subplot(4, 2, 8)
    # # # sns.histplot(i_mean_list_negative)
    plt.subplot(2, 3, 3)
    p = sns.histplot(angle_list, bins=100)
    p.set_title("angle_list")

    plt.subplot(2, 3, 4)
    p = sns.histplot(r_diff_list, bins=100)
    p.set_title("r_diff_list")

    plt.subplot(2, 3, 5)
    p = sns.histplot(i_diff_list, bins=100)
    p.set_title("i_diff_list")


    r_std_limit = [0, np.mean(r_std_list_positive)+ 2*np.std(r_std_list_positive)] #
    i_std_limit = [0, np.mean(i_std_list_positive)+ 2*np.std(i_std_list_positive)] #
    # print(r_std_limit, i_std_limit)
    angle_limit = [0, np.mean(angle_list) + 2*np.std(angle_list)]
    r_diff_limit = [-math.fabs(10*min(r_diff_list)),np.mean(r_diff_list) + 2*np.std(r_diff_list)]
    i_diff_limit = [-math.fabs(10*min(i_diff_list)), np.mean(i_diff_list) + 2*np.std(i_diff_list)]
    # print(r_std_limit, i_std_limit, angle_limit, r_diff_limit, i_diff_limit)
    # print(f"{np.mean(r_diff_list)} np.mean(r_diff_list) {np.std(r_diff_list)} np.std(r_diff_list)")
    # plt.show()
    return r_std_limit, i_std_limit, angle_limit, r_diff_limit, i_diff_limit


if __name__ == '__main__':
    pass