import time

import file_io
from swc_base import *
from recons_human import *
# import cv2
import skimage
import cc3d
import warnings
import uuid
warnings.filterwarnings("ignore", category=UserWarning, module="libtiff")

if(uuid.UUID(int=uuid.getnode()).hex[-12:] == "bc6ee23a04f7"
        or os.getlogin() == '12626'):
    project_root = r"D:\tracing_ws\dataset"
    v3d_path = r"D:/Vaa3D-x.1.1.2_Windows_64bit/Vaa3D-x"
else:
    project_root = "Z://SEU-ALLEN//Users//KaifengChen//human_brain"
    v3d_path = r""
mask_root = project_root + "//img//mask"


def Eswc2swc(eswc_path, out_path=""):
    if (not out_path):
        out_path = eswc_path[:-4] + ".swc"
    if (os.path.exists(out_path)):
        os.remove(out_path)
    with open(eswc_path, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        if (line[0] == '#'):
            continue
        temp_line = line.split()[:7]
        lines[i] = "%s %s %s %s %s %s %s\n" % (
            temp_line[0], temp_line[1], temp_line[2], temp_line[3], temp_line[4], temp_line[5], temp_line[6]
        )
        # print(temp_line, len(temp_line))

    file_handle = open(out_path, mode="a")
    file_handle.writelines(lines)
    file_handle.close()

    return out_path


from sort_swc import *


# Unable to handle the problem of annotation of human brain
def Sortswc_Gaoyu(swc_path, out_path=""):
    if (not out_path):
        out_path = swc_path[0:-4] + "_sort.swc"
    if (os.path.exists(out_path)):
        """debug mode off"""
        # return out_path
        os.remove(out_path)

    tree = parse_swc(swc_path)
    root = get_root(tree)
    filename = os.path.split(swc_path)[-1]
    # outfile = os.path.join(outdir, filename)
    sort_swc(swc_path, out_path, root, v3d_path)
    tree_new = parse_swc(out_path)
    idx, type_, x, y, z, r, p = tree_new[0]
    type_ = 1
    tree_new[0] = (idx, type_, x, y, z, r, p)
    write_swc(tree_new, out_path)
    return out_path


def Binarize_img(image, threshold):
    binary_image = np.zeros_like(image)
    binary_image[image > threshold] = 255
    return binary_image


def CalcRadius(img_path, swc_path, tho=-1, radius2d=1, out_path=""):
    if (not out_path):
        out_path = swc_path[0:-4] + "_radius.swc"
    if (os.path.exists(out_path)):
        """debug mode off"""
        # return out_path
        os.remove(out_path)
    if (tho == -1):
        # img = PBD().load(img_path)[0].astype(np.uint8)
        img = file_io.load_image(img_path)
        img_mean, img_std = img_mean_std(img)
        tho = int(img_mean + 1 * img_std)
        img_b = Binarize_img(img.copy(), tho)
        # file_io.save_image(img_path+"b.tif", img_b, False)
        file_io.save_image(img_path + "b.tif", img_b, True)

    if (platform.system() == "Windows"):
        subprocess.run(
            f'{v3d_path} /x neuron_radius /f neuron_radius /i {img_path + "b.tif"} {swc_path} /o {out_path} /p {tho} {radius2d}',
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)  # 全路径
    else:
        print("failed for linux")
        exit(0)

    if (os.path.exists(img_path + "b.tif")): os.remove(img_path + "b.tif")
    return out_path


def RePointRadius(img, x, y, z, radius_2d=True):
    lower_r, upper_r = 1, 30
    z_limit, y_limit, x_limit = img.shape[0], img.shape[1], img.shape[2]
    while (True):
        if (upper_r - lower_r < 0.1):
            # if(not lower_r == 1):
            #     print(lower_r)
            # time.sleep(812202)
            return max(lower_r, 1)
        mid_r = (lower_r + upper_r) / 2
        # print(mid_r)
        # flag = True
        bkg_num = 0
        vox_num = 0
        if (not radius_2d):
            z_lower, z_upper = max(0, int(z - mid_r)), min(z_limit, int(z + mid_r))
            for i in range(z_lower, z_upper):
                y_range = math.sqrt(max(0, mid_r ** 2 - (i - z) ** 2))
                y_lower, y_upper = max(0, int(y - y_range)), min(y_limit, int(y + y_range))
                for j in range(y_lower, y_upper):
                    x_range = math.sqrt(max(0, y_range ** 2 - (j - y) ** 2))
                    x_lower, x_upper = max(0, int(x - x_range)), min(x_limit, int(x + x_range))
                    for k in range(x_lower, x_upper):
                        vox_num = vox_num + 1
                        if (img[i][j][k] == 0):
                            bkg_num = bkg_num + 1
        else:
            z_lower, z_upper = int(z), int(z) + 1
            for i in range(z_lower, z_upper):
                y_range = math.sqrt(max(0, mid_r ** 2 - (i - z) ** 2))
                y_lower, y_upper = max(0, int(y - y_range)), min(y_limit, int(y + y_range))
                for j in range(y_lower, y_upper):
                    x_range = math.sqrt(max(0, y_range ** 2 - (j - y) ** 2))
                    x_lower, x_upper = max(0, int(x - x_range)), min(x_limit, int(x + x_range))
                    for k in range(x_lower, x_upper):
                        vox_num = vox_num + 1
                        if (img[i][y_limit - j][k] == 0):
                            bkg_num = bkg_num + 1
                        # if(mid_r <= 5 and x == 224 and y == 238 and z == 34):
                        #     print(k, j, i)
                        #     print(img[i][j][k])

        if (not bkg_num or (vox_num - bkg_num) / vox_num < 0.5):
            # if (mid_r <= 5 and x == 224 and y == 238 and z == 34):
            #     print((vox_num - bkg_num), vox_num)
            upper_r = mid_r
        else:
            lower_r = mid_r


from seg_prune_v1 import *


# import seg_prune_v1
# from examples_cy import *
# def CalcRadius2(img_path, swc_path, out_path=""):
#     if (not out_path):
#         out_path = swc_path[0:-4] + "_radius.swc"
#     if (os.path.exists(out_path)):
#         """debug mode off"""
#         # return out_path
#         os.remove(out_path)
#
#     img = file_io.load_image(img_path)
#     img = cv2.blur(img, (5, 5))
#     img_mean, img_std = img_mean_std(img)
#     tho = int(img_mean + 1 * img_std)
#     img_b = Binarize_img(img.copy(), tho)
#     file_io.save_image(img_path + "b.tif", img_b)
#     # need flip, maybe
#     # file_io.save_image(img_path + "b.tif", img_b, False)
#
#     # point_l = Readswc_v2(swc_path)
#     res_lines = []
#     with open(swc_path, 'r') as f:
#         lines = f.readlines()
#     num = 0
#     for line in lines:
#         if (line[0] == '#'): continue
#         temp_line = line.split()
#         temp_line[5] = RePointRadius(np.array(img_b).astype(int), float(temp_line[2]), float(temp_line[3]),
#                                      float(temp_line[4]))
#         res_line = "%s %s %s %s %s %s %s\n" % (
#             temp_line[0], temp_line[1], temp_line[2],
#             temp_line[3],
#             temp_line[4], str(temp_line[5]), temp_line[6]
#         )
#         res_lines.append(res_line)
#         num = num + 1
#
#         if (not num % 100):
#             print(f"{num} / {len(lines)}")
#
#     file_handle = open(out_path, mode="a")
#     file_handle.writelines(res_lines)
#     file_handle.close()


def Flipswc(img_path, swc_path, out_path=""):
    if (not out_path):
        out_path = swc_path[0:-4] + "_flip.swc"
    if (os.path.exists(out_path)):
        """debug mode off"""
        # return out_path
        os.remove(out_path)

    img = file_io.load_image(img_path)
    y_limit = img.shape[1]

    with open(swc_path, 'r') as f:
        lines = f.readlines()
    res_lines = []
    for line in lines:
        if (line[0] == '#'): continue
        temp_line = line.split()
        res_line = "%s %s %s %s %s %s %s\n" % (
            temp_line[0], temp_line[1], temp_line[2],
            str(y_limit - float(temp_line[3])),
            temp_line[4], str(temp_line[5]), temp_line[6]
        )
        res_lines.append(res_line)

    file_handle = open(out_path, mode="a")
    file_handle.writelines(res_lines)
    file_handle.close()

    return out_path


def Refineswc(img_path, swc_path, out_path=""):
    if (not out_path):
        out_path = swc_path[0:-4] + "_refine.swc"
    if (os.path.exists(out_path)):
        """debug mode off"""
        # return out_path
        os.remove(out_path)

    v3d_qt6_path = r"D:/Vaa3D_V4.001_Windows_MSVC_64bit/vaa3d_msvc"
    nrrs_path = r"D:/Vaa3D_V4.001_Windows_MSVC_64bit/plugins/refine/refine_swc.dll"
    if (platform.system() == "Windows"):
        subprocess.run(
            f'{v3d_qt6_path} /x {nrrs_path} /f refine_image /i {swc_path} {img_path} /o {out_path}',
            stdout=subprocess.DEVNULL)  # 全路径
    else:
        print("failed for linux")
        exit(0)

    return out_path


def Resampleswc(swc_path, resample_step=2, out_path=""):
    def check_resample(resample_swc_path):
        with open(resample_swc_path, "r") as f:
            lines = f.readlines()
            data = []
            if (lines[0][0] == '#' and lines[1][0] == '2'):
                lines[0] = lines[0][lines[0].find(",pid") + 4:]
            for i in lines:
                temp_line = i.split()
                line = "%d %d %f %f %f %f %d\n" % (
                    int(temp_line[0]), int(temp_line[1]), float(temp_line[2]),
                    float(temp_line[3]),
                    float(temp_line[4]), float(temp_line[5]), int(temp_line[6])
                )
                data.append(line)  # 记录每一行
        # write
        with open(resample_swc_path, "w") as f:
            for i in data:
                f.writelines(i)

    if (not out_path):
        out_path = swc_path[:-4] + "_resample.swc"
    if (os.path.exists(out_path)):
        """debug mode off"""
        # return out_path
        os.remove(out_path)

    if (platform.system() == "Windows"):
        subprocess.run(
            f'{v3d_path} /x resample_swc /f resample_swc /i {swc_path} /o {out_path} /p {resample_step}',
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)  # 全路径
    else:
        print("failed for linux")
        exit(0)
    check_resample(out_path)

    return out_path


def Sortswc(img_path, swc_path, soma_num=-1, out_path=""):
    def get_soma_num(img_path, swc_path):
        img = file_io.load_image(img_path).astype(np.uint16)
        soma_num_res = -1
        soma_num_dis = -1
        # print(img.shape)
        temp_path1 = swc_path + "1.marker"
        # print(img.shape)
        temp_path2, x_center, y_center, z_center, r_center = soma_detection(img, temp_path1)
        # print(f"x_center, y_center, z_center, r_center", x_center, y_center, z_center, r_center)
        if (os.path.exists(temp_path1)): os.remove(temp_path1)
        if (os.path.exists(temp_path2)): os.remove(temp_path2)
        # print(x_center, y_center, z_center)
        soma_x, soma_y, soma_z = 0, 0, 0

        with open(swc_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if (line[0] == '#'): continue
            temp_line = line.split()
            # print(temp_line)
            if (not float(temp_line[6]) == -1): continue
            temp_dis = (x_center - float(temp_line[2])) ** 2 + \
                       (y_center - float(temp_line[3])) ** 2
            # print(f"{temp_line[0]} temp_dis {temp_dis}, x_center {temp_line[2]}, y_center {temp_line[3]}")
            if (soma_num_res == -1 or soma_num_dis > temp_dis):
                soma_num_res, soma_x, soma_y, soma_z = temp_line[0], temp_line[2], temp_line[3], temp_line[4]
                soma_num_dis = temp_dis

        # print(soma_num_res, soma_x, soma_y, soma_z)
        return soma_num_res, soma_x, soma_y, soma_z

    def check_0(swc_path, out_path):
        with open(out_path, 'r') as f:
            lines = f.readlines()
        out_path2 = swc_path[:-4] + "_sort2.swc"
        if (os.path.exists(out_path2)): os.remove(out_path2)
        res_lines = []
        for line in lines:
            if (line[0] == '#'): continue
            temp_line = line.split()
            if (float(temp_line[2]) == 0 and float(temp_line[3]) == 0 and float(temp_line[4]) == 0): continue
            res_lines.append(line)

        file_handle = open(out_path2, mode="a")
        file_handle.writelines(res_lines)
        file_handle.close()
        if (os.path.exists(out_path)): os.remove(out_path)

        return out_path2

    def check_soma(swc_path, out_path, soma_x, soma_y, soma_z):
        with open(out_path, 'r') as f:
            lines = f.readlines()
        # print(out_path)
        temp_soma_num = -1
        for line in lines:
            if (line[0] == '#'): continue
            temp_line = line.split()
            if (abs(float(temp_line[2]) - float(soma_x) < 0.1) and
                    abs(float(temp_line[3]) - float(soma_y)) < 0.1 and
                    abs(float(temp_line[4]) - float(soma_z)) < 0.1):
                # print("!!!!!")
                temp_soma_num = int(temp_line[0])
                break
        # print(f"soma {out_path, soma_x, soma_y, soma_z, temp_soma_num}")
        if (temp_soma_num == 1): return out_path, True
        if (temp_soma_num == -1):
            print("fail to find soma")
            exit(0)

        out_path2 = swc_path[:-4] + "_sort3.swc"
        if (os.path.exists(out_path2)): os.remove(out_path2)
        if (platform.system() == "Windows"):
            subprocess.run(
                f'{v3d_path} /x sort /f sort_swc /i {out_path} /o {out_path2} /p 0 {temp_soma_num}',
                stdout=subprocess.DEVNULL)  # 全路径
        else:
            print("failed for linux")
            exit(0)
        if (os.path.exists(out_path)): os.remove(out_path)
        return out_path2, False

    if (not out_path):
        out_path = swc_path[:-4] + "_sort.swc"
    if (os.path.exists(out_path)):
        """debug mode off"""
        # return out_path
        os.remove(out_path)

    if (soma_num == -1):
        soma_num, soma_x, soma_y, soma_z = get_soma_num(img_path, swc_path)
    # time.sleep(10000)
    if (platform.system() == "Windows"):
        subprocess.run(
            f'{v3d_path} /x sort /f sort_swc /i {swc_path} /o {out_path} /p 0 {soma_num}',
            stdout=subprocess.DEVNULL)  # 全路径
    else:
        print("failed for linux")
        exit(0)

    out_path = check_0(swc_path, out_path)
    check_times = 0
    while(True):
        check_times = check_times + 1
        out_path, check_res = check_soma(swc_path, out_path, soma_x, soma_y, soma_z)
        if(check_res) == True:
            folder_path, file_name = os.path.split(swc_path)
            file_name = str(file_name).split("_")[0]
            soma_marker_path = os.path.join(folder_path, file_name + ".marker")
            line = "%s, %s, %s, 0.000, 1, , , 255,0,0\n" % (soma_x, soma_y, soma_z)
            file_handle = open(soma_marker_path, mode="a")
            file_handle.writelines(line)
            file_handle.close()
            break
        if(check_times > 3):exit(100)
    return out_path

def is_in_box(x, y, z, imgshape):
    """
    imgshape must be in (z,y,x) order
    """
    if x < 0 or y < 0 or z < 0 or \
        x > imgshape[2] - 1 or \
        y > imgshape[1] - 1 or \
        z > imgshape[0] - 1:
        return False
    return True

from skimage.draw import line_nd

def swc_to_image(tree, imgshape=(256, 512, 512)):
    # Note imgshape in (z,y,x) order
    # initialize empty image
    img = np.zeros(shape=imgshape, dtype=np.uint8)
    # get the position tree and parent tree
    pos_dict = {}
    soma_node = None
    for i, leaf in enumerate(tree):
        idx, type_, x, y, z, r, p = leaf
        if p == -1:
            soma_node = leaf

        leaf = (idx, type_, x, y, z, r, p, is_in_box(x, y, z, imgshape))
        pos_dict[idx] = leaf
        tree[i] = leaf

    xl, yl, zl = [], [], []
    for _, leaf in pos_dict.items():
        idx, type_, x, y, z, r, p, ib = leaf
        if idx == 1: continue  # soma

        if p not in pos_dict:
            continue
        parent_leaf = pos_dict[p]
        if (not ib) and (not parent_leaf[ib]):
            print('All points are out of box! do trim_swc before!')
            raise ValueError

        # draw line connect each pair
        cur_pos = leaf[2:5]
        par_pos = parent_leaf[2:5]
        lin = line_nd(cur_pos[::-1], par_pos[::-1], endpoint=True)

        xl.extend(list(lin[2]))
        yl.extend(list(lin[1]))
        zl.extend(list(lin[0]))

    xn, yn, zn = [], [], []
    for (xi, yi, zi) in zip(xl, yl, zl):
        if is_in_box(xi, yi, zi, imgshape):
            xn.append(xi)
            yn.append(yi)
            zn.append(zi)
    img[zn, yn, xn] = 255

    return img

def swc2img(img_path, swc_path, out_path=""):
    # "Usage v3d -x swc_to_maskimage_sphere_unit -f swc_to_maskimage -i <input.swc> [-p <sz0> <sz1> <sz2>] [-o <output_image.raw>]\n"
    # "Usage v3d -x swc_to_maskimage_sphere_unit -f swc_filter -i <input.tif> <input.swc> [-o <output_image.raw>]\n"
    if (not out_path):
        out_path = swc_path[:-4] + "_ano.tif"
    if (os.path.exists(out_path)):
        """debug mode off"""
        # return out_path
        os.remove(out_path)

    img = file_io.load_image(img_path)

    # tree = parse_swc(swc_path)
    # lab = swc_to_image(tree, imgshape=img.shape)
    # file_io.save_image(out_path, lab)

    if (platform.system() == "Windows"):
        subprocess.run(
            f'{v3d_path} /x swc_to_maskimage_sphere_unit /f swc_to_maskimage /i {swc_path} '
            f'/p {img.shape[2]} {img.shape[1]} {img.shape[0]} /o {out_path}',
            stdout=subprocess.DEVNULL)  # 全路径
    else:
        print("failed for linux")
        exit(0)
    # print(img_path)
    # print(swc_path)
    # print(out_path)
    # print(img.shape)
    # print(os.path.exists(out_path))
    # time.sleep(500)
    # img = file_io.load_image(out_path)
    # print(f"swc_img {out_path}")
    # print(type(img))
    # print(np.max(img))
    return out_path


def Dilateimg(img_path, kern = 3, out_path=""):
    # cv2.dilate(img, kernel, iteration)
    if (not out_path):
        out_path = img_path[:-4] + "_dilate.tif"
    if (os.path.exists(out_path)):
        """debug mode off"""
        # return out_path
        os.remove(out_path)

    img = file_io.load_image(img_path)
    # kernel = np.ones((kern, kern, kern), np.uint8)
    # img_d = cv2.dilate(img, kernel=kernel, iterations=iter)
    # kernel = skimage.morphology.cube(kern)
    kernel = np.ones((kern, kern, round(kern / 2)), np.uint8)
    img_d = skimage.morphology.dilation(img, kernel)
    file_io.save_image(out_path, img_d)

    return out_path

def Erodeimg(img_path, kern = 3, out_path=""):
    # cv2.dilate(img, kernel, iteration)
    if (not out_path):
        out_path = img_path[:-4] + "_erode.tif"
    if (os.path.exists(out_path)):
        """debug mode off"""
        # return out_path
        os.remove(out_path)

    img = file_io.load_image(img_path)
    kernel = np.ones((kern, kern, round(kern / 2)), np.uint8)
    img_e = skimage.morphology.erosion(img, kernel)
    file_io.save_image(out_path, img_e)

    return out_path

def Andimg(origin_img_path, ano_path, out_path=""):
    if (not out_path):
        out_path = origin_img_path[:-4] + "_mask.tif"
    if (os.path.exists(out_path)):
        """debug mode off"""
        # return out_path
        os.remove(out_path)
    img = file_io.load_image(origin_img_path)
    img_mean, img_std = img_mean_std(img)
    tho = int(img_mean + 1 * img_std)
    img_b = Binarize_img(img.copy(), tho)
    img_a = file_io.load_image(ano_path)

    # print(img.shape, img_a.shape, img_b.shape)

    # img_mask = cv2.bitwise_and(img_b, img_a)
    img_mask = np.logical_and(img_b, img_a).astype(np.uint8)
    file_io.save_image(out_path, img_mask)

    return out_path

def Andimg2(ano_path, soma_region_path, out_path=""):
    if (not out_path):
        out_path = ano_path[:-4] + "_mask.tif"
    if (os.path.exists(out_path)):
        """debug mode off"""
        # return out_path
        os.remove(out_path)
    img = file_io.load_image(ano_path)
    img_s = file_io.load_image(soma_region_path)

    # img_mask = cv2.bitwise_and(img, img_s)
    img_mask = np.logical_and(img, img_s).astype(np.uint8)
    file_io.save_image(out_path, img_mask)

    return out_path

def Orimg(img_path1, img_path2, out_path=""):
    if (not out_path):
        out_path = img_path1[:-4] + "_or.tif"
    if (os.path.exists(out_path)):
        """debug mode off"""
        # return out_path
        os.remove(out_path)
    img1 = file_io.load_image(img_path1).astype(np.uint8)
    # file_io.save_image(out_path+"_test1.tif", img1)
    img2 = file_io.load_image(img_path2).astype(np.uint8)
    # file_io.save_image(out_path + "_test2.tif", img2)
    # img_or = cv2.bitwise_or(img1, img2)
    img_or = np.logical_or(img1, img2).astype(np.uint8)
    file_io.save_image(out_path, img_or)

    return out_path

def Dustimg(img_path, kern=3, out_path = ""):
    if (not out_path):
        out_path = img_path[:-4] + "_dust.tif"
    if (os.path.exists(out_path)):
        """debug mode off"""
        # return out_path
        os.remove(out_path)

    img = file_io.load_image(img_path)

    kernel = np.ones((kern, kern, kern), np.uint8)
    closing_image = skimage.morphology.closing(img, kernel)

    labeled_image = cc3d.connected_components(closing_image, connectivity=26)
    largest_label = np.argmax(np.bincount(labeled_image.flat)[1:]) + 1
    largest_component_binary = ((labeled_image == largest_label) * 255).astype(np.uint8)

    file_io.save_image(out_path, largest_component_binary)

    return out_path


def findAllFile(base, file_type):
    file_list = []
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith(file_type):
                fullname = os.path.join(root, f)
                file_list.append(fullname)
                # yield fullname
    return file_list

from seg_prune_v1 import *
from swc_tool_lib import *
from scipy.ndimage import zoom
def soma_region_growing(img_path, swc_path, threshold=0.95, out_path = ""):
    if (not out_path):
        out_path = img_path[:-4] + "_soma.tif"
    if (os.path.exists(out_path)):
        """debug mode off"""
        # return out_path
        os.remove(out_path)
    image = file_io.load_image(img_path).astype(np.uint8)
    # print(type(image))
    origin_image = image.copy()
    oi_shape = origin_image.shape
    zoom_times = 1
    max_soma_r = 50
    point_l = Readswc_v2(swc_path)
    seed = (round(point_l.p[1].z),
            round(point_l.p[1].y),
            round(point_l.p[1].x))
    image = image[:, seed[1]-max_soma_r:seed[1]+max_soma_r, seed[2]-max_soma_r:seed[2]+max_soma_r]
    seed = (seed[0], round(max_soma_r/zoom_times), round(max_soma_r/zoom_times))
    # print(image.shape)
    zoom_factors = (1/zoom_times, 1/zoom_times, 1/zoom_times)
    image = zoom(image, zoom_factors, order=1)
    # file_io.save_image(img_path[:-4] + "_asdfasdfasd.tif", image)

    # print(seed)
    visited = np.zeros_like(image, dtype=np.uint8)
    segmented_region = np.zeros_like(image, dtype=np.uint8)
    depth, height, width = image.shape
    seed_value = image[seed]

    # 创建一个队列，用于存储待生长的像素
    queue = []
    queue.append(seed)
    num = 0
    # print(np.mean(image))
    # time.sleep(1000)

    # while queue:
    #     d, h, w = queue.pop(0)
    #     visited[d, h, w] = 1
    #     segmented_region[d, h, w] = 255
    #     num = num + 1
    #     if(num % 100000 == 0):
    #         print(num)
    #         print(d, h, w)
    #         print(image[d, h, w])
    #         print(np.mean(visited))
    #
    #     # 将相邻的像素加入队列，继续生长
    #     for i in [-1, 0, 1]:
    #         for j in [-1, 0, 1]:
    #             for k in [-1, 0, 1]:
    #                 if (0 <= d + i < depth) and (0 <= h + j < height) and (0 <= w + k < width):
    #                     # print(d + i, h + j, w + k)
    #                     # print(visited[d + i, h + j, w + k])
    #                     # print(seed_value)
    #                     # print(image[d + i, h + j, w + k])
    #                     # print((1-threshold) * seed_value)
    #                     if(visited[d + i, h + j, w + k] == 0 and
    #                             seed_value - image[d + i, h + j, w + k] < (1-threshold) * seed_value):
    #                         queue.append((d + i, h + j, w + k))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    if(seed_value - image[i,j,k] < (1-threshold) * seed_value):
                        segmented_region[i, j, k] = 255
    # print(f"num {num}")
    # file_io.save_image(out_path+"_ss.tif", segmented_region)
    zoom_factors = (zoom_times, zoom_times, zoom_times)
    segmented_region = zoom(segmented_region, zoom_factors, order=0)
    segmented_region = cc3d.dust(
        segmented_region, threshold=100,
        connectivity=6, in_place=False
    )
    segmented_region = pad_3d_image(segmented_region, oi_shape)
    file_io.save_image(out_path, segmented_region)
    return out_path


from skimage import filters, morphology, measure

from scipy.ndimage import binary_fill_holes, label, binary_dilation
from skimage.feature import canny
from skimage.filters import laplace, sobel

def denoise_and_fill(img, kern=3):
    kernel = np.ones((kern, kern, kern), np.uint8)
    dilation_image = skimage.morphology.dilation(img, kernel)

    labeled_image = cc3d.connected_components(dilation_image, connectivity=6)
    largest_label = np.argmax(np.bincount(labeled_image.flat)[1:]) + 1
    largest_component_binary = ((labeled_image == largest_label) * 255).astype(np.uint8)
    # print(np.max(largest_component_binary))
    # time.sleep(123123)

    labeled_image1, num_labels = label(largest_component_binary)
    # 找到每个联通区域的边界
    boundaries = binary_dilation(labeled_image1 > 0) & ~labeled_image1
    filled_image = (binary_fill_holes(boundaries) * 255).astype(np.uint8)

    # erosion_image = skimage.morphology.erosion(filled_image, kernel)

    return filled_image
def edge_detection_and_fill(image, kern = 3):
    # Step 1: 边缘检测
    edges = sobel(image)

    # Step 2: 二值化，将边缘映射为二值图像
    binary_edges = (edges > 0.1).astype(np.uint8) # 调整阈值以获得合适的二值图像


    # print(np.max(binary_edges))
    # time.sleep(11000)
    kernel = np.ones((kern, kern, round(kern / 2)), np.uint8)
    closing_image = skimage.morphology.closing(binary_edges, kernel)

    labeled_image = cc3d.connected_components(closing_image, connectivity=6)
    largest_label = np.argmax(np.bincount(labeled_image.flat)[1:]) + 1
    largest_component_binary = ((labeled_image == largest_label)*255).astype(np.uint8)
    # print(np.max(largest_component_binary))
    # time.sleep(123123)


    labeled_image, num_labels = label(largest_component_binary)
    boundaries = binary_dilation(labeled_image > 0) & ~labeled_image
    filled_image = binary_fill_holes(boundaries).astype(np.uint8)*255

    return filled_image

def soma_region(img_path, out_path = ""):
    if (not out_path):
        out_path = img_path[:-4] + "_soma.tif"
    if (os.path.exists(out_path)):
        """debug mode off"""
        # return out_path
        os.remove(out_path)
    image = file_io.load_image(img_path).astype(np.uint8)
    # image = edge_detection_and_fill(image)
    image = Binarize_img(image, 255*0.5)

    kern = 3
    kernel = np.ones((kern, kern, round(kern / 2)), np.uint8)
    closing_image = skimage.morphology.closing(image, kernel)
    labeled_image = cc3d.connected_components(closing_image, connectivity=6)
    largest_label = np.argmax(np.bincount(labeled_image.flat)[1:]) + 1
    largest_component_binary = ((labeled_image == largest_label)*255).astype(np.uint8)
    labeled_image, num_labels = label(largest_component_binary)
    boundaries = binary_dilation(labeled_image > 0) & ~labeled_image
    filled_image = binary_fill_holes(boundaries).astype(np.uint8)*255

    file_io.save_image(out_path, filled_image)
    return out_path
def pad_3d_image(image, target_size):
    depth, height, width = image.shape
    pad_depth = target_size[0] - (depth % target_size[0])
    pad_height = target_size[1] - (height % target_size[1])
    pad_width = target_size[2] - (width % target_size[2])
    padded_image = np.zeros((depth + pad_depth, height + pad_height, width + pad_width), dtype=np.uint8)
    start_d = pad_depth // 2
    start_h = pad_height // 2
    start_w = pad_width // 2
    padded_image[start_d:start_d+depth, start_h:start_h+height, start_w:start_w+width] = image

    return padded_image

def kill_point_in_soma(swc_path, soma_region_path, out_path=""):
    if (not out_path):
        out_path = swc_path[:-4] + "_soma_killed.swc"
    if (os.path.exists(out_path)):
        """debug mode off"""
        # return out_path
        os.remove(out_path)
    soma_region = file_io.load_image(soma_region_path)
    kernel = np.ones((3, 3, 1), np.uint8)
    soma_region = skimage.morphology.dilation(soma_region, kernel)
    point_l = Readswc_v2(swc_path)
    point_l, fiber_l = BuildFiberList_v2(point_l)

    for p in point_l.p:
        x, y, z = round(p.x), round(p.y), round(p.z)
        x = max(0, min(x, soma_region.shape[2]-1))
        y = max(0, min(y, soma_region.shape[1]-1))
        z = max(0, min(z, soma_region.shape[0]-1))
        if(soma_region[z,y,x]):
            point_l.p[p.n].r = 1
        else:
            point_l.p[p.n].r = math.sqrt(point_l.p[p.n].r)
    Writeswc_v2(out_path, point_l, fiber_l)
    return out_path


def Segmask(origin_swc_path, img_path):
    # origin_swc_path = r"D:/tracing_ws/refine/syntheticdataset/swc_refined/02540_origin.swc"
    # img_path = r"D:/tracing_ws/refine/syntheticdataset/swc_refined/02540.tif"
    # mask_root = "Z://SEU-ALLEN//Users//KaifengChen//human_brain//img//mask"
    file_name, extension = os.path.splitext(os.path.basename(origin_swc_path))
    if(os.path.exists(mask_root + "//" + str(file_name) + ".tif")):return

    flip_path = Flipswc(img_path, origin_swc_path)
    sort_path = Sortswc(img_path, flip_path)
    # time.sleep(10000)
    # refine_path = Refineswc(img_path, sort_path)
    soma_region_path = soma_region(img_path)
    #
    radius_path = CalcRadius(img_path, sort_path)
    killed_soma_swc_path = kill_point_in_soma(radius_path, soma_region_path)
    # resample_path = Resampleswc(radius_path)
    #
    ano_path = swc2img(img_path, killed_soma_swc_path)
    ano_path3 = Orimg(ano_path, soma_region_path)

    dilate_path = Dilateimg(ano_path3)
    mask_path = Andimg(img_path, dilate_path)
    #
    ano_path2 = swc2img(img_path, sort_path)
    mask_path2 = Orimg(mask_path, ano_path2)
    # close_path = Closeimg(mask_path2, 3)
    # dust_path = Dustimg(mask_path2)
    # close_path = Closeimg(mask_path2, 3)

    dust_path = Dustimg(mask_path2, 3, mask_root + "//" + str(file_name) + ".tif")

    if(os.path.exists(flip_path)):os.remove(flip_path)
    if (os.path.exists(sort_path)): os.remove(sort_path)
    if(os.path.exists(soma_region_path)): os.remove(soma_region_path)
    if (os.path.exists(radius_path)): os.remove(radius_path)
    # if (os.path.exists(resample_path)): os.remove(resample_path)
    if (os.path.exists(killed_soma_swc_path)): os.remove(killed_soma_swc_path)
    if (os.path.exists(ano_path)): os.remove(ano_path)
    if (os.path.exists(ano_path3)): os.remove(ano_path3)
    if (os.path.exists(dilate_path)): os.remove(dilate_path)
    if (os.path.exists(mask_path)): os.remove(mask_path)
    if (os.path.exists(ano_path2)): os.remove(ano_path2)
    if (os.path.exists(mask_path2)): os.remove(mask_path2)

if __name__ == '__main__':
    # eswc_dir = r"D:\tracing_ws\apical dendrite-XY"
    # eswc_dir = r"D:\tracing_ws\refine\swc_refined\swc_refined"
    # arglist = findAllFile(eswc_dir, ".eswc")
    # # for i in arglist:
    # #     Eswc2swc(i)
    #
    # v3dpbd_dir = r"D:\tracing_ws\human_brain_data_v3dpbd_00001-02363"
    # arglist = findAllFile(v3dpbd_dir, ".v3dpbd")

    # origin_swc_path = r"D:/tracing_ws/dataset/label/swc/2631.swc"
    origin_swc_path = r"D:\tracing_ws\sq\2876.swc"
    # img_path = r"D:/tracing_ws/dataset/img/raw/2631.tif"
    img_path = r"D:\tracing_ws\sq\2876.tif"
    # Segmask(origin_swc_path, img_path)

    # flip_path = Flipswc(img_path, origin_swc_path)
    flip_path = origin_swc_path
    sort_path = Sortswc(img_path, flip_path)
    # time.sleep(10000)
    # refine_path = Refineswc(img_path, sort_path)
    # soma_region_path = soma_region(img_path)
    # #
    # radius_path = CalcRadius(img_path, sort_path)
    # killed_soma_swc_path = kill_point_in_soma(radius_path, soma_region_path)
    # # resample_path = Resampleswc(radius_path)
    # #
    # ano_path = swc2img(img_path, killed_soma_swc_path)
    # ano_path3 = Orimg(ano_path, soma_region_path)
    #
    # dilate_path = Dilateimg(ano_path3)
    # mask_path = Andimg(img_path, dilate_path)
    # #
    # ano_path2 = swc2img(img_path, sort_path)
    # mask_path2 = Orimg(mask_path, ano_path2)
    # # close_path = Closeimg(mask_path2, 3)
    # dust_path = Dustimg(mask_path2)
    # # #
