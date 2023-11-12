import re
import time

import file_io
from swc_base import *
from tqdm import tqdm
import os, sys
import subprocess
import platform
from multiprocessing import Pool
from segmask import findAllFile, Eswc2swc
from pandas import DataFrame
import pandas as pd
from shutil import copyfile
import uuid

if(uuid.UUID(int=uuid.getnode()).hex[-12:] == "bc6ee23a04f7"
        or os.getlogin() == '12626'):
    project_root = r"D:\tracing_ws\dataset"
else:
    project_root = "Z://SEU-ALLEN//Users//KaifengChen//human_brain"
label_root = project_root + "//label"
img_root = project_root + "//img"
label_swc_root = label_root + "//origin_swc"
label_eswc_root = label_root + "//origin_eswc"
raw_root = img_root + "//raw"
swc_root = label_root + "//origin_swc"
eswc_path = label_root + r"\all_reconstruced"
marker_dir = label_root + r"//origin_swc"
label_info_path = project_root + r'//label_info.xlsx'

class Label_info:
    def __init__(self, number):
        self.number = number
        self.p_number = -1
        self.t_number = -1
        self.s_number = -1
        self.regions = "None"
        self.resolution = -1
        self.slice_maker = "None"
        self.slice_date = 0
        self.imager = "None"
        self.labeler = []

from segmask import *


def get_mask(img_path):

    if (not os.path.exists(img_path)): return
    file_name, extension = os.path.splitext(os.path.basename(img_path))
    num = int(re.split("_|-|-_|_-| ", file_name)[0])
    if (uuid.UUID(int=uuid.getnode()).hex[-12:] == "bc6ee23a04f7"
            or os.getlogin() == '12626'):
        pass
    else:
        if(not os.path.exists(raw_root + "//" + str(num) + ".tif")):
            img = file_io.load_image(img_path)
            file_io.save_image(raw_root + "//" + str(num) + ".tif", img[0], flip_tif=False)
            return
    # print(img.shape)
    try:
        Segmask(swc_root + "//" + str(num) + ".swc", raw_root + "//" + str(num) + ".tif")
    except:
        print(f"error at {num}")
    print(f"seg No.{num} done")
def img_main(argslist, swc_list, debug = False):

    temp_list = []
    for i in argslist:
        img_path = i
        # print(swc_list)
        file_name, extension = os.path.splitext(os.path.basename(img_path))
        num = int(re.split("_|-|-_|_-| ", file_name)[0])
        if(not num in swc_list):continue
        temp_list.append(img_path)
    # for i in temp_list:
    #     get_mask(i)
    with Pool(8) as p:
        for res in tqdm(p.imap(get_mask, temp_list), total=len(temp_list)): pass


def swc_main(args, swc_list, df, debug=False):
    # print("!")
    eswc_path = args
    # print(eswc_path)
    file_name, extension = os.path.splitext(os.path.basename(eswc_path))
    # file_split = file_name.split("_|-")
    file_split = re.split("_|-|-_|_-| ", file_name)
    re_split = []
    for i in file_split:
        if(i == " "):continue
        if(len(i)>1 and i[0] == '('):continue
        if(not i):continue
        if(len(i) == 1):continue
        re_split.append(i)

    num_else = 0
    linfo = Label_info(int(re_split[0]))
    swc_list.append(int(re_split[0]))

    # return swc_list

    # linfo.number = int(re_split[0])
    linfo.p_number = int(re_split[1][1:])
    linfo.t_number = int(re_split[2][1:])
    linfo.s_number = int(re_split[3 + num_else][1:])
    if(not (re_split[5 + num_else][0] == "R")):
        linfo.regions = str(re_split[4 + num_else]) + "_" + str(re_split[5 + num_else])
        num_else = num_else + 1
    else:
        linfo.regions = str(re_split[4 + num_else])
    linfo.resolution = int(re_split[5 + num_else][1:])
    linfo.slice_maker = re_split[6 + num_else]
    linfo.slice_date = int(re_split[7 + num_else])
    linfo.imager = re_split[8 + num_else]
    linfo.labeler = []
    # linfo.labeler.append(re_split[9])
    # linfo.labeler.append(re_split[10])
    for i in re_split[9 + num_else:-1]:
        linfo.labeler.append(i)
    data2 = {'number': [linfo.number],
             'p_number': [linfo.p_number],
             't_number': [linfo.t_number],
             's_number': [linfo.s_number],
             'regions': [linfo.regions],
             'resolution': [linfo.resolution],
             'slice_maker': [linfo.slice_maker],
             'slice_date': [linfo.slice_date],
             'imager': [linfo.imager],
             'labeler': [str(linfo.labeler)]}
    df2 = DataFrame(data2)
    if(not linfo.number in df['number'].values):
        df = df._append(df2)
    # df.to_excel(label_info_path, index=False)
        # print(df)


    #
    # new_eswc_path = f"{label_eswc_root}//{linfo.number}.eswc"
    # new_swc_path = f"{label_swc_root}//{linfo.number}.swc"
    # if(not os.path.exists(new_eswc_path)):
    #     copyfile(eswc_path, new_eswc_path)
    # if(not os.path.exists(new_swc_path)):
    #     Eswc2swc(eswc_path, new_swc_path)

    # print(f"swc {linfo.number} done")
    return swc_list, df

def Simple_swc_list(marker_dir, swc_list):
    arglist = findAllFile(eswc_path, ".swc")

def Update_soma(marker_dir, label_info_path):
    df = pd.read_excel(label_info_path)
    marker_list = findAllFile(marker_dir, ".marker")
    for i in marker_list:
        with open(i, 'r') as file:
            line = file.read()
        folder_path, file_name = os.path.split(i)
        line = str(line).split(",")
        soma_x = "{:.3f}".format(float(line[0]))
        soma_y = "{:.3f}".format(float(line[1]))
        soma_z = "{:.3f}".format(float(line[2]))
        # soma_list.append([int(file_name[:-7]), soma_x, soma_y, soma_z])
        df.loc[df['number'] == int(file_name[:-7]), ['soma_x', 'soma_y', 'soma_z']] = [soma_x, soma_y, soma_z]
    df.to_excel(label_info_path, index=False)

if __name__ == '__main__':
    # print(eswc_path)
    arglist = findAllFile(eswc_path, ".eswc")
    # print(arglist)
    # label_info_path = r'D://tracing_ws//label_info.xlsx'
    swc_list = []
    if(os.path.exists(label_info_path)):
        os.remove(label_info_path)
    # label_info_path = label_root + '//label_info.xlsx'
    data = {'number': [],
            'p_number': [],
            't_number': [],
            's_number': [],
            'regions': [],
            'resolution': [],
            'slice_maker': [],
            'slice_date': [],
            'imager': [],
            'labeler': []}
    df = DataFrame(data)
    # print(arglist)
    for i in range(len(arglist)):
        swc_list, df = swc_main(arglist[i], swc_list, df)
        if(i%100 == 0):
            print(f"{i} / {len(arglist)} done in swc_main")
    # print(df)
    # print(label_info_path)
    df.to_excel(label_info_path, index=False)
    # with Pool(12) as p:
    #     for res in tqdm(p.imap(main, arglist), total=len(arglist)): pass
    #
    # if (uuid.UUID(int=uuid.getnode()).hex[-12:] == "bc6ee23a04f7"
    #         or os.getlogin() == '12626'):
    #     img_path = img_root
    #     arglist = findAllFile(img_path, ".tif")
    #     img_main(arglist, swc_list)
    # else:
    #     img_path = r"Z:\SEU-ALLEN\Projects\Human_Neurons\all_human_cells\all_human_cells_v3dpbd"
    #     arglist = findAllFile(img_path, ".v3dpbd")
    #     img_main(arglist, swc_list)

    Update_soma(marker_dir, label_info_path)