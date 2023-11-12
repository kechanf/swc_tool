import os, os.path
import time
import numpy as np
import math
from skimage import io
import matplotlib.pyplot as plt
from openpyxl import Workbook

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torchvision import datasets
# from torchvision.transforms import ToTensor, Lambda
import igraph as ig
from pylib.file_io import *
from sklearn.metrics import recall_score
# from sklearn.externals import joblib
# import joblib
from sklearn import svm
from sklearn.model_selection import train_test_split
import queue
import Topology_scoring.metrics_delin as md
import glob

block_limit = [256, 256, 128]


home_path = "E://KaifengChen//neuTube_plus//dataset//result//" + str(block_limit[0]) + "//segments//"
test_path = "E://KaifengChen//neuTube_plus//dataset//result//" + str(block_limit[0]) + "//segments_test//"
tiff_path = "E://KaifengChen//neuTube_plus//dataset//img//" + str(block_limit[0]) + "//tiff//"
tiffGS_path = "E://KaifengChen//neuTube_plus//dataset//swc//" + str(block_limit[0]) + "//tiff//"
swcGS_path = "E://KaifengChen//neuTube_plus//dataset//swc//" + str(block_limit[0]) + "//raw//"
ConnParam_path = 'E://KaifengChen//neuTube_plus//dataset//result//256//ConnParam.xlsx'
nn_model_path = "E://KaifengChen//neuTube_plus//dataset//result//256//model.pth"
conn_res_path = "E://KaifengChen//neuTube_plus//dataset//result//" + str(block_limit[0]) + "//connector_result//"
origin_path = "E://KaifengChen//neuTube_plus//dataset//result//" + str(block_limit[0]) + "//origin//"
svm_model_path = "E://KaifengChen//neuTube_plus//dataset//result//" + str(block_limit[0]) + "//svm_model.m"
opt_scoring_path = "E://KaifengChen//neuTube_plus//dataset//result//" + str(block_limit[0]) + "//opt_scoring//"
topology_scoring_result = "E://KaifengChen//neuTube_plus//dataset//result//" + str(block_limit[0]) + "//opt_result.txt"

from swc_base import *
from keystructure_finder import *
from compare_GS import *
from calc_connect_parameter import *
from dataset import *
from learning import *
from connector_test import *
from calc_connect_parameter_v2 import *
from  svm_learning import *

