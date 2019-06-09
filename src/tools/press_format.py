# -*- coding: utf-8 -*-
#
#
#
# created by wangquan at 2019-06-09 3:36:13
#


import pandas as pd
import numpy as np
import re
from tqdm import tqdm as tq


press_path = '../../data/test/press_test2.csv'
tmp_path = '../../data/test/clear_press_test2.csv'
prefix = 'pv_'


with open(press_path, 'r') as fp:
    press_lines = fp.readlines()


clear_press_lines = []
l = ''
for line in tq(press_lines[1:]):
    if not line.startswith(prefix):
        l += line
    else:
        r = l.replace('\n', '')
#         r = re.subn(',"', ",yinhao", r, 1)[0]
#         r = r.replace('\'', '').replace('"', '').replace(',yinhao', ',"')
        r = r + '\n'
        clear_press_lines.append(r)
        l = line


with open(tmp_path, 'w+') as tmp:
    clear_press_lines[0] = press_lines[0]
    tmp.writelines(clear_press_lines)



