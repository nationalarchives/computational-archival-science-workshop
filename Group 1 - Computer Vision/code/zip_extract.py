#!/usr/bin/env python
# coding: utf-8

# ## Essential imports and path settings

# In[10]:


import sys, os
from utils.helper import get_all_files, get_all_dirs, make_new_dir
import pandas as pd
import numpy as np
import shutil
from zipfile import ZipFile

DATA_ROOT = '/home/tb0035/projects/tna_datathon/data'

def zip_extract(in_dir, out_dir=None):
    """find .zip files in in_dir and extract to out_dir
    if out_dir=None, extract to in_dir
    """
    out = in_dir if out_dir is None else out_dir
    lst = get_all_files(in_dir, trim=1, extension='zip')
    for path in lst:
        print('Extracting %s' % path)
        parent = os.path.dirname(path)
        out_path = os.path.join(out, parent)
        make_new_dir(out_path, False)
        with ZipFile(os.path.join(in_dir, path), 'r') as zip_ref:
            zip_ref.extractall(out_path)
        

BT_IMG = os.path.join(DATA_ROOT, 'BT_images')
zip_extract(BT_IMG)


