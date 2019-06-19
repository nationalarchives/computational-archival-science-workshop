#!/usr/bin/env python
# coding: utf-8

# ## Essential imports and path settings

# In[1]:


import sys, os
from utils.helper import get_all_files, get_all_dirs
import pandas as pd
import numpy as np
import shutil


# In[2]:


DATA_ROOT = '/home/tb0035/projects/tna_datathon/data'


# ## Find and replace white space with underscore

# In[3]:


def fix_path(my_dir):
    """replace white space with underscore '_' in files and directories under my_dir
    Note: only files/dirs inside my_dir are checked. Parent directories above my_dir are ignored.
    """
    sep = os.path.sep
    dir_lst = get_all_dirs(my_dir, trim=1)
    tree_depth = max([len(p.split(sep)) for p in dir_lst])
    dcount = 0
    for i in range(1):#tree_depth):  # repeat the path fixing process several times 
        dir_lst = get_all_dirs(my_dir, trim=1)[::-1]
        for p in dir_lst:
            leaf = p.split(sep)[-1]
            parent = os.path.dirname(p)
            new_leaf = leaf.replace(' ','_')
            if new_leaf != leaf and os.path.exists(os.path.join(my_dir, p)):
                print('Renaming "%s" to "%s"' % (p, os.path.join(parent, new_leaf)))
                shutil.move(os.path.join(my_dir, p), os.path.join(my_dir, parent, new_leaf))
                dcount += 1
    file_lst = get_all_files(my_dir, trim=0)
    fcount = 0
    for f in file_lst:
        leaf = f.split(sep)[-1]
        parent = os.path.dirname(f)
        new_leaf = leaf.replace(' ', '_')
        if new_leaf != leaf:
            print('Renaming "%s" to "%s"' % (f, os.path.join(parent, new_leaf)))
            shutil.move(f, os.path.join(parent, new_leaf))
            fcount += 1
    print('Done. Rename %d dirs and %d files' % (dcount, fcount))


# In[4]:

fix_path(DATA_ROOT)


