# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 13:00:18 2021

@author: gmeinwieserch
"""

##############################################################################
import os
from datetime import datetime
from pathlib import Path


##############################################################################
current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                           str(datetime.now().isoformat(timespec='minutes')))

sources_folder = Path('C:/Users/gmeinwieserch/Desktop/211007_Martha/Roh_wav/')

NR = {'Wand_0_0'  : ['27'], # LR =   0, H =   0, NeutrH = 1.6
      'Wand_+_0'  : ['28'], # LR = +.4, H =   0, NeutrH = 1.6
      'Wand_-_0'  : ['29'], # LR = -.4, H =   0, NeutrH = 1.6
      'Wand_0_+'  : ['30'], # LR =   0, H = +.4, NeutrH = 1.6
      'Wand_0_-'  : ['31'], # LR =   0, H = -.4, NeutrH = 1.6
      'Boden_1.5' : ['32'], # A  = 1.5, H = .93
      'Boden_.2'  : ['33'], # A  =  .2, H = .93
      'Direct_.8' : ['35']} # A  =  .8, H = 2

##############################################################################
now = datetime.now().isoformat(timespec='minutes')
print(now)

raw_dir = os.path.join(current_dir, '01_raw_data')
imp_dir = os.path.join(current_dir, '02_imp_data')
avg_dir = os.path.join(current_dir, '03_avg_data')

for el in [current_dir, raw_dir, imp_dir, avg_dir]:
    os.mkdir(os.path.abspath(el))

    try:
        os.mkdir(os.path.abspath(el))
    except:
        print(el + ' - already present')

##############################################################################
par_sweep = [63, 10, 5e3]   # parameter of sweep [fstart, T, fend]
t_cycle = 15                # cycle time 

##############################################################################