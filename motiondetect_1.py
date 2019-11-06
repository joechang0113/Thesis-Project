#! /usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import numpy as np

# 開啟 CSV 檔案
with open('output_test_cross0614_7.csv') as csvfile:

    # 讀取 CSV 檔案內容
    reader = csv.reader(csvfile)
    
    rows = [row[1] for row in reader]
    
    #column = [row[1] for row in rows]
    #for row in rows:
    print(rows)
