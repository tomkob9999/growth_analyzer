# Growth Analyzer

# Author: Tomio Kobayashi
# Version 1.0.8
# Updated: 2024/03/11

import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats as statss
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import sympy as sp
from scipy.optimize import curve_fit
    
class growth_analyzer:

    def analyzer(data, xcols, use_lasso=False):
    #     col 0: Timeline
    #     col 1-: Variables
        if use_lasso:
            data = [d[1:] + [d[0]] for d in data]
            print("data", data)
            res = relation_finder.find_relations(data, "", "Timeline", cols=xcols, const_thresh=0.1, skip_inverse=False, use_lasso=use_lasso)
            return res
        else:
            res = []
            for i, c in enumerate(xcols):
                pdata = [[row[0], row[i+1]] for row in data]
                ret = relation_finder.find_relations(pdata, c, "Timeline", const_thresh=0.1, skip_inverse=True, use_lasso=use_lasso)
                res.append(ret)
            return res
    