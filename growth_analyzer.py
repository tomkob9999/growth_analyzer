# Growth Analyzer

# Author: Tomio Kobayashi
# Version 1.1.0
# Updated: 2024/03/12

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

    def __init__(self):
        self.finder = relation_finder()
        
    def analyzer(self, data, xcols, use_lasso=False, skip_outliers=False):
    #     col 0: Timeline
    #     col 1-: Variables
        if use_lasso:
            data = [d[1:] + [d[0]] for d in data]
            res = self.finder.find_relations2(data, "", "Timeline", cols=xcols, const_thresh=0.1, skip_inverse=False, skip_outliers=skip_outliers, use_lasso=use_lasso, xy_switch=True)
            return res
        else:
            res = []
            for i, c in enumerate(xcols):
                pdata = [[row[0], row[i+1]] for row in data]
                ret = self.finder.find_relations2(pdata, c, "Timeline", const_thresh=0.1, skip_inverse=True, skip_outliers=skip_outliers, use_lasso=use_lasso, xy_switch=True)
                res.append(ret)
            return res
    
    def predict(self, x):
        return self.finder.predict(x)