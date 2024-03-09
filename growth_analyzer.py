# Growth Analyzer

# Author: Tomio Kobayashi
# Version 1.0.5
# Updated: 2024/03/09

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Lasso

import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats as statss
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import sympy as sp
from scipy.optimize import curve_fit
    
class relation_finder:
    
    def remove_outliers(x, y):
        # Convert lists to numpy arrays if necessary
        x = np.array(x)
        y = np.array(y)

        # Function to calculate IQR and filter based on it
        def iqr_filter(data):
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Return mask for values within the IQR bounds
            return (data >= lower_bound) & (data <= upper_bound)
        y_mask = iqr_filter(y)
        outs = [i for i, o in enumerate(y_mask) if o == False]
        if len(outs) > 0:
            print("Outliers skipped (lines):", outs)
        x_filtered = x[y_mask]
        y_filtered = y[y_mask]

        return x_filtered, y_filtered

    def exp_func(x, a, b, c):
        return (a+c*x) * np.exp(b * x)
    
    def poly_func(x, a, b, c):
        return a + c*x + b*x**2

    def fit_exp(x_data, y_data, init_guess=[]):
        try:
            return curve_fit(relation_finder.exp_func, x_data, y_data, method="dogbox", nan_policy="omit")
#             return curve_fit(relation_finder.exp_func, x_data, y_data, nan_policy="omit")
        except RuntimeError as e:
            try:
                return curve_fit(relation_finder.exp_func, x_data, y_data, method="dogbox", nan_policy="omit", p0=[min(y_data), 0, (max(y_data)-min(y_data))/len(y_data)])
#                 return curve_fit(relation_finder.exp_func, x_data, y_data, nan_policy="omit", p0=[min(y_data), 0, (max(y_data)-min(y_data))/len(y_data)])
            except RuntimeError as ee:
                return None, None

    def fit_poly(x_data, y_data, init_guess=[]):
        return curve_fit(relation_finder.poly_func, x_data, y_data, method="dogbox", nan_policy="omit", p0=[min(y_data), 0, (max(y_data)-min(y_data))/len(y_data)])
#         return curve_fit(relation_finder.poly_func, x_data, y_data, nan_policy="omit", p0=[min(y_data), 0, (max(y_data)-min(y_data))/len(y_data)])
        

    def find_relations(data, colX, colY, cols=[], const_thresh=0.1, skip_inverse=True, use_lasso=False):

        if use_lasso:
            dic_relation = {
                0: ("P", "Proportional Linearly (Y=a*X)"),
                1: ("IP", "Inversely Proportional Linearly (Y=a*(1/X))"),
                2: ("QP", "Proportional Quadruply (Y=a*(X^2))"),
                3: ("IQP", "Inversely Proportional Quadruply (Y=a*(1/X^2))"),
                4: ("SP", "Proportional by Square Root (Y=a*sqrt(X)"),
                5: ("ISP", "Inversely Proportional by Square Root (Y=a*(1/sqrt(X))"),
            }
            num_incs = 6

            if skip_inverse:
                dic_relation = {
                    0: ("P", "Proportional Linearly (Y=a*X)"),
                    1: ("QP", "Proportional Quadruply (Y=a*(X^2))"),
                    2: ("SP", "Proportional by Square Root (Y=a*sqrt(X)"),
                }
                num_incs = 3

            xcol_size = len(data[0])-1
            if not skip_inverse:
                for i in range(xcol_size):
                    for row in data:
                        row.insert(-1, 1/row[i])
            for i in range(xcol_size):
                for row in data:
                    row.insert(-1, row[i] ** 2)
            if not skip_inverse:
                for i in range(xcol_size):
                    for row in data:
                        row.insert(-1, 1/row[i] ** 2)
            for i in range(xcol_size):
                for row in data:
                    row.insert(-1, np.sqrt(row[i]) if row[i] > 0 else np.sqrt(row[i]*-1)*-1)
            if not skip_inverse:
                for i in range(xcol_size):
                    for row in data:
                        row.insert(-1, 1/np.sqrt(row[i]) if row[i] > 0 else 1/np.sqrt(row[i]*-1)*-1)

            model = Lasso()
            X_train = [r[:-1]for r in data]
            Y_train = [r[-1]for r in data]
            X_train, Y_train = growth_analyzer.remove_outliers(X_train, Y_train)
            model.fit(X_train, Y_train)

            print(f"Relation to {colY}")
            print("  Intersect:", model.intercept_)
            print("  Coeffeicients:")

            for i, c in enumerate(model.coef_):
                if np.abs(c) > 0.0000001:
                    print("    ", cols[int(i/num_incs)] if len(cols) > 0 else "    Col" + str(int(i/num_incs)), ":", dic_relation[i%num_incs][1], round(c, 10))
            predictions = model.predict(X_train)
            r2 = r2_score(Y_train, predictions)
            print("  R2:", round(r2, 5))

            for i in range(len(cols)):
                pdata = [[row[i], row[-1]] for row in data]
                df = pd.DataFrame(pdata, columns=[cols[i], colY])
                plt.title("Scatter Plot of " + cols[i] + " and " + colY)
                plt.scatter(data=df, x=cols[i], y=colY)
                plt.figure(figsize=(3, 2))
                plt.show()

            return model.coef_.tolist() + [model.intercept_]

        else:
            
        # Fit a polynomial of the specified degree to the data
            X_train = [r[0]for r in data]
            Y_train = [r[-1]for r in data]
            X_train, Y_train = relation_finder.remove_outliers(X_train, Y_train)
#             print("X_train", X_train)
#             print("Y_train", Y_train)
            X_train_org = X_train
            # reduced to less than 10 so that exponential can be used in regression
            div = 10**int(np.log10(max(X_train)))
            X_train = [r/div for r in X_train]
#             print("Y_train", Y_train)
            params, covariance = relation_finder.fit_exp(X_train, Y_train)
            
            poly_used = False
            if params is None:
                params, covariance = relation_finder.fit_poly(X_train, Y_train)
                poly_used = True
                
            a, b, c = params
#             print("a", a, "b", b, "c", c)
            if poly_used:
                predictions = [relation_finder.poly_func(x, a, b, c) for x in X_train]
            else:
#                 predictions = [(a + c * x) * np.e**(b*x) for x in X_train]
                predictions = [relation_finder.exp_func(x, a, b, c) for x in X_train]
            r2 = r2_score(Y_train, predictions)
            if np.abs(b) < const_thresh and np.abs(c) < const_thresh :
                print(f"{colY} is CONSTANT to {colX} with constant value of {a:.5f} with confidence level (R2) of {r2*100:.2f}%")
            else:
                if c > 0 and not poly_used and b > 0.10:
                    print("   *   *   *   *   *")
                    print(f"EXPONENTIAL GROWH DETECTED {b:.5f}")
                    print("   *   *   *   *   *")
                if poly_used:
                    equation = f"y = {a:.5f} + {c:.5f}*x + {b:.5f}*x**2)"
                    print(f"Equation:", equation)
                else:
                    print(f"Intercept: {a:.5f}")
                    print(f"Slope (original scale): {c/div:.5f}")
                    print(f"Exponential Factor: {b:.5f}")
                    equation = f"y = ({a:.5f}+{c:.5f}*x) * e**({b:.5f}*x)"
                    print(f"Equation (slope scaled by {div}):", equation)
#                 pdata = [[row[0], row[-1]] for row in data]
                pdata = [[x, Y_train[i]] for i, x in enumerate(X_train)]
#                 print("pdata", pdata)
                df = pd.DataFrame(pdata, columns=[colX, colY])
                plt.title("Scatter Plot of " + colX + " and " + colY)
                plt.scatter(data=df, x=colX, y=colY)
#                 print("df", df)
                # Generate x values for the line
                x_line = np.linspace(min(X_train), max(X_train), 1000)  # 100 points from min to max of scatter data
                y_line = [relation_finder.exp_func(x, a, b, c) if not poly_used else relation_finder.poly_func(x, a, b, c) for x in x_line]
                # Plot the line
                plt.plot(x_line, y_line, color='red', label='Line: ' + equation)
                plt.figure(figsize=(3, 2))
                plt.show()
                return [a, b]
    
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