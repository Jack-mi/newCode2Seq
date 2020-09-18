import os
import csv
import pandas as pd

if __name__ == '__main__':
    data_path = '/Users/liuxiaosu/codebase/1-tomcat'
    # df = pd.read_csv(os.path.join(data_path, 'allData.csv'))
    # x = df['func']
    # y = df['comm']
    # print(x[1], y[4])
    methods_path = os.path.join(data_path, 'allJavaMethods/')
    comments_path = os.path.join(data_path, 'allJavaComments/')
    X, Y = [], []
    for idx in range(len([f for f in os.listdir(methods_path)])):
        with open(methods_path + '{:0>6d}func.txt'.format(idx), 'r') as f:
            X.append(f.readline())
        with open(comments_path + '{:0>6d}comm.txt'.format(idx), 'r') as f:
            Y.append(float(f.readline()[0]))

    dic = {'func': X, 'comm':Y}
    df = pd.DataFrame(dic)
    df.to_csv(os.path.join(data_path, 'allData.csv'), index=0)