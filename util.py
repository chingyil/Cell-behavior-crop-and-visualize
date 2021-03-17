from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
import csv
import pandas as pd
import numpy as np
import cv2
from median_filter import median_filter

def get_data_all(exprs, cdata_dir):
    tdata_all = {}
    cdata_all = {}
    for expr in exprs:
        cdata_fname = cdata_dir + "/exp%d_capData.csv" % expr
        tdata_fname = cdata_dir + "/exp%d_timeData_matlab.csv" % expr
        tdata_all[expr], cdata_all[expr] = get_data(tdata_fname, cdata_fname)
    return tdata_all, cdata_all

def get_data(tdata_fname, cdata_fname):
    cdata_pd = pd.read_csv(cdata_fname)
    tdata_pd = pd.read_csv(tdata_fname, header=None)

    cdata_raw = cdata_pd.rename(columns={"%2d" % k:k for k in range(17)})
    map_theader_cheader = {i:v for i, v in enumerate(list(cdata_raw.columns))}
    tdata = tdata_pd.rename(columns=map_theader_cheader, errors='raise')
    cdata_raw_all = [cdata_raw[x] for x in cdata_raw.columns]
    cdata_raw_avg = sum(cdata_raw_all) / len(cdata_raw.columns)

    cchannel_nopeak = {}
    for c in cdata_raw.columns:
        cchannel = cdata_raw[c]
        cavg_median  = median_filter(cdata_raw_avg, l1=400, l2=300, l3=200, l4=300)
        cavg_peak = cdata_raw_avg - cavg_median
        cchannel_nopeak[c] = (cchannel - cavg_peak).to_numpy()

    return tdata, cchannel_nopeak
def get_hr(fname):
    t_str = fname.split('_')[1].split('.')[0]
    assert t_str[0] == 't'
    assert len(t_str.split('-')[1]) == 3
    hr, hr_decimal = [int(s) for s in t_str[1:].split('-')]
    return hr + hr_decimal / 1000
    
import io
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def get_idx_from_x(ts, t, eps=5e-3):
    assert len(ts) > 0
    if type(ts) is list:
        return sum(np.array(ts) < t - eps)
    elif type(ts) is np.ndarray:
        return min(np.searchsorted(ts, t - eps), len(ts) - 1)
    raise ValueError("Unsupport type %s" % type(ts))

