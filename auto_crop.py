from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
import csv
import pandas as pd
import numpy as np
import cv2
from median_filter import median_filter

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
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default="/home/chingyi/Datasets/images/exp22/exp22_t0-100.jpg")
    parser.add_argument("--cap-data", default="/home/chingyi/Datasets/capData_csv/exp22_capData.csv")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--channel-num", type=int, default=3)
    args = parser.parse_args()
    
    # Get some required variable
    expr_folder_name = args.filename.split('/')[-2]
    assert expr_folder_name[:3] == 'exp'
    expr_num = int(expr_folder_name[3:])
    assert ("%d" % expr_num) in args.cap_data
    
    print("Experiement #%02d, channel #%02d" % (expr_num, args.channel_num))
    
    # Get image names
    path = os.path.dirname(os.path.realpath(args.filename))
    img_fnames_unsorted = [fname for fname in os.listdir(path) if 'jpg' in fname]
    img_fnames_unsorted_with_hr = []
    for fname in img_fnames_unsorted:
        img_fnames_unsorted_with_hr.append((get_hr(fname), fname))
    img_fnames = [fname for _, fname in sorted(img_fnames_unsorted_with_hr)]
    
    tdata_fname = "/home/chingyi/Datasets/capData_csv/exp%d_timeData_matlab.csv" % expr_num
    tdata, cchannel_nopeak_all = get_data(tdata_fname, args.cap_data)
    
    if args.channel_num not in cchannel_nopeak_all.keys():
        print("No channel %2d exists" % args.channel_num)
        exit(0)

    tchannel = tdata[args.channel_num].to_numpy() / 3600
    cchannel_nopeak = cchannel_nopeak_all[args.channel_num]
    tlen = len(tdata[args.channel_num])
    print("Time resolution = %2f(s)" % ((tdata[args.channel_num][tlen-1] - tdata[args.channel_num][0]) / tlen))
    
    plt.plot(tchannel, cchannel_nopeak, color='lightgrey', label='raw')
    last_x, last_y = 0, 0
    last_x, last_y = 0, cchannel_nopeak[0]
    plt.plot(tchannel, cchannel_nopeak, color='green', label='no peak')
    plt.legend()
    plt.show()
    
    fig, (ax0, ax1) = plt.subplots(1, 2)
    
    class SnaptoCursor(object):
        """
        Like Cursor but the crosshair snaps to the nearest x, y point.
        For simplicity, this assumes that *x* is sorted.
        """
    
        # def __init__(self, ax, x, y):
        def __init__(self, ax, im):
            self.click_counter = 0
            self.xpick = [0, 1000]
            self.ypick = [0, 1000]
            self.ax = ax
            self.im = im
            self.img_update()
    
        def img_update(self):
            x1, x2 = min(self.xpick), max(self.xpick)
            y1, y2 = min(self.ypick), max(self.ypick)
            ax1.imshow(self.im.crop((x1, y1, x2, y2)))
            plt.draw()
    
        def on_click(self, event):
            x, y = event.xdata, event.ydata
            self.xpick[self.click_counter] = x
            self.ypick[self.click_counter] = y
            self.click_counter = (self.click_counter + 1) % 2
            self.img_update()
    
    im = Image.open(args.filename)
    snap_cursor = SnaptoCursor(ax1, im)
    ax0.imshow(im)
    fig.canvas.mpl_connect('button_press_event', snap_cursor.on_click)
    
    plt.show()
    
    n_img = len(img_fnames)
    x1, x2 = int(min(snap_cursor.xpick)), int(max(snap_cursor.xpick))
    y1, y2 = int(min(snap_cursor.ypick)), int(max(snap_cursor.ypick))
    
    fig, axs = plt.subplots(4, 4)
    for i in range(16):
        ix, iy = i // 4, i % 4
        img_idx = i * (n_img // 16)
        im = Image.open(path + '/' + img_fnames[i])
        axs[ix][iy].imshow(im.crop((x1, y1, x2, y2)))
    plt.show()
    
    fps = args.video_fps
    width, height = x2 - x1, y2 - y1
    frame_size = (width + 300, height)
    vname = 'exp%d-channel%02d.avi' % (expr_num, args.channel_num)
    out = cv2.VideoWriter(vname , cv2.VideoWriter_fourcc(*'DIVX'), fps, frame_size)
    
    for i, fname in enumerate(img_fnames):
        fig = plt.figure(figsize=(3,2))
        plt.plot(tchannel, cchannel_nopeak, color='lightgrey')
        plt.text(0, 1300, fname)
        plt.ylim(-400,1200)
        plt.yticks([])
    
        # writing to a image array
        background = -np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    
        t_hr = get_hr(fname)
        t_index = (tdata[args.channel_num] > t_hr * 3600).argmax()
        if t_index > 0:
            print(i, t_index, t_hr, fname)
            c_value = cchannel_nopeak[t_index]
            plt.scatter(t_hr, c_value + 30, s=18)
        else:
            print("No corresponding capacitive data")
        plt_np = get_img_from_fig(fig)
        plt.close()
        background[-200:, -300:, 2] = plt_np[:,:,0]
        background[-200:, -300:, 1] = plt_np[:,:,1]
        background[-200:, -300:, 0] = plt_np[:,:,2]
    
        # Attach image
        img = Image.open(path + '/' + fname).crop((x1, y1, x2, y2))
        background[:height, :width, :] = np.array(img)
        out.write(background)
    
    out.release()
    
