from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
import csv
import pandas as pd
import numpy as np
from util import get_hr
import io
from util import get_idx_from_x
from util import get_data_all
import sys
import json
from visualize_util import CropInterface

def get_mitosis_label(slopes):
    idx_pair = []
    idx0, idx1 = None, None
    if type(slopes) is pd.core.frame.DataFrame:
        behs = slopes['behavior'].tolist()
        for idx, beh in enumerate(behs):
            if beh == 'Mitosis-phase1':
                idx0 = idx
                assert idx1 is None
            elif beh == 'Mitosis-phase2':
                assert idx0 is not None
                idx1 = idx
                idx_pair.append((idx0, idx1))
                idx0, idx1 = None, None
    else:
        for idx, slope in enumerate(slopes.values.tolist()):
            if len(slope) == 9:
                _, _, _, beh, t_occur, _, _, _, _ = slope
            else:
                beh = slope[3]
            if beh == 'Mitosis-phase1':
                idx0 = idx
                assert idx1 is None
            elif beh == 'Mitosis-phase2':
                assert idx0 is not None
                idx1 = idx
                idx_pair.append((idx0, idx1))
                idx0, idx1 = None, None
    return idx_pair

def get_offset_range(phases, label='mitosis'):
    if label != 'mitosis':
        raise NotImplementedError()
    idx_mitosis = get_mitosis_label(phases)
    y_value_all_list = []
    for idx0, idx1 in idx_mitosis:
        phase1_tokens = phases.values.tolist()[idx0]
        phase2_tokens = phases.values.tolist()[idx1]
        expr, c = phase1_tokens[0], phase2_tokens[1]
        y_value_each = []
        if expr in args.expr:
            tchannel = tchannel_all[expr][c].to_numpy() / 3600
            cchannel_nopeak = cchannel_all[expr][c]
            if phases.iloc[idx0]['mitosis-phase1'] != '-':
                t_phase0 = float(phases.iloc[idx0]['mitosis-phase1'])
                y_phase0 = cchannel_nopeak[get_idx_from_x(tchannel, t_phase0)]
                y_value_each.append(y_phase0)
            if phases.iloc[idx0]['mitosis-phase2'] != '-':
                t_phase1 = float(phases.iloc[idx0]['mitosis-phase2'])
                y_phase1 = cchannel_nopeak[get_idx_from_x(tchannel, t_phase1)]
                y_value_each.append(y_phase1)
            if phases.iloc[idx0]['mitosis-phase3'] != '-':
                t_phase2 = float(phases.iloc[idx0]['mitosis-phase3'])
                y_phase2 = cchannel_nopeak[get_idx_from_x(tchannel, t_phase2)]
                y_value_each.append(y_phase2)
            if phases.iloc[idx0]['mitosis-phase4'] != '-':
                t_phase3 = float(phases.iloc[idx0]['mitosis-phase4'])
                y_phase3 = cchannel_nopeak[get_idx_from_x(tchannel, t_phase3)]
                y_value_each.append(y_phase3)
            y_value_all_list.append(y_value_each)

    ydiff = [max(y_each) - min(y_each) for y_each in y_value_all_list]
    ymax_side = ((max(ydiff) / 2) // 100 + 1) * 100
    offset_range = np.arange(-ymax_side, ymax_side + 100, 100)
    return offset_range

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-path", default="/home/chingyi/Datasets/images")
    parser.add_argument("--cdata-dir", default="~/Datasets/capData_csv")
    parser.add_argument("--phase-csv", default="mitosis_phase.csv")
    parser.add_argument("--expr", nargs='+', default=[1,3,7,20,22])
    parser.add_argument("--nmax-perexpr", default=8, type=int)
    parser.add_argument("--label", default='Migrate-in')
    parser.add_argument("--log-json", default="log_mitosis.json")
    parser.add_argument("--skip-labeled", action='store_true')
    args = parser.parse_args()
    
    # Get image names
    path_root = os.path.realpath(args.img_path)
    img_fnames_all = {}
    for expr in args.expr:
        path_expr = os.path.join(path_root, "exp%d" % expr)
        img_fnames_unsorted = [fname for fname in os.listdir(path_expr) if 'jpg' in fname]
        img_fnames_unsorted_with_hr = []
        for fname in img_fnames_unsorted:
            img_fnames_unsorted_with_hr.append((get_hr(fname), fname))
        img_fnames = [fname for _, fname in sorted(img_fnames_unsorted_with_hr)]
        img_fnames = [_ for _ in sorted(img_fnames_unsorted_with_hr)]
        img_fnames_all[expr] = img_fnames

    phases = pd.read_csv(args.phase_csv)
    tchannel_all, cchannel_all = get_data_all(args.expr, args.cdata_dir)
    with open(args.log_json) as f:
        log = json.loads(f.readline())

    mitosis_counter = 0
    offset_range = get_offset_range(phases)
    idx_mitosis = get_mitosis_label(phases)
    print("log = ", log)
    for idx0, idx1 in idx_mitosis:
        phase1_tokens = phases.values.tolist()[idx0]
        phase2_tokens = phases.values.tolist()[idx1]

        expr, c = phases.iloc[idx0]['Exp. index'], phases.iloc[idx0]['channel']
        str_occur_time = str(int(phases.iloc[idx0]['occur time']))
        print("expr, c = ", expr, c)

        if expr in args.expr:
            img_fnames = img_fnames_all[expr]
            num_phase = sum([int(phases.iloc[idx0][phase_name] != '-') for phase_name in ['mitosis-phase1',  'mitosis-phase2', 'mitosis-phase3', 'mitosis-phase4']])
            if num_phase == 0:
                raise ValueError("Invalid number of phases")
            fig, axs = plt.subplots(1, num_phase + 1)
            fig.set_size_inches((num_phase + 1) * 3, 3)
            tchannel = tchannel_all[expr][c].to_numpy() / 3600
            cchannel_nopeak = cchannel_all[expr][c]
            arg_expr = args.expr.index(expr)
    
            t_value, y_value = [], []
            if phases.iloc[idx0]['mitosis-phase1'] != '-':
                t_phase0 = float(phases.iloc[idx0]['mitosis-phase1'])
                y_phase0 = cchannel_nopeak[get_idx_from_x(tchannel, t_phase0)]
                # axs[0].scatter(t_phase0, y_phase0, label='phase0')
                y_value.append(y_phase0)
                t_value.append(t_phase0)
            if phases.iloc[idx0]['mitosis-phase2'] != '-':
                t_phase1 = float(phases.iloc[idx0]['mitosis-phase2'])
                y_phase1 = cchannel_nopeak[get_idx_from_x(tchannel, t_phase1)]
                # axs[0].scatter(t_phase1, y_phase1, label='phase1')
                y_value.append(y_phase1)
                t_value.append(t_phase1)
            if phases.iloc[idx0]['mitosis-phase3'] != '-':
                t_phase2 = float(phases.iloc[idx0]['mitosis-phase3'])
                y_phase2 = cchannel_nopeak[get_idx_from_x(tchannel, t_phase2)]
                # axs[0].scatter(t_phase2, y_phase2, label='phase2')
                y_value.append(y_phase2)
                t_value.append(t_phase2)
            if phases.iloc[idx0]['mitosis-phase4'] != '-':
                t_phase3 = float(phases.iloc[idx0]['mitosis-phase4'])
                y_phase3 = cchannel_nopeak[get_idx_from_x(tchannel, t_phase3)]
                y_value.append(y_phase3)
                t_value.append(t_phase3)
 
            if args.skip_labeled and str_occur_time in log[str(expr)][str(c)]:
                idx_start = get_idx_from_x(tchannel, t_phase1 - 3)
                idx_end   = get_idx_from_x(tchannel, t_phase2 + 3)

                # Show images at the rest of subplots
                ims = []
                path_expr = os.path.join(path_root, "exp%d" % expr)
                for i, t in enumerate(t_value):
                    idx_img = np.searchsorted([tt for tt, _ in img_fnames], t)
                    idxs_img = np.isclose(t, [tt for tt, _ in img_fnames]).nonzero()[0]
                    if idxs_img.shape != (1,):
                        print(t, idxs_img)
                    idx_img = idxs_img.item()
                    im = Image.open(path_expr + '/' + img_fnames[idx_img][1])

                    # Create a new image with padding
                    new_wl = max(im.size)
                    im_pad = Image.new(im.mode, (new_wl, new_wl), (128,0,64))
                    im_pad.paste(im, (0, 0))
                    ims.append(im_pad)
    
                border = log[str(expr)][str(c)][str_occur_time]
                print("Already labeled, with border", border)

            else:
                # Plot signal at the first subplot
                idx_start = get_idx_from_x(tchannel, t_phase1 - 3)
                idx_end   = get_idx_from_x(tchannel, t_phase2 + 3)
                axs[0].plot(tchannel[idx_start:idx_end], cchannel_nopeak[idx_start:idx_end], color='grey', zorder=0)
                yavg = (sum(y_value) / len(y_value)) # // 100 * 100
                axs[0].set_yticks(yavg + offset_range)
                axs[0].set_yticklabels([])
                axs[0].set_ylim(yavg-200, yavg + 200)
                axs[0].set_title("Expr %d, channel %d" % (expr, c))
                axs[0].legend()
    
                # Show images at the rest of subplots
                ims = []
                path_expr = os.path.join(path_root, "exp%d" % expr)
                for i, t in enumerate(t_value):
                    axs[0].scatter(t, y_value[i], label='phase%d' % i)

                    idx_img = np.searchsorted([tt for tt, _ in img_fnames], t)
                    idxs_img = np.isclose(t, [tt for tt, _ in img_fnames]).nonzero()[0]
                    if idxs_img.shape != (1,):
                        print(t, idxs_img)
                    idx_img = idxs_img.item()
                    im = Image.open(path_expr + '/' + img_fnames[idx_img][1])
                    new_wl = max(im.size)
                    im_pad = Image.new(im.mode, (new_wl, new_wl), (128,0,64))
                    im_pad.paste(im, (0, 0))
                    axs[i+1].imshow(im_pad)
                    ims.append(im_pad)
    
                print("Expr %d, channel %d" % (expr, c))
                if str_occur_time in log[str(expr)][str(c)]:
                    default_border = log[str(expr)][str(c)][str_occur_time]
                else:
                    default_border = [-1, -1, -1, -1]
                print("Default border:", default_border)
                snap_cursor = CropInterface(axs[1:], ims, default_border=default_border)
                fig.canvas.mpl_connect('button_press_event', snap_cursor.on_click)
                fig.canvas.mpl_connect('key_press_event', snap_cursor.on_press)
                plt.show()
    
                # Write log
                border = snap_cursor.border
                log[str(expr)][str(c)][str_occur_time] = border
                with open(args.log_json, 'w') as f:
                    f.write(json.dumps(log))

            # Duplicate and save figure
            fig, axs = plt.subplots(1, num_phase + 1)
            fig.set_size_inches((num_phase + 1) * 4, 4)
            axs[0].plot(tchannel[idx_start:idx_end], cchannel_nopeak[idx_start:idx_end], color='grey', zorder=0)
            yavg = (sum(y_value) / len(y_value)) # // 100 * 100
            axs[0].set_xlabel('Time (hr)')
            axs[0].set_yticks(yavg + offset_range)
            axs[0].set_ylabel(r'|$\Delta$C| (aF)')
            if phases.iloc[idx0]['mitosis-phase1'] != '-':
                axs[0].scatter(t_phase0, y_phase0, label='phase0')
            if phases.iloc[idx0]['mitosis-phase2'] != '-':
                axs[0].scatter(t_phase1, y_phase1, label='phase1')
            if phases.iloc[idx0]['mitosis-phase3'] != '-':
                axs[0].scatter(t_phase2, y_phase2, label='phase2')
            if phases.iloc[idx0]['mitosis-phase4'] != '-':
                axs[0].scatter(t_phase3, y_phase3, label='phase3')
            axs[0].legend()

            for ax, im in zip(axs[1:], ims):
                ax.imshow(im.crop(border))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            img_fname = "%s-%02d.jpg" % ("Mitosis", mitosis_counter)
            print("%s saved" % img_fname)
            plt.savefig(img_fname)
            plt.close('all')
            mitosis_counter += 1

