import pandas as pd
import argparse
import matplotlib.pyplot as plt
from auto_crop import get_data
from signal_seg import get_idx_from_x

def get_data_all(exprs, cdata_dir):
    tdata_all = {}
    cdata_all = {}
    for expr in exprs:
        cdata_fname = cdata_dir + "/exp%d_capData.csv" % expr
        tdata_fname = cdata_dir + "/exp%d_timeData_matlab.csv" % expr
        tdata_all[expr], cdata_all[expr] = get_data(tdata_fname, cdata_fname)
    return tdata_all, cdata_all

if __name__ == "__main__":
    from eval_conjugate_falsepos import get_mitosis_label

    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-phase1-csv", default="result_slope/conjugate_phase1.csv")
    parser.add_argument("--valid-phase2-csv", default="result_slope/conjugate_phase2.csv")
    parser.add_argument("--cases", default="FP")
    parser.add_argument("--plot", action='store_true')
    args = parser.parse_args()
    
    slopes = pd.read_csv('slope.csv')
    phase1_all = pd.read_csv(args.valid_phase1_csv)
    phase2_all = pd.read_csv(args.valid_phase2_csv)
    n_phase1 = len(phase1_all.values.tolist())
    n_phase2 = len(phase2_all.values.tolist())
    assert n_phase1 == n_phase2
    
    n_detected_mitosis = 0
    n_detected_migration = 0
    n_mitosis = len(slopes[(slopes['behavior'] == 'Mitosis-phase1')])
    n_migration = len(slopes[(slopes['behavior'] == 'Migrate-out')])
    
    TP, FN, FP = 0, 0, 0
    idx_tp, idx_fn, idx_fp = [], [], []
    
    # Get mitotis indices
    idx_mitosis = get_mitosis_label(slopes)
    for idx0, idx1 in idx_mitosis:
        expr0, c0, _, beh0, t_occur0, _, _, _, _ = slopes.values.tolist()[idx0]
        expr1, c1, _, beh1, t_occur1, _, _, _, _ = slopes.values.tolist()[idx1]
        assert beh0 == 'Mitosis-phase1' and beh1 == 'Mitosis-phase2'
        assert expr0 == expr1 and c0 == c1
    
        # Iterate over all detected pair
        mitosis_detected = False
        expr_target, c_target = expr0, c0
        for m1, m2 in zip(phase1_all.values.tolist(), phase2_all.values.tolist()):
             if m1[0] == expr_target and m1[1] == c_target:
                 _, _, t_start_phase1, t_end_phase1 = m1
                 _, _, t_start_phase2, t_end_phase2 = m2
                 if t_start_phase1 < t_occur0 and t_occur0 < t_end_phase1:
                     mitosis_detected = True
                     break
    
        # Record the result
        if mitosis_detected:
            if t_start_phase2 < t_occur1 and t_occur1 < t_end_phase2:
                TP += 1
            else:
                FP += 1
                idx_fp.append(idx0)
        else:
            FN += 1
            idx_fn.append(idx0)
    
    print("TP(%d), FN(%d), FP(%d)" % (TP, FN, FP))
    
    if args.plot:
        if args.cases.lower() == 'fp':
            idx_target = idx_fp[:]
        elif args.cases.lower() == 'fn':
            idx_target = idx_fn[:]
        else:
            raise ValueError()
    
        tdata_all, cchannel_all = get_data_all((1, 3, 7, 20, 22))
        for idx in idx_target:
            slope = slopes.values.tolist()[idx]
            expr, c, _, beh, t_occur, _, _, _, _ = slope
            tdata, cchannel_nopeak_all = tdata_all[expr], cchannel_all[expr]
            tchannel = tdata[c].to_numpy() / 3600
            cchannel_nopeak = cchannel_nopeak_all[c]
        
            # Plot raw signal around t_occur
            idx_start = get_idx_from_x(tchannel, t_occur - 3)
            idx_end   = get_idx_from_x(tchannel, t_occur + 3) + 1
            t_partial = tchannel[idx_start:idx_end]
            y_partial = cchannel_nopeak[idx_start:idx_end]
            plt.plot(t_partial, y_partial, label='raw')
        
            for m1, m2 in zip(phase1_all.values.tolist(), phase2_all.values.tolist()):
                expr_detected, c_detected = int(m1[0]), int(m1[1])
                t1_start, t1_end = float(m1[2]), float(m1[3])
                t2_start, t2_end = float(m2[2]), float(m2[3])
                if expr == expr_detected and c == c_detected and abs(t1_start - t_occur) < 3:
                    idx1_start = get_idx_from_x(tchannel, t1_start)
                    idx1_end = get_idx_from_x(tchannel, t1_end)
                    y1_start, y1_end = cchannel_nopeak[idx1_start], cchannel_nopeak[idx1_end]
                    idx2_start = get_idx_from_x(tchannel, t2_start)
                    idx2_end = get_idx_from_x(tchannel, t2_end)
                    y2_start, y2_end = cchannel_nopeak[idx2_start], cchannel_nopeak[idx2_end]
                    plt.plot((t1_start, t1_end), (y1_start, y1_end), color='red', label='Phase 1')
                    plt.plot((t1_end, t2_start), (y1_end, y2_start), color='lightgrey')
                    plt.plot((t2_start, t2_end), (y2_start, y2_end), color='green', label='Phase 2')
            idx_occur = get_idx_from_x(tchannel, t_occur)
            plt.scatter(tchannel[idx_occur], cchannel_nopeak[idx_occur])
            plt.show()
