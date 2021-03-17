from util import get_data_all
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cdata-dir", default="~/Datasets/capData_csv")
    parser.add_argument("--log-json", default="log.json")
    parser.add_argument("--expr", nargs='+', default=[1,3,7,20,22])
    args = parser.parse_args()
    
    _, cchannel_all = get_data_all(args.expr, args.cdata_dir)

    log = {}
    for expr in cchannel_all.keys():
        log[int(expr)] = {}
        for c in cchannel_all[expr].keys():
            log[int(expr)][int(c)] = {}

    log_json = json.dumps(log)
    with open(args.log_json, "w") as f:
        f.write(log_json)
