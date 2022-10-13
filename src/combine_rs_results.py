import argparse
import os
import shutil
from adversarialML.biologically_inspired_models.src.utils import load_json, write_pickle

parser = argparse.ArgumentParser()
parser.add_argument('--dir')
args = parser.parse_args()

outfile = f'{os.path.dirname(args.dir)}/randomized_smoothing_preds_and_radii.pkl'

result_files = [f'{args.dir}/{f}' for f in os.listdir(args.dir)]
pnr = {}
labels = []
for fp in result_files:
    r = load_json(fp)
    name, _ = os.path.basename(fp).replace('rs_result_', '').split('_')
    r['radii'] = r.pop('radius')
    labels.append(r["Y"])
    for k,v in r.items():
        pnr.setdefault(name, {}).setdefault(k, []).append(v)
print(pnr)
outdict = {'Y':labels, 'preds_and_radii': pnr}

if os.path.exists(outfile):
    os.rename(outfile, f'{outfile}.bak')

write_pickle(outdict, outfile)