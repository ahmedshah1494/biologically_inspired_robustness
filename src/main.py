from argparse import ArgumentParser
from importlib import import_module
import os
from mllib.runners.base_runners import BaseRunner
import torch
from runners import AdversarialExperimentRunner, ConsistentActivationAdversarialExperimentRunner

def load_model_from_ckpdir(d):
    files = os.listdir(d)
    model_file = [f for f in files if f.startswith('model')][0]
    model = torch.load(os.path.join(d, model_file))
    return model

def get_task_class_from_str(s):
    split = s.split('.')
    modstr = '.'.join(split[:-1])
    cls_name =  split[-1]
    mod = import_module(modstr)
    task_cls = getattr(mod, cls_name)
    return task_cls

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--ckp_dir', type=str)
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()

    task_cls = get_task_class_from_str(args.task)
    # task = MNISTConsistentActivationClassifier()
    # task = MNISTMLP()
    task = task_cls()
    runner = ConsistentActivationAdversarialExperimentRunner(task, ckp_dir=args.ckp_dir, load_model_from_ckp=(args.ckp_dir is not None))
    if args.eval_only:
        runner.create_trainer()
        runner.test()
    else:
        runner.run()