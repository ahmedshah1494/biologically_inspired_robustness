from argparse import ArgumentParser
from importlib import import_module
import os
from mllib.runners.base_runners import BaseRunner
import numpy as np
import torch
from evaluation_tasks import get_adversarial_battery_task
from runners import AdversarialAttackBatteryRunner, AdversarialExperimentRunner, ConsistentActivationAdversarialExperimentRunner
from utils import get_model_checkpoint_dirs

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
    parser.add_argument('--run_adv_attack_battery', action='store_true')
    args = parser.parse_args()

    # s = 9999
    # np.random.seed(s)
    # torch.manual_seed(s)
    # torch.cuda.manual_seed(s)

    task_cls = get_task_class_from_str(args.task)
    task = task_cls()
    # task = MNISTConsistentActivationClassifier()
    # task = MNISTMLP()
    if args.run_adv_attack_battery:
        task = get_adversarial_battery_task(task_cls)()
        runner_cls = AdversarialAttackBatteryRunner
    else:
        runner_cls = ConsistentActivationAdversarialExperimentRunner
    if args.ckp_dir is not None:
        ckp_dirs = get_model_checkpoint_dirs(args.ckp_dir)
    else:
        ckp_dirs = [None]
    for ckp_dir in ckp_dirs:
        print(ckp_dir)
        runner = runner_cls(task, ckp_dir=ckp_dir, load_model_from_ckp=(ckp_dir is not None))
        if args.eval_only:
            runner.create_trainer()
            runner.test()
        else:
            runner.run()