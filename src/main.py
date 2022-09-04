from argparse import ArgumentParser
from importlib import import_module
import os
from time import time
from mllib.tasks.base_tasks import AbstractTask
import numpy as np
import torch
import evaluation_tasks as eval
from adversarialML.biologically_inspired_models.src.runners import AdversarialAttackBatteryRunner, AdversarialExperimentRunner, RandomizedSmoothingRunner
from utils import get_model_checkpoint_paths

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
    parser.add_argument('--ckp', type=str)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--prune_and_test', action='store_true')
    parser.add_argument('--run_adv_attack_battery', action='store_true')
    parser.add_argument('--run_randomized_smoothing_eval', action='store_true')
    args = parser.parse_args()

    # s = time()
    # np.random.seed(s)
    # torch.manual_seed(s)
    # torch.cuda.manual_seed(s)

    task_cls = get_task_class_from_str(args.task)
    task: AbstractTask = task_cls()
    exp_params = task.get_experiment_params()
    if args.run_adv_attack_battery:
        task = eval.get_adversarial_battery_task(task_cls, 1000, 256, 
                                                    {
                                                      'APGD': eval.get_apgd_atk, 
                                                        # 'APGD_EOT20': eval.get_eot20_apgd_atk,
                                                        # 'APGD_EOT50': eval.get_eot50_apgd_atk,
                                                        # 'APGD_Transfer_MM8L':eval.get_transfered_atk(
                                                        # '/share/workhorse3/mshah1/biologically_inspired_models/logs/cifar10-0.0/'
                                                        # 'Cifar10AutoAugmentMLPMixer8LTask-50K/4/'
                                                        # 'checkpoints/model_checkpoint.pt', eval.get_apgd_atk
                                                        # ),
                                                        # 'APGD_Transfer_SC_LP_MM8L':eval.get_transfered_atk(
                                                        # '/share/workhorse3/mshah1/biologically_inspired_models/logs/cifar10-0.0/'
                                                        # 'Cifar10AutoAugmentSupConLinearProjMLPMixer8LTask-50K/0/'
                                                        # 'checkpoints/model_checkpoint.pt', eval.get_apgd_atk
                                                        # ),
                                                        # 'Square': eval.get_square_atk,
                                                        # 'CWL2': eval.get_cwl2_atk
                                                    }
                                                    # , [0., 0.008, 0.016, 0.024, 0.032]
                                                    ,[0, 0.9]
                                                    )()
        runner_cls = AdversarialAttackBatteryRunner
    elif args.run_randomized_smoothing_eval:
        task = eval.get_randomized_smoothing_task(task_cls, 512, [0.125], 128, rs_batch_size=1024*10)()
        runner_cls = RandomizedSmoothingRunner
    else:
        runner_cls = AdversarialExperimentRunner
    if args.ckp is not None:
        if os.path.isdir(args.ckp):
            ckp_pths = get_model_checkpoint_paths(args.ckp)
        elif os.path.isfile(args.ckp) and os.path.exists(args.ckp):
            ckp_pths = [args.ckp]
    else:
        ckp_pths = [None]
    for ckp_pth in ckp_pths:
        runner = runner_cls(task, ckp_pth=ckp_pth, load_model_from_ckp=(ckp_pth is not None))
        if args.eval_only:
            runner.create_trainer()
            runner.test()
        elif args.prune_and_test:
            runner.create_trainer()
            runner.trainer.prune()
            runner.test()
        else:
            runner.run()