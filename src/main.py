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

attacks =  {
            'APGD': eval.get_apgd_atk, 
            'APGD_EOT20': eval.get_eot20_apgd_atk,
            'APGD_EOT50': eval.get_eot50_apgd_atk,
            'APGDL2': eval.get_apgd_l2_atk, 
            'APGDL2_EOT20': eval.get_eot20_apgd_l2_atk,
            'APGDL2_EOT50': eval.get_eot50_apgd_l2_atk,
            'GNoise': eval.get_gnoise_params,
            'UNoise': eval.get_unoise_params,
            'RandOcc': eval.get_randOcc_params,
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
            'Square': eval.get_square_atk,
            'CWL2': eval.get_cwl2_atk
        }

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--ckp', type=str)
    parser.add_argument('--num_test', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_trainings', type=int, default=1)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--prune_and_test', action='store_true')
    parser.add_argument('--run_adv_attack_battery', action='store_true')
    parser.add_argument('--attacks', nargs='+', type=str, choices=attacks.keys(), default=['APGD'])
    parser.add_argument('--eps_list', nargs='+', type=float, default=[0., 0.008, 0.016, 0.024])
    parser.add_argument('--run_randomized_smoothing_eval', action='store_true')
    parser.add_argument('--output_to_task_logdir', action='store_true')
    parser.add_argument('--center_fixation', action='store_true')
    parser.add_argument('--five_fixations', action='store_true')
    parser.add_argument('--disable_retina', action='store_true')
    parser.add_argument('--add_fixed_noise_patch', action='store_true')
    parser.add_argument('--view_scale', type=int, default=None)
    parser.add_argument('--use_lightning_lite', action='store_true')
    parser.add_argument('--use_bf16_precision', action='store_true')
    parser.add_argument('--use_f16_precision', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print(args)

    if args.run_randomized_smoothing_eval or args.run_adv_attack_battery:
        args.eval_only = True

    # s = time()
    # np.random.seed(s)
    # torch.manual_seed(s)
    # torch.cuda.manual_seed(s)

    task_cls = get_task_class_from_str(args.task)
    task: AbstractTask = task_cls()
    exp_params = task.get_experiment_params()
    runner_kwargs={}
    if args.run_adv_attack_battery:
        task = eval.get_adversarial_battery_task(task_cls, args.num_test, args.batch_size, 
                                                   {k:v for k,v in attacks.items() if k in args.attacks},
                                                    args.eps_list, center_fixation=args.center_fixation,
                                                    five_fixation_ensemble=args.five_fixations, 
                                                    view_scale=args.view_scale, disable_retina=args.disable_retina,
                                                    add_fixed_noise_patch=args.add_fixed_noise_patch
                                                    )()
        runner_cls = AdversarialAttackBatteryRunner
        runner_kwargs = {
            'output_to_ckp_dir': (not args.output_to_task_logdir)
        }
    elif args.run_randomized_smoothing_eval:
        task = eval.get_randomized_smoothing_task(task_cls, args.num_test, args.eps_list, args.batch_size, rs_batch_size=args.batch_size,
                                                    center_fixation=args.center_fixation, five_fixation_ensemble=args.five_fixations)()
        runner_cls = RandomizedSmoothingRunner
        if args.use_lightning_lite:
            runner_kwargs = {
                'wrap_trainer_with_lightning': True,
                'lightning_kwargs': {
                    'strategy': "ddp",
                    'devices': 'auto',
                    'accelerator': "gpu",
                    'precision': "bf16"
                }
            }
    else:
        runner_cls = AdversarialExperimentRunner
        if args.use_lightning_lite:
            runner_kwargs = {
                'wrap_trainer_with_lightning': True,
                'lightning_kwargs': {
                    'strategy': "ddp",
                    'devices': 'auto',
                    'accelerator': "gpu",
                    'precision': "bf16"
                }
            }
        else:
            if args.use_f16_precision:
                precision = 16
            elif args.use_bf16_precision:
                precision = 'bf16'
            else:
                precision = 32
            runner_kwargs = {
                'lightning_kwargs': {
                    'precision': precision,
                    # 'profiler': 'simple',
                    'fast_dev_run': args.debug,
                }
            }
    if args.ckp is not None:
        if os.path.isdir(args.ckp):
            ckp_pths = get_model_checkpoint_paths(args.ckp)
        elif os.path.isfile(args.ckp) and os.path.exists(args.ckp):
            ckp_pths = [args.ckp]
    else:
        ckp_pths = [None]
    for ckp_pth in ckp_pths:
        runner = runner_cls(task, num_trainings=args.num_trainings, ckp_pth=ckp_pth, load_model_from_ckp=(ckp_pth is not None), **runner_kwargs)
        if args.eval_only:
            runner.create_trainer()
            runner.test()
        elif args.prune_and_test:
            runner.create_trainer()
            runner.trainer.prune()
            runner.test()
        else:
            runner.run()