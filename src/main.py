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

# torch.autograd.set_detect_anomaly(True)

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
            'APGD_75': eval.get_apgd_75s_atk,
            'APGD_50': eval.get_apgd_50s_atk,
            'APGD_25': eval.get_apgd_25s_atk,
            'APGD_10': eval.get_apgd_10s_atk,
            'APGD_5': eval.get_apgd_5s_atk,
            'APGD_1': eval.get_apgd_1s_atk,
            'APGD_EOT20': eval.get_eot20_apgd_atk,
            'APGD_EOT50': eval.get_eot50_apgd_atk,
            'APGDL2': eval.get_apgd_l2_atk,
            'APGDL2_25': eval.get_apgd_l2_25s_atk,
            'APGDL2_EOT20': eval.get_eot20_apgd_l2_atk,
            'APGDL2_EOT50': eval.get_eot50_apgd_l2_atk,
            'APGDL1': eval.get_apgd_l1_atk,
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
            'CWL2': eval.get_cwl2_atk,
            'PGD': eval.get_pgd_atk,
            'PGD_25': eval.get_pgd_25s_atk,
            'PGD_10': eval.get_pgd_10s_atk,
            'PGD_5': eval.get_pgd_5s_atk,
            'PGD_1': eval.get_pgd_1s_atk,
            'FA-APGD': eval.get_fixation_aware_atk,
            'PcFmap-APGD': eval.get_precomputed_fixation_apgd_atk,
            'PcFmap-APGD_1': eval.get_precomputed_fixation_apgd_1s_atk,
            'PcFmap-APGD_5': eval.get_precomputed_fixation_apgd_5s_atk,
            'PcFmap-APGD_10': eval.get_precomputed_fixation_apgd_10s_atk,
            'PcFmap-APGD_25': eval.get_precomputed_fixation_apgd_25s_atk,
            'PcFmap-APGD_50': eval.get_precomputed_fixation_apgd_50s_atk,
            'PcFmap-APGDL2_25': eval.get_precomputed_fixation_apgd_l2_25s_atk,
            'PcFmap-APGD_25_EOT10': eval.get_precomputed_fixation_eot10_apgd_25s_atk,
            'AutoAttackLinf': eval.get_autoattack_linf_atk,
        }

if __name__ == '__main__':
    parser = ArgumentParser()
    # Model args
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--ckp', type=str)
    # Data settings
    parser.add_argument('--num_test', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--output_to_task_logdir', action='store_true')
    parser.add_argument('--num_trainings', type=int, default=1)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--prune_and_test', action='store_true')
    # Adversarial attack settings
    parser.add_argument('--run_adv_attack_battery', action='store_true')
    parser.add_argument('--attacks', nargs='+', type=str, choices=attacks.keys(), default=['APGD'])
    parser.add_argument('--eps_list', nargs='+', type=float, default=[0.])
    # Randomized smoothing settings
    parser.add_argument('--run_randomized_smoothing_eval', action='store_true')
    parser.add_argument('--rs_start_batch_idx', type=int, default=0)
    parser.add_argument('--rs_end_batch_idx', type=int)
    # Fixation Settings
    parser.add_argument('--center_fixation', action='store_true')
    parser.add_argument('--five_fixations', action='store_true')
    parser.add_argument('--bb_fixations', action='store_true')
    parser.add_argument('--fixate_on_max_loc', action='store_true')
    parser.add_argument('--view_scale', type=int, default=None)
    parser.add_argument('--hscan_fixations', action='store_true')
    # Fixation model Settings
    parser.add_argument('--add_fixation_predictor', action='store_true')
    parser.add_argument('--fixation_prediction_model', type=str, default='deepgazeII')
    parser.add_argument('--retina_after_fixation', action='store_true')
    parser.add_argument('--use_precomputed_fixations', action='store_true')
    parser.add_argument('--precompute_fixation_map', action='store_true')
    parser.add_argument('--use_clickme_data', action='store_true')
    parser.add_argument('--num_fixations', type=int, default=1)
    parser.add_argument('--many_fixations', action='store_true')
    
    parser.add_argument('--disable_retina', action='store_true')
    parser.add_argument('--straight_through_retina', action='store_true')
    parser.add_argument('--disable_reconstruction', action='store_true')
    parser.add_argument('--use_residual_img', action='store_true')

    parser.add_argument('--use_common_corruption_testset', action='store_true')

    parser.add_argument('--add_fixed_noise_patch', action='store_true')
    parser.add_argument('--add_random_noise', action='store_true')
    parser.add_argument('--multi_randaugment', action='store_true')

    parser.add_argument('--use_lightning_lite', action='store_true')
    parser.add_argument('--use_bf16_precision', action='store_true')
    parser.add_argument('--use_f16_precision', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', default=45551323)
    args = parser.parse_args()

    print(args)

    if args.run_randomized_smoothing_eval or args.run_adv_attack_battery:
        args.eval_only = True

    # s = time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    task_cls = get_task_class_from_str(args.task)
    task: AbstractTask = task_cls()
    exp_params = task.get_experiment_params()
    runner_kwargs={}
    if args.run_adv_attack_battery:
        task = eval.get_adversarial_battery_task(task_cls, args.num_test, args.batch_size, 
                                                   {k:v for k,v in attacks.items() if k in args.attacks},
                                                    args.eps_list, center_fixation=args.center_fixation,
                                                    five_fixation_ensemble=args.five_fixations, 
                                                    hscan_fixation_ensemble=args.hscan_fixations,
                                                    view_scale=args.view_scale, disable_retina=args.disable_retina,
                                                    add_fixed_noise_patch=args.add_fixed_noise_patch, 
                                                    use_common_corruption_testset=args.use_common_corruption_testset,
                                                    disable_reconstruction=args.disable_reconstruction,
                                                    use_residual_img=args.use_residual_img, fixate_in_bbox=args.bb_fixations,
                                                    enable_random_noise=args.add_random_noise,
                                                    apply_rand_affine_augments=args.multi_randaugment,
                                                    fixate_on_max_loc=args.fixate_on_max_loc,
                                                    clickme_data=args.use_clickme_data,
                                                    use_precomputed_fixations=args.use_precomputed_fixations,
                                                    num_fixations=args.num_fixations,
                                                    precompute_fixation_map=args.precompute_fixation_map,
                                                    add_fixation_predictor=args.add_fixation_predictor,
                                                    fixation_prediction_model=args.fixation_prediction_model,
                                                    retina_after_fixation=args.retina_after_fixation,
                                                    straight_through_retina=args.straight_through_retina,
                                                    )()
        runner_cls = AdversarialAttackBatteryRunner
        runner_kwargs = {
            'output_to_ckp_dir': (not args.output_to_task_logdir)
        }
    elif args.run_randomized_smoothing_eval:
        task = eval.get_randomized_smoothing_task(task_cls, args.num_test, args.eps_list, args.batch_size, rs_batch_size=args.batch_size,
                                                    center_fixation=args.center_fixation, five_fixation_ensemble=args.five_fixations,
                                                    start_idx=args.rs_start_batch_idx, end_idx=args.rs_end_batch_idx,
                                                    add_fixed_noise_patch=args.add_fixed_noise_patch)()
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
        if (not args.eval_only) and ('lightning_kwargs' in runner_kwargs):
            runner_kwargs['lightning_kwargs']['resume_from_checkpoint'] = ckp_pth
        runner = runner_cls(task, num_trainings=args.num_trainings, ckp_pth=ckp_pth, load_model_from_ckp=(ckp_pth is not None), **runner_kwargs)
        if args.eval_only:
            with open(f'{os.path.dirname(os.path.dirname(ckp_pth))}/eval_cmd_{time()}.txt', 'w') as f:
                f.write(str(args))
            runner.create_trainer()
            runner.test()
        elif args.prune_and_test:
            runner.create_trainer()
            runner.trainer.prune()
            runner.test()
        else:
            runner.run()
            with open(f'{runner.trainer.logdir}/train_cmd_{time()}.txt', 'w') as f:
                f.write(str(args))