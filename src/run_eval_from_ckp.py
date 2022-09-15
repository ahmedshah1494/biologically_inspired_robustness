from argparse import ArgumentParser
import os

_LOGDIR = '/share/workhorse3/mshah1/biologically_inspired_models/logs'

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--ckp_start', type=int, required=True)
    parser.add_argument('--ckp_end', type=int, required=True)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--gpu_idx', type=int, required=True)
    parser.add_argument('--attacks', nargs='+', type=str, default=['APGD'])
    parser.add_argument('--eps_list', nargs='+', type=str, default=['0.', '0.008', '0.016', '0.024'])
    parser.add_argument('--run_randomized_smoothing_eval', action='store_true')
    parser.add_argument('--output_to_task_logdir', action='store_true')
    parser.add_argument('--center_fixation', action='store_true')
    args = parser.parse_args()

    if args.exp_name != '':
        args.exp_name = '-'+args.exp_name

    for i in range(args.ckp_start, args.ckp_end+1):
        cmd = f'CUDA_VISIBLE_DEVICES={args.gpu_idx} python main.py --task {args.task} --ckp {_LOGDIR}/{args.dataset}/{args.task.split(".")[1]}{args.exp_name}/{i}/checkpoints/model_checkpoint.pt --eval_only --run_adv_attack_battery --attacks {" ".join(args.attacks)} --eps_list {" ".join(args.eps_list)}'
        for k,v in vars(args).items():
            if isinstance(v, bool) and v:
                cmd += f' --{k}'
        print(cmd)
        os.system(cmd)