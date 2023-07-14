# Code for Retina Blur
## Getting Started
### Directory structure
The directory structure we used is as follows:
```
    -- <root>
     | -- imagenet
        | -- train
        | -- val
        | -- eval_dataset_dir (used to prevent the evaluation code from parsing the training set, which becomes a bottleneck on slow NFS storage)
            | -- train (symlink to some small image folder dataset)
            | -- val (symlink to imagenet/val)
     | -- ecoset
     | -- ecoset-10
     | -- biologically_inspired_models
        | -- logs
            | -- <model_name>
                | -- 0
                    | -- checkpoints
                        | -- model_checkpoint.pt
                | -- 1
                    | -- ...
                | -- ...
```

### Installation
Ensure you have Anaconda installed. Then:
1. `mkdir rblur-neurips-2023`
2. Extract `rblur-neurips-2023.tar.gz` into `rblur-neurips-2023`
3. Download `deepgaze_checkpoints.tar.gz` from https://drive.google.com/file/d/1qYAd30rlfk2vnwQoBq6bT64U__RJSJrt/view?usp=share_link
4. Extract `deepgaze_checkpoints.tar.gz` into a location of your choice and update lines 611 and 622 in `rblur-neurips-2023/adversarialML/biologically_inspired_models/src/fixation_prediction/models.py` with the path to the extracted checkpoints.
5. Update `logdir_root` and `LOGDIR` in `rblur-neurips-2023/adversarialML/biologically_inspired_models/src/task_utils.py` to `<root>` and the log directory of your choice, respectively.
6. Download and extract `model_checkpoints.tar.gz` from https://drive.google.com/file/d/1mzJlixLUBYtYSZhVcToXRlfi7NeXCkza/view?usp=share_link
7. We suggest you to store each checkpoint in the following directory structure:
```-- <model_name>
    |-- 0
        |-- checkpoints
            |-- model_checkpoint.pt
```
This structure is not necessary but it is the one we used so errors might be less likely if you mimic it. Moreover, the evaluation code stores the results in the grand-parent directory of the checkpoint. For example, if the checkpoint is stored in `rblur-neurips-2023/checkpoints/0/checkpoints/model_checkpoint.pt`, the results will be stored in `rblur-neurips-2023/checkpoints/0/`.
1. Run `conda env create -f environment.yml` to create the conda environment
2. Run `conda activate rblur7` to activate the conda environment
3. Run `pip install git+https://github.com/fra31/auto-attack` to install the AutoAttack library
4. Run `ln -s $(pwd)/mllib -T ~/anaconda3/envs/rblur7/lib/python3.9/site-packages/mllib` to link the mllib folder to the conda environment
5. Run `ln -s $(pwd)/adversarialML -T ~/anaconda3/envs/rblur7/lib/python3.9/site-packages/adversarialML` to link the adversarialML folder to the conda environment

### Training
`python main.py --task ICLR22.<task_file>.<task_name> --use_f16_precision`

The task files are located in `rblur-neurips-2023/adversarialML/biologically_inspired_models/src/tasks/ICLR22/`. 
- `noisy_retina_blur.py` contains tasks for RBlur
- `baseline.py` contains tasks for the baseline models.
- `retina_warp.py` contains tasks for RWarp.
- `adversarial_training.py` contains tasks for adversarial training.
- `randomized_smoothing.py` contains tasks for GNoise models.
- `vonenets.py` contains tasks for models with VOneBlocks.

E.g. `python main.py --task ICLR22.noisy_retina_blur.Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18 --use_f16_precision`

### Evaluation
Example evaluation commands:
- To evaluate  on L2 attacks: `python main.py --task ICLR22.retina_warp.ImagenetRetinaWarpCyclicLRRandAugmentXResNet2x18 --ckp [...]/ImagenetRetinaWarpCyclicLRRandAugmentXResNet2x18/0/checkpoints/epoch\=24-step\=122900.pt --run_adv_attack_battery --attacks PcFmap-APGDL2_25 --eps_list 0.5 1.5 2. --batch_size 25 --num_test 1000 --add_fixation_predictor --num_fixations 5 --fixation_prediction_model deepgazeIII:rwarp-6.1-7.0-7.1-in1k`
- To evaluate RBlur on Linf attacks: `python main.py --task ICLR22.noisy_retina_blur.ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18 --ckp [...]/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18/1/checkpoints/ --run_adv_attack_battery --attacks PcFmap-APGD_25 --eps_list 0.004 .008 --batch_size 10 --num_test 2000 --add_fixation_predictor --add_fixed_noise_patch --view_scale 3 --num_fixations 5 --fixation_prediction_model deepgazeIII:rblur-6.1-7.0-7.1-in1k --precompute_fixation_map`
- To evaluate random affine on L2 attacks: `python main.py --task ICLR22.baseline.EcosetCyclicLRRandAugmentXResNet2x18 --ckp [...]/EcosetCyclicLRRandAugmentXResNet2x18/0/checkpoints/epoch\=24-step\=140800.pt --run_adv_attack_battery --attack APGDL2 --eps_list 1. --batch_size 25 --multi_randaugment --num_test 1130`
- To evaluate baseline on Linf attacks: `python main.py --task ICLR22.baseline.EcosetCyclicLRRandAugmentXResNet2x18 --ckp [...]/EcosetCyclicLRRandAugmentXResNet2x18/0/checkpoints/epoch\=24-step\=140800.pt --run_adv_attack_battery --attack APGD --eps_list 0. .004 .008 --batch_size 25 --num_test 1130`
- To evaluate adversarial training on L2 attacks: `python main.py --task ICLR22.adversarial_training.ImagenetAdvTrainCyclicLRRandAugmentXResNet2x18 --ckp [...]/ImagenetAdvTrainCyclicLRRandAugmentXResNet2x18/0/checkpoints/epoch\=24-step\=122900.pt --run_adv_attack_battery --batch_size 125 --attack APGD --eps_list 0. .004 .008 --num_test 2000`