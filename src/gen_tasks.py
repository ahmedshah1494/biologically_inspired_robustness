from itertools import product

def get_adam_linear_warmup_decay_task_str(d, w, dr, wd):
    base_batch_size = 512
    base_rounded_steps_per_epoch = 250
    if w > 2:
        batch_size = base_batch_size // 2
        rounded_steps_per_epoch = base_rounded_steps_per_epoch * 2
    else:
        batch_size = base_batch_size
        rounded_steps_per_epoch = base_rounded_steps_per_epoch
    return f"""
class Imagenet100_64AutoAugmentMLPMixer{d}L{w}xWide{int(dr*10):02d}Dropout{wd[0]}e_{wd[1]}WDAdamLinearWarmupDecayTask(AbstractTask):
    def get_dataset_params(self):
        return get_imagenet100_64_params(
            train_transforms=get_resize_crop_flip_autoaugment_transforms(64, 8, torchvision.transforms.AutoAugmentPolicy.IMAGENET)
        )
    
    def get_model_params(self):
        return get_basic_mlp_mixer_params([3,64,64], 100, 8, {w}*128, {w}*512, {w}*64, nn.GELU, {dr}, {d})
    
    def get_experiment_params(self) -> BaseExperimentConfig:
        return get_adv_experiment_params(
            MixedPrecisionAdversarialTrainer,
            get_common_training_params(),
            get_apgd_testing_adversarial_params(),
            AdamOptimizerConfig(weight_decay={wd[0]}e-{wd[1]}),
            CyclicLRConfig(base_lr=1e-6, max_lr=0.001, step_size_up={rounded_steps_per_epoch}*30, step_size_down={rounded_steps_per_epoch}*270, cycle_momentum=False),
            {batch_size},
            num_training=2
        )
"""

def get_retina_blur_adam_linear_warmup_decay_task_str(d, w, dr, wd):
    base_batch_size = 512
    base_rounded_steps_per_epoch = 250
    if w > 2:
        batch_size = base_batch_size // 2
        rounded_steps_per_epoch = base_rounded_steps_per_epoch * 2
    else:
        batch_size = base_batch_size
        rounded_steps_per_epoch = base_rounded_steps_per_epoch
    return f"""
class Imagenet100_64AutoAugmentRetinaBlurMLPMixer{d}L{w}xWide{int(dr*10):02d}Dropout{wd[0]}e_{wd[1]}WDAdamLinearWarmupDecayTask(AbstractTask):
    def get_dataset_params(self):
        return get_imagenet100_64_params(
            train_transforms=get_resize_crop_flip_autoaugment_transforms(64, 8, torchvision.transforms.AutoAugmentPolicy.IMAGENET)
        )
    
    def get_model_params(self):
        return get_retina_blur_mlp_mixer_params([3,64,64], 100, 8, {w}*128, {w}*512, {w}*64, nn.GELU, {dr}, {d})
    
    def get_experiment_params(self) -> BaseExperimentConfig:
        return get_adv_experiment_params(
            MixedPrecisionAdversarialTrainer,
            get_common_training_params(),
            get_apgd_eot_testing_adversarial_params(10),
            AdamOptimizerConfig(weight_decay={wd[0]}e-{wd[1]}),
            CyclicLRConfig(base_lr=1e-6, max_lr=0.001, step_size_up={rounded_steps_per_epoch}*30, step_size_down={rounded_steps_per_epoch}*270, cycle_momentum=False),
            {batch_size},
            num_training=5
        )
"""

def get_retina_nonuniform_patch_adam_linear_warmup_decay_task_str(d, w, dr, wd):
    base_batch_size = 512
    base_rounded_steps_per_epoch = 250
    if w > 2:
        batch_size = base_batch_size // 2
        rounded_steps_per_epoch = base_rounded_steps_per_epoch * 2
    else:
        batch_size = base_batch_size
        rounded_steps_per_epoch = base_rounded_steps_per_epoch
    return f"""
class Imagenet100_64AutoAugmentRetinaNonUniformPatchEmbeddingMLPMixer{d}L{w}xWide{int(dr*10):02d}Dropout{wd[0]}e_{wd[1]}WDAdamLinearWarmupDecayTask(AbstractTask):
    def get_dataset_params(self):
        return get_imagenet100_64_params(
            train_transforms=get_resize_crop_flip_autoaugment_transforms(64, 8, torchvision.transforms.AutoAugmentPolicy.IMAGENET)
        )
    
    def get_model_params(self):
        return get_retina_nonuniform_patch_mlp_mixer_params([3,64,64], 100, {w}*128, {w}*512, {w}*64, nn.GELU, {dr}, {d})
    
    def get_experiment_params(self) -> BaseExperimentConfig:
        return get_adv_experiment_params(
            MixedPrecisionAdversarialTrainer,
            get_common_training_params(),
            get_apgd_eot_testing_adversarial_params(10),
            AdamOptimizerConfig(weight_decay={wd[0]}e-{wd[1]}),
            CyclicLRConfig(base_lr=1e-6, max_lr=0.001, step_size_up={rounded_steps_per_epoch}*30, step_size_down={rounded_steps_per_epoch}*270, cycle_momentum=False),
            {batch_size},
            num_training=5
        )
"""
COMMONS_IMPORT_STR = 'from adversarialML.biologically_inspired_models.src.imagenet_mlp_mixer_tasks_commons import *'
depths = [8, 12]
widths = [1, 2, 4]
dropout = [0.0, 0.1]
weight_decay = [(5,5), (1,4), (1,3), (1,2)]

def write_tasks_to_file(outfile, write_fn):
    with open(outfile, 'w') as f:
        f.write(COMMONS_IMPORT_STR)
        for d, w, dr, wd in product(depths, widths, dropout, weight_decay):
            s = write_fn(d, w, dr, wd)
            f.write(s + '\n')

write_tasks_to_file("imagenet100_adam_linear_warmup_decay_tasks.py", 
                        get_adam_linear_warmup_decay_task_str)
write_tasks_to_file("imagenet100_retina_blur_adam_linear_warmup_decay_tasks.py", 
                        get_retina_blur_adam_linear_warmup_decay_task_str)
write_tasks_to_file("imagenet100_retina_nonuniform_patch_adam_linear_warmup_decay_tasks.py", 
                        get_retina_nonuniform_patch_adam_linear_warmup_decay_task_str)

# with open("imagenet100_adam_linear_warmup_decay_tasks.py", 'w') as f:
#     f.write(COMMONS_IMPORT_STR)
#     for d, w, dr, wd in product(depths, widths, dropout, weight_decay):
#         s = get_adam_linear_warmup_decay_task_str(d, w, dr, wd)
#         f.write(s + '\n')

# with open("imagenet100_retina_blur_adam_linear_warmup_decay_tasks.py", 'w') as f:
#     f.write(COMMONS_IMPORT_STR)
#     for d, w, dr, wd in product(depths, widths, dropout, weight_decay):
#         s = get_retina_blur_adam_linear_warmup_decay_task_str(d, w, dr, wd)
#         f.write(s + '\n')

# with open("imagenet100_retina_nonuniform_patch_adam_linear_warmup_decay_tasks.py", 'w') as f:
#     f.write(COMMONS_IMPORT_STR)
#     for d, w, dr, wd in product(depths, widths, dropout, weight_decay):
#         s = get_retina_nonuniform_patch_adam_linear_warmup_decay_task_str(d, w, dr, wd)
#         f.write(s + '\n')