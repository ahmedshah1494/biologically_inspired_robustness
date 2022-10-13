import os
from argparse import ArgumentParser

import torch
import torchvision
import transformers
from adversarialML.biologically_inspired_models.src.runners import (
    AdversarialAttackBatteryRunner, AdversarialExperimentConfig,
    AdversarialExperimentRunner)
from adversarialML.biologically_inspired_models.src.tasks import (
    get_cifar10_adv_experiment_params, get_cifar10_params, set_adv_params,
    set_common_training_params, set_SGD_params)
from adversarialML.biologically_inspired_models.src.utils import \
    get_model_checkpoint_paths
from attrs import define
from mllib.adversarial.attacks import (AttackParamFactory, SupportedAttacks,
                                       SupportedBackend)
from mllib.runners.configs import BaseExperimentConfig
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
from mllib.tasks.base_tasks import AbstractTask

import adversarialML.biologically_inspired_models.src.evaluation_tasks as eval
from main import get_task_class_from_str


class ViTB16Classifier(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        feature_layer_idx: int = 13
        from_pretrained: bool = True
        vit_config: transformers.ViTConfig = None
        normalize_output: bool = True
        num_classes: int = None

    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.feature_layer_idx = self.params.feature_layer_idx
        self.from_pretrained = self.params.from_pretrained
        self.vit_config: transformers.ViTConfig = self.params.vit_config
        self.num_classes = self.params.num_classes
        self.normalize_output = self.params.normalize_output
        self._make_network()
        self._make_name()
    
    def _make_name(self):
        self.name = f"ViT_16B_{self.feature_layer_idx+1}LEncoder"

    def _make_network(self):
        if self.from_pretrained:
            self.vit = transformers.ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            # vit.pooler = None
            # self.vit = transformers.ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
            # self.vit.vit = vit
        else:
            self.vit = transformers.ViTModel(self.vit_config)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.classifier = torch.nn.Linear(self.vit.config.hidden_size, self.num_classes)

    def forward(self, x):
        x = (x - 0.5) / 0.5
        outputs = self.vit(x, output_hidden_states=True, return_dict=True)
        emb = outputs['hidden_states'][self.feature_layer_idx]
        if self.normalize_output:
            emb = self.vit.layernorm(emb)
        emb = emb[:, 0, :]
        # output_dict = self.vit(x, output_hidden_states=True)
        # layer_outputs = (*(output_dict['hidden_states']), output_dict['pooler_output'])
        # emb = layer_outputs[self.feature_layer_idx]
        # if emb.dim() == 3:
        #     emb = emb[:, 0]
        logits = self.classifier(emb)
        return logits
    
    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        if return_logits:
            return logits, loss
        else:
            return loss

class ViTB16_12Layer_Normalized_RobustnessTestTask(AbstractTask):
    feature_layer_idx = 12
    normalize_output = True
    def get_dataset_params(self):
        p = get_cifar10_params(num_train=10000, num_test=1000)
        p.custom_transforms = [torchvision.transforms.Compose([
                                torchvision.transforms.Resize(224),
                                torchvision.transforms.ToTensor()
                            ])]*2
        return p
    
    def get_model_params(self):
        p: ViTB16Classifier.ModelParams = ViTB16Classifier.get_params()
        p.num_classes = 10
        p.feature_layer_idx = self.feature_layer_idx
        p.normalize_output = self.normalize_output
        return p
    
    def get_experiment_params(self):
        p = AdversarialExperimentConfig()
        set_SGD_params(p)
        p.optimizer_config.lr = 0.01
        p.batch_size = 32
        p.training_params.nepochs = 1
        p.num_trainings = 1
        p.logdir = '/share/workhorse3/mshah1/biologically_inspired_models/logs/'
        p.training_params.early_stop_patience = 20
        p.training_params.tracked_metric = 'val_loss'
        p.training_params.tracking_mode = 'min'
        test_eps = [0.0, 0.002, 0.004, 0.008, 0.016]
        p.adv_config.training_attack_params = None
        def eps_to_attack(eps):
            if eps > 0.:
                atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.APGDLINF, SupportedBackend.TORCHATTACKS)
                atk_p.eps = eps
                atk_p.nsteps = 20
                atk_p.step_size = eps/15
                atk_p.random_start = True
                return atk_p
        p.adv_config.testing_attack_params = [eps_to_attack(eps) for eps in test_eps]
        dsp = self.get_dataset_params()
        mp = self.get_model_params()
        p.exp_name = f'{dsp.max_num_train//1000}K'
        return p

class ViTB16_12Layer_RobustnessTestTask(ViTB16_12Layer_Normalized_RobustnessTestTask):
    normalize_output = False

class ViTB16_12Layer_TransferAttackTestTask(ViTB16_12Layer_RobustnessTestTask):
    src_model_path = '/share/workhorse3/mshah1/biologically_inspired_models/logs/cifar10-0.0/ViTB16_12Layer_Normalized_RobustnessTestTask-10K/0/checkpoints/model_checkpoint.pt'
    def get_experiment_params(self):
        p = super().get_experiment_params()
        src_model = torch.load(self.src_model_path)
        for atk_p in p.adv_config.testing_attack_params:
            if atk_p is not None:
                atk_p.model = src_model
        return p

class ViTB16_10Layer_Normalized_RobustnessTestTask(ViTB16_12Layer_Normalized_RobustnessTestTask):
    feature_layer_idx = 10

class ViTB16_10Layer_RobustnessTestTask(ViTB16_10Layer_Normalized_RobustnessTestTask):
    normalize_output = False

class ViTB16_8Layer_RobustnessTestTask(ViTB16_10Layer_RobustnessTestTask):
    feature_layer_idx = 8

class ViTB16_8Layer_Normalized_RobustnessTestTask(ViTB16_8Layer_RobustnessTestTask):
    normalize_output = True

class ViTB16_4Layer_RobustnessTestTask(ViTB16_10Layer_RobustnessTestTask):
    feature_layer_idx = 4

class ViTB16_4Layer_Normalized_RobustnessTestTask(ViTB16_4Layer_RobustnessTestTask):
    normalize_output = True

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--ckp', type=str)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--run_adv_attack_battery', action='store_true')
    args = parser.parse_args()

    # s = 9999
    # np.random.seed(s)
    # torch.manual_seed(s)
    # torch.cuda.manual_seed(s)

    task_cls = locals()[args.task]
    # task_cls = ViTB16RobustnessTestTask
    task = task_cls()
    # task = MNISTConsistentActivationClassifier()
    # task = MNISTMLP()
    if args.run_adv_attack_battery:
        task = eval.get_adversarial_battery_task(task_cls, 512, 32, [eval.get_transfered_atk()])()
        runner_cls = AdversarialAttackBatteryRunner
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
        else:
            runner.run()
