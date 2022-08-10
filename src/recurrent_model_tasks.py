from copy import deepcopy
from typing import List, Tuple
from mllib.tasks.base_tasks import AbstractTask
from mllib.runners.configs import BaseExperimentConfig
from mllib.optimizers.configs import SGDOptimizerConfig, ReduceLROnPlateauConfig, AdamOptimizerConfig
from models import ConvEncoder, GeneralClassifier, IdentityLayer, SequentialLayers
from recurrent_models import BaseRecurrentCell, Conv2DSelfProjection, Conv2d, ConvTransposeUpscaler, InputConcatenationLayer, Linear, LinearUpscaler, RecurrentModel
from runners import AdversarialExperimentConfig

from tasks import get_cifar10_params, set_SGD_params, set_adv_params, set_common_training_params
from torch import nn
import torchvision

def get_input_concatenation_layer_params(dim, ninp):
    catp: InputConcatenationLayer.ModelParams = InputConcatenationLayer.get_params()
    catp.combiner_params.dim = dim
    catp.combiner_params.num_inputs = ninp
    return catp

def get_sequential_model_params(layer_params: List):
    p: SequentialLayers.ModelParams = SequentialLayers.get_params()
    p.layer_params = layer_params
    return p

def create_conv_recurrent_cell_params():
    c: BaseRecurrentCell.ModelParams = BaseRecurrentCell.get_params()
    c.common_params.activation = nn.GELU
    cf: Conv2d.ModelParams = Conv2d.get_params()
    c.forward_update_params = cf

    cl: Conv2DSelfProjection.ModelParams = Conv2DSelfProjection.get_params()
    c.lateral_update_params = cl

    cb: Conv2DSelfProjection.ModelParams = Conv2DSelfProjection.get_params()
    c.backward_update_params = cb

    cbh: ConvTransposeUpscaler.ModelParams = ConvTransposeUpscaler.get_params()
    c.hidden_backward_params = cbh

    cba: ConvTransposeUpscaler.ModelParams = ConvTransposeUpscaler.get_params()
    c.backward_act_params = cba

    c.forward_input_combiner_params = get_input_concatenation_layer_params(1, 2)
    c.lateral_input_combiner_params = get_input_concatenation_layer_params(1, 2)
    c.backward_input_combiner_params = get_input_concatenation_layer_params(1, 2)

    return c, cf, cl, cb, cbh, cba

def create_fc_recurrent_cell_params():
    c: BaseRecurrentCell.ModelParams = BaseRecurrentCell.get_params()
    c.common_params.activation = nn.Identity

    cf: Linear.ModelParams = Linear.get_params()
    cl: Linear.ModelParams = Linear.get_params()
    cb: Linear.ModelParams = Linear.get_params()
    cbh: LinearUpscaler.ModelParams = LinearUpscaler.get_params()
    cba: LinearUpscaler.ModelParams = LinearUpscaler.get_params()

    c.forward_update_params = cf
    c.lateral_update_params = cl
    c.backward_update_params = cb
    c.hidden_backward_params = cbh
    c.backward_act_params = cba

    c.forward_input_combiner_params = get_input_concatenation_layer_params(1, 2)
    c.lateral_input_combiner_params = get_input_concatenation_layer_params(1, 2)
    c.backward_input_combiner_params = get_input_concatenation_layer_params(1, 2)

    return c, cf, cl, cb, cbh, cba

def set_linear_params(params, idim, odim):
    params.common_params.input_size = idim
    params.common_params.num_units = odim

def set_conv_params(params, ic, oc, k, s, p):
    params.conv_params.in_channels = ic
    params.conv_params.out_channels = oc
    params.conv_params.kernel_size = k
    params.conv_params.stride = s
    params.conv_params.padding = p

class Cifar10Conv3LGELUModelTask(AbstractTask):
    def get_dataset_params(self):
        p = get_cifar10_params(num_train=50_000)
        return p
    
    def get_experiment_params(self):
        p = AdversarialExperimentConfig()
        set_SGD_params(p)
        p.optimizer_config.lr = 0.05
        # p.optimizer_config = AdamOptimizerConfig()
        p.scheduler_config = ReduceLROnPlateauConfig()
        set_common_training_params(p)
        p.batch_size = 128
        test_eps = [0.0, 0.008, 0.016, 0.024, 0.032, 0.048, 0.064]
        set_adv_params(p, test_eps)
        dsp = self.get_dataset_params()
        p.exp_name = f'{dsp.max_num_train//1000}K'
        return p
    
    def get_model_params(self):
        p1: ConvEncoder.ModelParams = ConvEncoder.get_params()
        p1.common_params.input_size = [3, 32, 32]
        p1.common_params.activation = nn.GELU
        p1.common_params.num_units = [64, 64, 64,]
        p1.common_params.dropout_p = 0.
        p1.conv_params.kernel_sizes = [5, 3, 3]
        p1.conv_params.padding = [0, 1, 1]
        p1.conv_params.strides = [3, 1, 2]

        cp: Linear.ModelParams = Linear.get_params()
        cp.common_params.num_units = 10

        p: GeneralClassifier.ModelParams = GeneralClassifier.get_params()
        p.feature_model_params = p1
        p.classifier_params = cp

        return p
    
class Cifar10AutoAugmentConv3LGELUModelTask(Cifar10Conv3LGELUModelTask):
    def get_dataset_params(self):
        p = get_cifar10_params(num_train=50_000)
        p.custom_transforms = (
            torchvision.transforms.Compose([
                torchvision.transforms.AutoAugment(torchvision.transforms.AutoAugmentPolicy.CIFAR10),
                torchvision.transforms.ToTensor()
            ]),
            torchvision.transforms.ToTensor()
        )
        return p

class Cifar10AutoAugmentConvRecurrentModel3L64UnitTask(AbstractTask):
    num_units = 64
    input_size = [3, 32, 32]
    num_steps = 5
    def get_dataset_params(self):
        p = get_cifar10_params(num_train=50_000)
        p.custom_transforms = (
            torchvision.transforms.Compose([
                torchvision.transforms.AutoAugment(torchvision.transforms.AutoAugmentPolicy.CIFAR10),
                torchvision.transforms.ToTensor()
            ]),
            torchvision.transforms.ToTensor()
        )
        return p
    
    def get_experiment_params(self):
        p = AdversarialExperimentConfig()
        # set_SGD_params(p)
        # p.optimizer_config.lr = 0.05
        p.optimizer_config = AdamOptimizerConfig()
        p.scheduler_config = ReduceLROnPlateauConfig()
        set_common_training_params(p)
        p.batch_size = 128
        test_eps = [0.0, 0.008, 0.016, 0.024, 0.032, 0.048, 0.064]
        set_adv_params(p, test_eps)
        dsp = self.get_dataset_params()
        p.exp_name = f'{dsp.max_num_train//1000}K'
        return p
    
    def get_model_params(self):
        p: RecurrentModel.ModelParams = RecurrentModel.get_params()
        p.recurrence_params.num_time_steps = self.num_steps
        p.common_params.input_size = self.input_size
        c1, c1f, c1l, c1b, c1bh, c1ba = create_conv_recurrent_cell_params()
        c1f.common_params.input_size = self.input_size[:]
        c1f.common_params.input_size[0] = 6
        set_conv_params(c1f, 6, self.num_units, 5, 3, 0)
        set_conv_params(c1l, 2*self.num_units, self.num_units*25, 5, 5, 0)
        set_conv_params(c1b, 2*self.num_units, self.num_units*25, 5, 5, 0)
        c1bh.conv_params = deepcopy(c1f.conv_params)
        c1.backward_act_params = None

        c2, c2f, c2l, c2b, c2bh, c2ba = create_conv_recurrent_cell_params()
        set_conv_params(c2f, self.num_units, self.num_units, 3, 1, 1)
        set_conv_params(c2l, 2*self.num_units, self.num_units*25, 5, 5, 0)
        set_conv_params(c2b, 2*self.num_units, self.num_units*25, 5, 5, 0)
        c2bh.conv_params = deepcopy(c2f.conv_params)
        c2ba.conv_params = deepcopy(c2f.conv_params)

        c3, c3f, c3l, c3b, c3bh, c3ba = create_conv_recurrent_cell_params()
        set_conv_params(c3f, self.num_units, self.num_units, 3, 2, 1)
        set_conv_params(c3l, 2*self.num_units, self.num_units*25, 5, 5, 0)
        set_conv_params(c3b, 2*self.num_units, self.num_units*25, 5, 5, 0)
        c3bh.conv_params = deepcopy(c3f.conv_params)
        c3ba.conv_params = deepcopy(c3f.conv_params)

        c4, c4f, c4l, c4b, c4bh, c4ba = create_fc_recurrent_cell_params()
        set_linear_params(c4f, 20, 10)
        set_linear_params(c4l, 20, 10)
        set_linear_params(c4b, 20, 10)
        c4.use_layernorm = False

        p.cell_params = [c1, c2, c3, c4]
        return p

class Cifar10AutoAugmentConvRecurrentModel3L128UnitTask(Cifar10AutoAugmentConvRecurrentModel3L64UnitTask):
    num_units = 128

class Cifar10AutoAugmentConvRecurrentModel3L196UnitTask(Cifar10AutoAugmentConvRecurrentModel3L64UnitTask):
    num_units = 196