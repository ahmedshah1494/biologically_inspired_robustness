import os
os.system('python gen_tasks.py')
from rblur.imagenet100_adam_linear_warmup_decay_tasks import *
from rblur.imagenet100_retina_blur_adam_linear_warmup_decay_tasks import *
from rblur.imagenet100_retina_nonuniform_patch_adam_linear_warmup_decay_tasks import *
from rblur.imagenet100_32x32_adam_linear_warmup_decay_tasks import *
from rblur.imagenet100_32x32_retina_blur_adam_linear_warmup_decay_tasks import *
from rblur.imagenet75_32x32_adam_linear_warmup_decay_tasks import *
from rblur.imagenet75_32x32_retina_blur_adam_linear_warmup_decay_tasks import *