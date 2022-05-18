from mllib.runners.base_runners import BaseRunner

from tasks import MNISTConsistentActivationClassifier


task = MNISTConsistentActivationClassifier()
runner = BaseRunner(task)
runner.create_trainer()
runner.train()
runner.test()