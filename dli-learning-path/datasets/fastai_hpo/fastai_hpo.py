from fastai.imports import *
from fastai.script import *
from fastai.vision import *
from fastai.distributed import *

# HPO - import dependent lib
import json
import os
import argparse

model_path = os.environ["RESULT_DIR"]+"/model/saved_model"
print ("model_path: %s" %model_path)

# HPO - get hpo experiment hyper-parameter values from config.json
# The hyperparameters and the search space is defined when submitting the HPO task
# WMLA HPO will generate hpo experiment candidates and writes to config.json
try:
    hyper_params = json.loads(open("config.json").read())
    print('hyper_params: ', hyper_params)
    learning_rate = float(hyper_params.get("learning_rate", "0.01"))
except:
    print('failed to get hyper-parameters from config.json')
    learning_rate = 0.001
    pass




class HPOMetric(Callback):
    def on_train_begin(self, **kwargs):
        self.hpo_metrics = []

    def on_epoch_end(self, last_metrics, **kwargs):
        print('epoch:', kwargs['epoch'], 'val_loss', last_metrics[0].item(), 'train_loss:', kwargs['smooth_loss'].item())
        self.hpo_metrics.append((kwargs['epoch'], {'loss': last_metrics[0].item()}))

    def on_train_end(self, **kwargs):
        training_out = []
        for hpo_metric in self.hpo_metrics:
            out = {'steps': hpo_metric[0]}
            for (metric, value) in hpo_metric[1].items():
                out[metric] = value
            training_out.append(out)

        with open('{}/val_dict_list.json'.format(os.environ['RESULT_DIR']), 'w') as f:
        # with open('val_dict_list.json', 'w') as f:
            json.dump(training_out, f)



def main():
    parser = argparse.ArgumentParser(description='Fastai MNIST Example')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='input epochs for training (default: 64)')

    args, unknow = parser.parse_known_args()
    print('args: ', args)
    print('epochs args:', args.epochs)

    path = untar_data(URLs.MNIST_SAMPLE)
    tfms = (rand_pad(2, 28), [])
    data = ImageDataBunch.from_folder(path, ds_tfms=tfms, bs=64).normalize(imagenet_stats)

    learn = create_cnn(data, models.resnet18, metrics=accuracy, callbacks=[HPOMetric()])
    learn.fit_one_cycle(args.epochs, learning_rate)


if __name__ == '__main__':
    main()


