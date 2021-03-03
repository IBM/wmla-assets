#!/usr/bin/env python

import redhareapiversion
from redhareapi import Kernel
import json
import time
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os
import base64
from PIL import Image
from io import BytesIO

data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class MatchKernel(Kernel):
    def on_kernel_start(self, kernel_context):
        try:
            #Kernel.log_info("kernel id: " + kernel_context.get_id())
            Kernel.log_info("kernel input: " + kernel_context.get_model_description())
            model_desc = json.loads(kernel_context.get_model_description())
            model_file = os.path.join(model_desc['model_path'], "model_epoch_final.pth")
            self.model = models.__dict__["resnet18"]()
            self.model.fc = nn.Linear(512, 10)
            self.model.load_state_dict(torch.load(model_file))
        except Exception as e:
            Kernel.log_error(str(e))

    def on_task_invoke(self, task_context):
        try:
            in_invoke_time = time.time()
            Kernel.log_info("on_task_invoke")
            while task_context != None:
                input_data = json.loads(task_context.get_input_data())
                req_id = input_data['id']
                d = input_data["inputs"][0]
                #input_name = d["name"]
                data = d["data"]
                image_data = base64.b64decode(data)
                image_data = BytesIO(image_data)
                image_t = data_transforms(Image.open(image_data)).float()
                image_t = image_t.unsqueeze(0)
                outs = self.model(image_t)
                output_data = {"name":"output0", "datatype":"FP32", "shape": [1, 10], "data": outs.data[0].numpy().tolist()}
                task_context.set_output_data(json.dumps({"id": req_id, "outputs":[output_data]}))
                task_context = task_context.next()
            done_invoke_time = time.time()
            Kernel.log_info("Inference costs time: %.4f" % (done_invoke_time - in_invoke_time))
        except Exception as e:
            Kernel.log_error("Do task invoke failed: {}".format(e))

    def on_kernel_shutdown(self):
        Kernel.log_info('on_kernel_shutdown')

if __name__ == '__main__':
    obj_kernel = MatchKernel()
    obj_kernel.run()
