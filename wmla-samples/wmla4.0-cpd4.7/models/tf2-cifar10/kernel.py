#!/usr/bin/env python
# -*- coding:utf-8 -*-
###############################################################################
# Licensed Materials - Property of IBM
# 5725-Y38
# @ Copyright IBM Corp. 2020 All Rights Reserved
# US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

try:
    import redhareapiversion
except:
    pass
from redhareapi import Kernel

import os
import json

import tensorflow as tf
import base64

#tf.compat.v1.enable_eager_execution()

'''
Note: This is not finalized yet, so it may not work for some saved models, such as 
model with multiple input
'''

class TFSavedModelKernel(Kernel):
    def on_kernel_start(self, kernel_context):
        model_desc = json.loads(kernel_context.get_model_description())

        model_path = model_desc['model_path']
        TFSavedModelKernel.log_info("model_path: " + model_path)
        saved_path = os.path.join(model_path, 'model')

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            TFSavedModelKernel.log_info("Using GPU: " + str(len(gpus)))
            # TODO: gpu split from model configure
            tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        self.model = tf.saved_model.load(saved_path)
        # tf v1
        #self.model = tf.saved_model.load_v2(saved_path)

    def on_task_invoke(self, task_context):
        try:
            while task_context != None:
                input_data = json.loads(task_context.get_input_data())
                #print(input_data)
                signature = input_data.get('signature_name')
                if not signature:
                    signature = 'serving_default'

                infer_func = self.model.signatures["serving_default"]

                output_dict = infer_func.structured_outputs
                print(output_dict.keys())

                data_type = input_data['data_type']
                if data_type == 'image:jpeg_uri' or data_type == 'image:raw_data':
                    im_files = [(x['key'], x['value']) for x in input_data['data_value']]
                else:
                    TFSavedModelKernel.log_error("unsupport the data type:" + data_type)
                    task_context = task_context.next()
                    continue

                # TODO: decode this info from input structure
                height = 32
                weight = 32
                channel = 3

                result = {'predictions': []}
                ims = []
                names = []
                for name, img in im_files:
                    if data_type == 'image:jpeg_uri':
                        if os.path.isfile(img):
                            image = tf.io.read_file(img)
                            image = tf.image.decode_jpeg(image, channels=channel)
                            im = tf.image.resize(image, [height, weight])
                        else:
                            import urllib
                            req = urllib.request.urlopen(img)
                            image = tf.image.decode_jpeg(req.read(), channels=channel)
                            im = tf.image.resize(image, [height, weight])
                    else:
                        img_rb = base64.b64decode(img)
                        image = tf.image.decode_jpeg(img_rb, channels=channel)
                        im = tf.image.resize(image, [height, weight])
                    im = tf.cast(im, tf.float32)
                    ims.append(im)
                    names.append(name)

                # im = np.random.random((1, 32,32, 3))
                # TODO: batching
                outputs = infer_func(tf.stack(ims))
                output = {'keys': names, 'results': {}}
                for key, value in outputs.items():
                    output['results'][key] = value.numpy().tolist()
                result['predictions'].append(output)

                task_context.set_output_data(json.dumps(result))
                task_context = task_context.next()

        except Exception as ex:
            task_context.set_output_data("Failed due to:" + str(ex));

    def on_kernel_shutdown(self):
        Kernel.log_info("on_kernel_shutdown")


if __name__ == '__main__':
    obj_kernel = TFSavedModelKernel()
    obj_kernel.run()

