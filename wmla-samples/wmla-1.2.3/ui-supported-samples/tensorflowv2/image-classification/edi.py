#!/usr/bin/env python

"""
edi.py Elastic Distributed Inference kernel for Tensorflow.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import redhareapiversion
from redhareapi import Kernel
from inference_helper import getClassificationResult
import os, re, json, time, base64, io
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf
from input_data import common_preprocessing
import shutil

class InferenceKernel(Kernel):
    def __init__(self):
        self.prob_thresh = 0.1
        self.model = None
        self.class_names = []
        self.output_dir = ""
        self.output_file = ""

    def on_kernel_start(self, kernel_context):
        try:
            print("on_kernel_start")
            print("mc instance id: " + kernel_context.get_instance_id())
            print("mc input content: " + kernel_context.get_model_description())
            model_desc = json.loads(kernel_context.get_model_description())
            print("loading inference model..")

            self.output_dir = model_desc['model_path']
            self.output_file = self.output_dir + "/result.json"
            model = model_desc['weight_path']
            if model == ' ':
                raise IOError(('error: Model not found.\n'))
            
            #self.model = tf.compat.v1.keras.experimental.load_from_saved_model(model)
            self.model = tf.saved_model.load(model)
            inf_func = self.model.signatures["serving_default"]
            

        except Exception as e:
            print("-------------------------")
            print(e)
            print("-------------------------")

    def on_task_invoke(self, task_context):
        try:
            print("===================================================")
            print('on_task_invoke')
            print("task id: " + task_context.get_id())
            print("session id: " + task_context.get_session_id())
            print("input data: " + task_context.get_input_data())
            input_data = json.loads(task_context.get_input_data())
            inf_func = self.model.signatures["serving_default"]
            attributes = input_data.get('attributes')
            prob_thresh = 0.0
            network= "vgg19"
            if attributes:
                for attribute in attributes:
                    if attribute['key'] == "threshold":
                        prob_thresh = attribute['value']
                        break
                    if attribute['key'] == "network":
                        network = attribute['value']
                        break
            
            tmp_img_path=self.output_dir+"/"+task_context.get_id()
            if not os.path.exists(tmp_img_path):
                os.mkdir(tmp_img_path)
                
            im_files = []
            im_keys = [x['key'] for x in input_data['data_value']]

            data_type=input_data['data_type']
            if data_type == 'image:raw_data':
                for im_key_idx, im_key in enumerate(im_keys):
                    img_byte = base64.b64decode(input_data['data_value'][im_key_idx]['value'])
                    tmp_img= os.path.join(tmp_img_path, im_key)
                    im_files.append(tmp_img)
                    with open(tmp_img, 'wb') as fw:
                        fw.write(img_byte)
                        fw.close()
                    
            else:
                im_files = [x['value'] for x in input_data['data_value']]

            results = []
           
            from input_data import data_factory

            imageBaseNames = [os.path.basename(image_file) for image_file in im_files]
            
            preprocessing_map = {
                'vgg19': common_preprocessing,
                'inceptionv3': common_preprocessing,
                'resnet50': common_preprocessing,
                'resnet101': common_preprocessing,
                'resnet152': common_preprocessing,
                'resnet50v2': common_preprocessing,
                'resnet101v2': common_preprocessing,
                'resnet152v2': common_preprocessing,
                'densenet121': common_preprocessing,
                'densenet169': common_preprocessing,
                'densenet201': common_preprocessing
            }
            preprocess_func = preprocessing_map[network].preprocess_bin_image_inference

            from uncompiled_models import models_factory
            image_size = models_factory.get_default_size(network)
            
            dataset = tf.data.Dataset.from_tensor_slices((imageBaseNames, im_files))
            dataset = dataset.map(lambda imageBaseNames, image_file: preprocess_func(imageBaseNames, image_file, image_size))
    
            dataset = dataset.batch(1)
            
            val_names = []
            predictions = []
            val_labels = []
            for step, (names, images, labels) in enumerate(dataset):
                #predictions.extend(self.model.predict(images.numpy()))
                predictions.extend(list(inf_func(images).values())[0].numpy())
                val_labels.extend(labels.numpy())
                if names.numpy()[0] != -1:
                    val_names.extend(names.numpy())
                else:
                    val_names.extend(range(step, step + len(labels)))
            
            for i in range(len(im_files)):
                for j in range(len(predictions[0])):
                    s={"sampleId":im_files[i], "label":str(j), "prob":float(predictions[i][j])}
                    results.append(s)
                
            output_dict = {"type": "classification", "result": results}
            with open(os.path.join(self.output_dir, self.output_file), 'w') as f:
            	json.dump(output_dict, f, indent=4)
            task_context.set_output_data(json.dumps(output_dict))
            shutil.rmtree(tmp_img_path)
            
        except Exception as e:
            print("-------------------------")
            print(e)
            print("-------------------------")

    def on_kernel_shutdown(self):
        print('on_kernel_shutdown')


if __name__ == '__main__':
    obj_kernel = InferenceKernel()
    obj_kernel.run()  
