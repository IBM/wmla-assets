import tensorflow as tf
#tf.enable_eager_execution()  #run in tf1
import argparse
import glob
import tf_parameter_mgr
import numpy

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='VGG19 inference')

    parser.add_argument('--model', dest='model', help='Model path',
                        required=True)
    parser.add_argument('--input_dir', dest='input_dir', help='Directory where the inference images are',
                        default=None)
    parser.add_argument('--image_file', dest='image_file', help='Particuarl image file to run inference upon',
                        default=None)
    parser.add_argument('--output_dir', dest='output_dir', help='Where to put the image with bbox drawed',
                        required=True)
    parser.add_argument('--label_file', dest='label_file', help='Where to put the image with bbox drawed')
    parser.add_argument('--output_file', dest='output_file', help='A .txt file contains the inference result with format as: sampleid label score xmin:ymin:xmax:ymax',
                        default=None)
    parser.add_argument('--prob_thresh', dest='prob_thresh', help='Threashold to control the minimal probility in the result',
                        default=0.8, type=float)
    parser.add_argument('--validate', dest='validate', help='Evaluating this model with validation dataset or not',
                        default=False, type=bool)
    
    ##### model parameters #######
    parser.add_argument('--network', dest='network', type=str, default='vgg19', 
                        help='name of the network used to train the classification model, \
                        one of vgg19, inceptionv3, mobilenetv2, resnet50, resnet101, resnet152, densenet')    

    args = parser.parse_args()

    return args


BATCH_SIZE = tf_parameter_mgr.getTrainBatchSize()

if __name__ == '__main__':
    args = parse_args()
    if args.validate:
      from input_data import data_factory
      filenames = tf_parameter_mgr.getValData()
      dataset = data_factory.get_dataset_from_tfrecords(args.network, filenames, BATCH_SIZE, is_train=False)

    else:
      from input_data import data_factory
      dataset = data_factory.get_dataset_from_images_inference(args.network, args.input_dir, BATCH_SIZE)#image_input_file()

    #restored_model = tf.keras.models.load_model(args.model + '/model_vgg19.h5')
    restored_model = tf.saved_model.load(args.model)
    inf_func = restored_model.signatures["serving_default"]

    val_names = []
    predictions = []
    val_labels = []
    for step, (names, images, labels) in enumerate(dataset):
        #predictions.extend(restored_model.predict(images.numpy()))
        predictions.extend(list(inf_func(images).values())[0].numpy())
        val_labels.extend(labels.numpy())
        if names.numpy()[0] != -1:
            val_names.extend(names.numpy())
        else:
            val_names.extend(range(step*BATCH_SIZE, step*BATCH_SIZE + len(labels)))
            
    import inference_helper,os
    if args.validate:
      inference_helper.writeClassificationResult(os.path.join(args.output_dir, args.output_file),
                                               val_names, numpy.array(predictions), ground_truth = val_labels, label_file = args.label_file)
    else:
      inference_helper.writeClassificationResult(os.path.join(args.output_dir, args.output_file),
                                               val_names, numpy.array(predictions),
                                               prob_thresh = args.prob_thresh,
                                               label_file = args.label_file)
    
