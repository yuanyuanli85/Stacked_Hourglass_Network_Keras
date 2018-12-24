import argparse
import os
import tensorflow as tf
from keras import backend as k
from keras.models import model_from_json
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io


def main_tf_save(model_json, model_wegiths, output_model_path, output_format):
    # Disable learning
    k.set_learning_phase(0)
    # Set channel last
    k.set_image_data_format('channels_last')

    # Load model config and weights
    with open(model_json) as f:
        net_model = model_from_json(f.read())
    net_model.load_weights(model_wegiths)

    for input in net_model.input_layers:
        print 'input', input.name

    # network output
    num_output = len(net_model.output_layers)
    pred = [None] * num_output
    pred_node_names = [None] * num_output
    for i in range(num_output):
        pred_node_names[i] = 'out_' + str(i)
        pred[i] = tf.identity(net_model.outputs[i], name=pred_node_names[i])

    # freeze graph and write to file
    if output_format == 'graph':
        sess = k.get_session()
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)

        graph_io.write_graph(constant_graph,
                             output_model_path,
                             'tf.pb',
                             as_text=False)
    else:
        # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
        # tf.reset_default_graph()

        saver = tf.train.Saver()

        with k.get_session() as sess:
            k.set_learning_phase(0)
            # inference_graph = graph_util.remove_training_nodes(sess.graph.as_graph_def())
            # sess.run(inference_graph)
            saver.save(sess, os.path.join(output_model_path, '.ckpt'))
            tf.train.write_graph(sess.graph_def, output_model_path, 'graph.pb')

    print 'model saved to ', output_model_path


if __name__ == "__main__":
    # convert keras model into tensorflow pd file
    # visualize the network graph by using tensorboard
    # i.e python import_pb_to_tensorboard.py --model_dir /path/to/your/tf/pd --log_dir ./logs/

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuID", default=0, type=int, help='gpu id')
    parser.add_argument("--input_model_json", help="model json file")
    parser.add_argument("--input_model_weights", help="model weight file")
    parser.add_argument("--out_tf_path", help="place to store converted tf meta info")
    parser.add_argument("--out_format", default='meta', help="meta or graph pd")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)

    main_tf_save(args.input_model_json, args.input_model_weights, args.out_tf_path, args.out_format)
