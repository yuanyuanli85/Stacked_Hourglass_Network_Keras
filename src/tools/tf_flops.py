import tensorflow as tf
from tensorflow.python.framework import graph_util


def load_graph(pbfile):
    with tf.gfile.GFile(pbfile, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph


if __name__ == "__main__":

    pbfile = '../../tf_models/tf.pb'
    g2 = load_graph(pbfile)

    #for node in g2.as_graph_def().node:
    #    print node.name

    with g2.as_default():
        flops = tf.profiler.profile(g2, options= tf.profiler.ProfileOptionBuilder.float_operation(), cmd='op')
        print flops.total_float_ops