from tensorflow.python.lib.io import file_io
from utils import *
from os import path

import tensorflow as tf
import pandas as pd
import numpy as np
import random
import pickle

def word2vec_review(int_word, int_to_word, work_dir, epochs=5, n_embedding=64):
    ## delete directory
    if file_io.is_directory( work_dir + '/w2v_model' ):
        file_io.delete_recursively( work_dir + '/w2v_model' )
    ## parameter
    epochs = epochs
    n_word = len(int_to_word)
    n_embedding = n_embedding # Number of embedding features 
    n_sampled = 15
    subsample_rate = 5e-4
    learning_rate_init = 0.001
    decay_rate = 0.7
    window_size=10
    
    ## word drop probability
    ## 빈번하게 발생하는 word의 샘플추출 확률은 줄이고 
    ## 희소한 word의 샘플추출 확률을 늘려 학습의 균형을 맞추기 위해 
    ## 단어의 샘플 확률을 계산한다.
    p_drop = word_drop_prob(int_word, threshold=subsample_rate)
    
    ### building the graph
    inputs = tf.placeholder(tf.int32, [None], name='inputs')
    labels = tf.placeholder(tf.int32, [None, None], name='labels')
    lr = tf.placeholder(tf.float32, name='lr')
    
    ## Embedding
    embedding = tf.get_variable("embedding", [n_word, n_embedding], 
                                initializer = tf.contrib.layers.xavier_initializer())
    softmax_w = tf.get_variable("softmax_w", [n_word, n_embedding], 
                                initializer = tf.contrib.layers.xavier_initializer())
    softmax_b = tf.get_variable("softmax_b", [n_word], initializer = tf.zeros_initializer())
    embed = tf.nn.embedding_lookup(embedding, inputs)
    
    # Calculate the loss using negative sampling
    loss = tf.nn.nce_loss(softmax_w, softmax_b, labels, embed, n_sampled, n_word)
    with tf.name_scope('cost'):
        cost = tf.reduce_mean(loss, name='cost')
        tf.summary.scalar('cost', cost)
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
    
    # validation sample
    valid_size = 10
    valid_examples = np.array(random.sample(range(len(int_to_word)//2), valid_size))
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
    # evaluation metric
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True))
    normalized_embedding = tf.div(embedding, norm, name="norm_embed")
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))
    
    ## Tensorboard
    summary_mg = tf.summary.merge_all()
    trn_writer = tf.summary.FileWriter(work_dir + '/w2v_model/tb/train', \
                                       graph=tf.get_default_graph())
    
    ## Saver
    saver = tf.train.Saver()
    saver.export_meta_graph(work_dir + '/w2v_model/abuse.ckpt.meta')
    
    ## Training
    init = tf.global_variables_initializer()
    sess = tf.Session()
    #graph = tf.Graph()
    #sess = tf.Session(graph=graph)
    sess.run(init)

    iteration = 1
    loss = 0

    tic = time.time()
    for e in range(epochs):
        train_word = subsampling(int_word, p_drop)
        batches = get_batches(train_word, wsize=window_size)
        learning_rate = np.power(decay_rate, e) * learning_rate_init
        btic = time.time()
        for x, y in batches:
            feed = {inputs: x, labels: np.array(y)[:, None], lr: learning_rate}
            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)
            loss += train_loss

            if iteration % 5000 == 0:
                summary = sess.run(summary_mg, feed_dict=feed)
                trn_writer.add_summary(summary, global_step=iteration)
                btoc = time.time()
                print("Epoch {}/{}".format(e+1, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss/1000),
                      "{:.4f} sec/batch".format((btoc-btic)/1000),
                      "{:.0f} total sec".format(time.time()-tic))
                loss = 0
                btic = time.time()

            if iteration % 100000 == 0:
                sim = sess.run(similarity)
                print_topk(sim, valid_examples, int_to_word, top_k=5)

            iteration += 1
            #if iteration==100000:
            #    break
        if e!=epochs: 
            saver.save(sess, work_dir+'/w2v_model/abuse.ckpt', global_step=e)
    trn_writer.close()
    sim = sess.run(similarity)
    print_topk(sim, valid_examples, int_to_word, top_k=5)
    saver.save(sess, work_dir+'/w2v_model/abuse.ckpt-last')    
    print('training finished')  
    sess.close()

## Import grahp
class ImportGraph():    
    """  Importing and running isolated TF graph """
    def __init__(self, model_dir):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            saver = tf.train.import_meta_graph(model_dir + "/abuse.ckpt.meta", clear_devices=True)
            saver.restore(self.sess, model_dir + "/abuse.ckpt-last")
            # There are TWO options how to get activation operation:
            # FROM SAVED COLLECTION:
            #self.activation = tf.get_collection('activation')[0]
            # BY NAME:
            self.embedding = self.graph.get_operation_by_name('embedding').outputs[0]
            self.softmax_w = self.graph.get_operation_by_name('softmax_w').outputs[0]
            self.softmax_b = self.graph.get_operation_by_name('softmax_b').outputs[0]
    
    def run(self):
        """ Running the activation operation previously imported """
        # The 'x' corresponds to name of input placeholder
        out = self.sess.run([self.embedding, self.softmax_w, self.softmax_b])
        return out
    
# visulizing embedding vector
def visualize_embedding(log_dir):
    ## setting dir
    model_dir = path.join(log_dir, 'w2v_model')
    tb_dir = path.join(model_dir, 'tb')
    embed_dir = path.join(tb_dir, 'embeddings')
    meta_dir = path.join(embed_dir, 'metadata.tsv')
    
    ## make dir
    if not file_io.is_directory(embed_dir): 
        file_io.create_dir(embed_dir)
        
    ## import model and embedding matrix
    model = ImportGraph(model_dir)
    embedding, softmax_w, softmax_b = model.run()
    embedding = tf.Variable(embedding, name = 'embedding')
    
    ## get meta data from bigquery cms and save
    with open(log_dir + '/word_dic.pkl', 'rb') as fp:
        int_to_word, word_to_int = pickle.load(fp)
    pd.Series([x[1] for x in int_to_word.items()]).to_csv(meta_dir, sep='\t', index=False)
    
    ## sess start
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    ## saver and writer
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(embed_dir, sess.graph)
    
    ## embedding
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embedding.name
    embedding_conf.metadata_path = meta_dir
    projector.visualize_embeddings(writer, config)
    saver.save(sess, embed_dir+'/embedding.ckpt')
    ## sess close
    
    sess.close()
    print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(tb_dir))