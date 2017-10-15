#coding=utf-8
import tensorflow as tf
from data_util import *
from model import KVMemNN
import time
import datetime
import os

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.005, "Learning rate for Adam Optimizer.")
#flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
flags.DEFINE_float("max_grad_norm", 10.0, "Clip gradients to this norm.")

flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
flags.DEFINE_integer("max_slots", 300, "maximum slots in the memory")
flags.DEFINE_integer("hops", 2, "Number of hops in the Memory Network.")
flags.DEFINE_integer("epochs", 100, "Number of epochs to train for.")
flags.DEFINE_integer("embedding_size", 200, "Embedding size for embedding matrices.")
flags.DEFINE_integer("dropout_memory", 0.5, "keep probability for keeping a memory slot")

flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory [checkpoints]")
flags.DEFINE_integer("evaluate_every", 1000, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

FLAGS = flags.FLAGS

train_path = './data/wiki-entities_train_kv.txt'
test_path = './data/wiki-entities_test_kv.txt'
dev_path = './data/wiki-entities_dev_kv.txt'

#读取vocab与索引的字典文件，以便将数据转化为索引值
#word_idx = read_file_as_dict('./data/wiki-entities_word_idx.txt')
entity_idx = read_file_as_dict('./data/wiki-entities_entity_idx.txt')
#relation_idx = read_file_as_dict('./data/wiki-entities_relation_idx.txt')
vocab_idx = read_file_as_dict('./data/wiki-entities_idx.txt')

# 读取训练测试验证数据，获得问题/答案/关系/suorce/target等各项的最大长度并保存在maxLen（字典）。
maxlen = get_maxlen(train_path, test_path, dev_path)
# key和value的最大长度应该选择相关记忆的最大长度和max_slots中的较小值。即一个QA对与maxlen['keys]个memory slot相关联
maxlen['keys'], maxlen['values'] = min(maxlen['sources'], FLAGS.max_slots), min(maxlen['sources'], FLAGS.max_slots)
print 'maxlen got finished'

# 读取训练测试验证数据集，并将其转化为idx索引,准备feed进入网络进行训练
train_data = data_loader(train_path, vocab_idx, entity_idx)
test_data = data_loader(test_path, vocab_idx, entity_idx)
dev_data = data_loader(dev_path, vocab_idx, entity_idx)
print 'data load finished'

batch_size = FLAGS.batch_size
num_train = len(train_data)
# 得到每次feed进入网络的mini-batch数据的索引位置
batches = zip(range(0, num_train - batch_size, batch_size), range(batch_size, num_train, batch_size))
batches = [(start, end) for start, end in batches]

with tf.Session() as sess:
    KV_MemNN = KVMemNN(batch_size, len(vocab_idx), len(entity_idx), maxlen['question'], FLAGS.max_slots,
                       FLAGS.embedding_size, FLAGS.hops)
    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(0.001)
    grads_and_vars = optimizer.compute_gradients(KV_MemNN.loss)
    clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], FLAGS.max_grad_norm), gv[1]) for gv in grads_and_vars]
    train_op = optimizer.apply_gradients(clipped_grads_and_vars, global_step=global_step)

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", KV_MemNN.loss)
    acc_summary = tf.summary.scalar("accuracy", KV_MemNN.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    def train_step(batch_data):
        feed_dict = {
            KV_MemNN.query: batch_data['question'],
            KV_MemNN.answer: batch_data['answer'],
            KV_MemNN.key: batch_data['keys'],
            KV_MemNN.value: batch_data['values'],
            KV_MemNN.keep_dropout:0.8
        }
        #labels = tf.constant(batch_dict['answer'], tf.int64)
        _, step, summaries, loss, accuracy = sess.run(
            [train_op, global_step, train_summary_op, KV_MemNN.loss, KV_MemNN.accuracy],
            feed_dict)
        #accuracy = tf.contrib.metrics.accuracy(predict, labels)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        train_summary_writer.add_summary(summaries, step)

    def dev_step(batch_data, writer=None):
        feed_dict = {
            KV_MemNN.query: batch_data['question'],
            KV_MemNN.answer: batch_data['answer'],
            KV_MemNN.key: batch_data['keys'],
            KV_MemNN.value: batch_data['values'],
            KV_MemNN.keep_dropout: 1.0
        }
        #labels = tf.constant(batch_dict['answer'], tf.int64)
        step, summaries, loss, accuracy = sess.run(
            [global_step, dev_summary_op, KV_MemNN.loss, KV_MemNN.accuracy],
            feed_dict)
        #accuracy = tf.contrib.metrics.accuracy(predict, labels)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {}:g".format(time_str, step, loss, accuracy))
        if writer:
            writer.add_summary(summaries, step)

    for epoch in range(FLAGS.epochs):
        np.random.shuffle(batches)
        for start, end in batches:
            batch_examples = train_data[start:end]
            # 对每个batch的数据进行pad操作，而且如果有多个答案将其切分成单个答案
            batch_dict = prepare_batch(batch_examples, maxlen, batch_size, len(entity_idx))
            train_step(batch_dict)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                batch_dict = prepare_batch(dev_data, maxlen, len(dev_data), len(entity_idx))
                dev_step(batch_dict, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))