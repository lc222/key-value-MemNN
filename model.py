import tensorflow as tf
import numpy as np

def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)

class KVMemNN():

    def __init__(self, batch_size, vocab_size, entity_size, query_size,
                 memory_slot, embedding_size, hops=2, l2_lambda=None, name='KeyValueMenN2N'):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.query_size = query_size
        self.entity_size = entity_size
        self.memory_slot = memory_slot
        self.embedding_size = embedding_size
        self.hops = hops
        self.name = name
        self.encoding = tf.constant(position_encoding(self.query_size, self.embedding_size), name="encoding")
        self.build_inputs()

        with tf.variable_scope('query_encoded'):
            query_embed = tf.nn.embedding_lookup(self.A, self.query)
            query_encoded = tf.reduce_sum(query_embed*self.encoding, axis=1) #[None, query_size, embedding_size]-->[None, embedding_size]

        self.q = [query_encoded]

        with tf.variable_scope('key_value_encoded'):
            key_embed = tf.nn.embedding_lookup(self.A, self.key)
            key_encoded = tf.reduce_sum(key_embed, axis=2) #[None, memory_size, 2, embedding_size]-->[None, memory_size, embedding_size]
            value_embed = tf.nn.embedding_lookup(self.A, self.value) #[None, memory_size, embedding_size]

        for i in range(self.hops):
            with tf.variable_scope('hops_{}'.format(i)):
                q_temp = tf.expand_dims(self.q[-1], axis=-1) #[None, embedding_size, 1]
                q_temp = tf.transpose(q_temp, [0, 2, 1]) #[None, 1, embedding_szie]
                # softmax get the prob
                p = tf.nn.softmax(tf.reduce_sum(key_encoded*q_temp, axis=2)) #[None, memory_size]

                p_temp = tf.transpose(tf.expand_dims(p, axis=-1), [0, 2, 1]) #[None, 1, memory_size]
                value_temp = tf.transpose(value_embed, [0, 2, 1]) #[None, embedding_size, memory_size]
                o = tf.reduce_sum(value_temp*p_temp, axis=2) #[None, embedding_szie]

                R_temp = self.Rs[i]
                q_next = tf.matmul(self.q[-1] + o,R_temp)
                self.q.append(q_next)

        with tf.name_scope("output"):
            self.out = self.q[-1]
            logits = tf.nn.dropout(tf.matmul(self.out, self.B) + self.bais, keep_prob=self.keep_dropout) #[None, entity_size]

        with tf.name_scope("loss"):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.answer, name='loss'))

            if l2_lambda:
                vars = tf.trainable_variables()
                lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars])
                self.loss = cross_entropy + l2_lambda * lossL2
            else:
                self.loss = cross_entropy

        with tf.name_scope("accuracy"):
            #probs = tf.nn.softmax(tf.cast(logits, tf.float32))
            self.predict = tf.argmax(logits, 1, name="predict")
            correct_predictions = tf.equal(self.predict, tf.argmax(self.answer, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def build_inputs(self):
        self.query = tf.placeholder(tf.int32, [None, self.query_size], name='query')
        #self.query_entity = tf.placeholder(tf.int32, [None, self.query_entity_size], name='query_entity')
        self.answer = tf.placeholder(tf.float32, [None, self.entity_size], name='answer')
        self.key = tf.placeholder(tf.int32, [None, self.memory_slot, 2], name='key_memory')
        self.value = tf.placeholder(tf.int32, [None, self.memory_slot], name='value_memory')
        self.keep_dropout = tf.placeholder(tf.float32, name='keep_dropout')

        self.A = tf.get_variable('A', [self.vocab_size, self.embedding_size], initializer=tf.contrib.layers.xavier_initializer())
        self.B = tf.get_variable('B', [self.embedding_size, self.entity_size], initializer=tf.contrib.layers.xavier_initializer())
        self.bais = tf.Variable(tf.constant(0.1, tf.float32, [self.entity_size]), name='bais')
        self.Rs = []

        for i in range(self.hops):
            R = tf.get_variable('R_{}'.format(i), [self.embedding_size, self.embedding_size], initializer=tf.contrib.layers.xavier_initializer())
            self.Rs.append(R)
