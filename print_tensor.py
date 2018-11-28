import tensorflow as tf
import numpy as np

reader = tf.train.NewCheckpointReader("./model/bert_model.ckpt")
tensor = reader.get_tensor('bert/embeddings/word_embeddings')
print(type(tensor), tensor.shape)
# np.save('./model/bert_word_embeddings.npy', tensor, allow_pickle=False)