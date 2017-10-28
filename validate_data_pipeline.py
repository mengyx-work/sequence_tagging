
# coding: utf-8

# In[ ]:

import sys, os
sys.path.append(os.path.join(os.path.expanduser('~'), 'dev/sequence_tagging'))


# In[2]:

#sys.path


# In[3]:

os.path.join(os.path.expanduser('~'), 'dev/sequence_tagging')


# In[4]:

from model.config import Config
from model.data_utils import CoNLLDataset, get_vocabs, UNK, NUM,     get_glove_vocab, write_vocab, load_vocab, get_char_vocab,     export_trimmed_glove_vectors, get_processing_word


# In[5]:

config = Config(load=False)
processing_word = get_processing_word(lowercase=True)


# In[6]:

config.batch_size


# #### the `processing_word` funcion is passed into the `CoNLLDataset` initialization function

# In[7]:

delimiter_ ='\t'


# In[8]:

dev   = CoNLLDataset(config.filename_dev, delimiter_, processing_word)
test  = CoNLLDataset(config.filename_test, delimiter_, processing_word)
train = CoNLLDataset(config.filename_train, delimiter_, processing_word)


# In[9]:

train.filename


# In[10]:

vocab_words, vocab_tags = get_vocabs([train, dev, test])


# In[12]:

#vocab_words


# In[13]:

vocab_tags


# In[ ]:

vocab_glove = get_glove_vocab(config.filename_glove)


# In[ ]:




# #### in training step, a different config object is created, mainly the `load` function is used
#  1. different sets of data: words, tags and chars are loaded from pickle files
#  2. the `load` function creates different lambda function from `get_processing_word` for tag and word processing
#  2. embedding is loaded

# In[ ]:

config = Config()


# #### the lambda functions from `config` is passed into the `CoNLLDataset`

# In[ ]:

train = CoNLLDataset(config.filename_train, config.processing_word, config.processing_tag, config.max_iter)


# In[ ]:

for (x, y) in train:
    print '\n', len(x), x, y


# In[ ]:



