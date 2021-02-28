# sent-bert model choice
# STS tasks
# baseline: paraphrase-distilroberta-base-v1 

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models, util
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# prepare corpus
dataset = pd.read_csv('data.csv')
dataset = dataset.sample(frac=1).reset_index(drop=True)
print(dataset.shape)

# basic corpus analysis
corpus_label_counts = dataset.value_counts(['label'])

# train_dev_test_split = [0.6, 0.2, 0.2]
train_corpus, dev_corpus, test_corpus = [], [], []
for idx in corpus_label_counts.index.tolist():
    label = idx[0]
    all_data = dataset[dataset['label'] == label].reset_index(drop=True)
    train_corpus.append(all_data[:int(len(all_data) * 0.6)])
    dev_corpus.append(all_data[int(len(all_data) * 0.6) : int(len(all_data) * 0.8)])
    test_corpus.append(all_data[int(len(all_data) * 0.8):])
train_corpus = pd.concat(train_corpus)
dev_corpus = pd.concat(dev_corpus)
test_corpus = pd.concat(test_corpus)
print("train data size: ", len(train_corpus), ", dev data size: ", len(dev_corpus), ", test data size: ", len(test_corpus))
print("train:\n")
print(train_corpus['label'].value_counts())
print("\n dev: \n")
print(dev_corpus['label'].value_counts())
print("\n test: \n")
print(test_corpus['label'].value_counts())

# create training sentence pairs
train_pair_corpus = []
for idx in corpus_label_counts.index.tolist():
    label = idx[0]
    train_label_data = train_corpus[train_corpus['label'] == label].reset_index(drop=True)
    print(len(train_label_data))
    for utter1 in train_label_data['utterance']:
        for utter2 in train_label_data['utterance']:
            if max(len(utter1), len(utter2)) > 2 * min(len(utter1), len(utter2)):
                inp_example = InputExample(texts=[utter1, utter2], label=0.2)  # todo: use baseline model's score
            else :
                inp_example = InputExample(texts=[utter1, utter2], label=1.0)
            train_pair_corpus.append(inp_example)
print(len(train_pair_corpus))

dev_pair_corpus = []
for idx in corpus_label_counts.index.tolist():
    label = idx[0]
    dev_label_data = dev_corpus[dev_corpus['label'] == label].reset_index(drop=True)
    print(len(dev_label_data))
    for utter1 in dev_label_data['utterance']:
        for utter2 in dev_label_data['utterance']:
            if max(len(utter1), len(utter2)) > 2 * min(len(utter1), len(utter2)):
                inp_example = InputExample(texts=[utter1, utter2], label=0.2)  # todo: use baseline model's score
            else :
                inp_example = InputExample(texts=[utter1, utter2], label=1.0)
            dev_pair_corpus.append(inp_example)
print(len(dev_pair_corpus))

word_embedding_model = models.Transformer("roberta-base")
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
train_dataloader = DataLoader(train_pair_corpus, shuffle=True, batch_size=256)
train_loss = losses.CosineSimilarityLoss(model=model)
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_pair_corpus, name='dev')
warmup_steps = math.ceil(len(train_dataloader) * 4  * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=4,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path="./models/crude")
    