# sent-bert model choice
# STS tasks
# baseline: paraphrase-distilroberta-base-v1
# QA: msmarco-distilroberta-base-v2
# Bert (wordPiece tokenization): stsb-distilbert-base

import pandas as pd
import torch
import math
from torch.utils.data import DataLoader
import spacy
import logging
import argparse
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models, util
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
# # try hacking sentence-transformers package
# from sentence_transformers_local.sentence_transformers import SentenceTransformer, LoggingHandler, losses, models, util
# from sentence_transformers_local.sentence_transformers.readers import InputExample
# from sentence_transformers_local.sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

torch.cuda.empty_cache()

# parse arguments: model type
argp = argparse.ArgumentParser()
argp.add_argument('variant',
    help="Which variant of the model to run ('vanilla' or 'synthesizer')",
    choices=["paraphrase-distilroberta-base-v1", "msmarco-distilroberta-base-v2", "stsb-distilbert-base", "msmarco-distilbert-base-v2"])
argp.add_argument('--output_path', default=None)
argp.add_argument('--corrupted', help='if use entity masked corpus', action="store_true")
argp.add_argument('--ner', help='if use entity concat', action="store_true")
args = argp.parse_args()

# Save the device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# NER prep
spacy_model = spacy.load("en_core_web_sm")
masking_map = {
	'EVENT': 'entity0',
	'FAC': 'entity1',
	 'GPE': 'entity2',
	 'LOC': 'entity3',
	 'NORP': 'entity4',
	 'ORG': 'entity5',
	 'PERSON': 'entity6',
	 'PRODUCT': 'entity7',
	 'WORK_OF_ART': 'entity8'
}

# prepare corpus
# clean_data_labels = ['auto-repair-appt', 'coffee-order', 'flight-search', 'food-order', 'hotel-search', 'movie-search', 'music-search', 'ride-book']
train_data_labels = ['auto-repair-appt', 'coffee-order', 'flight-search', 'food-order', 'movie-search', 'ride-book']
test_data_labels = ['auto-repair-appt', 'coffee-order', 'flight-search', 'food-order', 'hotel-search', 'movie-search', 'music-search', 'ride-book', 'pizza-order']

# # old data split: all labels in test corpus are also present in training corpus
# train_samples, dev_samples, test_samples = [], [], []
# for f in clean_data_labels:
#     filename = "./data/clean_csv/data_" + f + ".csv"
#     df = pd.read_csv(filename)[:900]
#     train_samples.append(df[:300])  
#     dev_samples.append(df[300 : 350])
#     test_samples.append(df[350:])
# train_corpus = pd.concat(train_samples)
# dev_corpus = pd.concat(dev_samples)
# test_corpus = pd.concat(test_samples)
# print("train corpus size: ", train_corpus.shape)
# print("dev corpus size: ", dev_corpus.shape)
# print("test corpus size: ", test_corpus.shape)

# improved data split: 3 labels present in test that're not in train
train_samples, dev_samples, test_samples = [], [], []
for f in test_data_labels:
    filename = "./data/clean_csv/data_" + f + ".csv"
    df = pd.read_csv(filename)[:900]
    if f in train_data_labels:
        train_samples.append(df[:360])
        dev_samples.append(df[360:430])
        test_samples.append(df[430:])
    else:
        test_samples.append(df[:470])
        dev_samples.append(df[470:540])
train_corpus = pd.concat(train_samples)
dev_corpus = pd.concat(dev_samples)
test_corpus = pd.concat(test_samples)
print("train corpus size: ", train_corpus.shape)
print("dev corpus size: ", dev_corpus.shape)
print("test corpus size: ", test_corpus.shape)

if args.ner:
	def concat_entity(u):
	    doc = spacy_model(u)
	    idxs = [0]
	    labels = []
	    ret = ""
	    for ent in doc.ents:
	        if ent.label_ in masking_map:
	            idxs.append(ent.start_char)
	            idxs.append(ent.end_char)
	            labels.append(ent.label_)
	    if (len(idxs) > 1 and len(labels) > 0):
	        for i in range(0, len(idxs) - 1, 2):
	            ret = ret + u[idxs[i] : idxs[i + 1]] + labels[int(i / 2)] + " / " + u[idxs[i + 1] : idxs[i + 2]]
	        ret += u[idxs[-1] :]
	        return ret
	    else :
	        return u
	train_corpus['utterance'] = train_corpus['utterance'].apply(concat_entity)
	dev_corpus['utterance'] = dev_corpus['utterance'].apply(concat_entity)
	test_corpus['utterance'] = test_corpus['utterance'].apply(concat_entity)
	print("entity-concat train corpus size: ", train_corpus.shape)
	print("entity-concat dev corpus size: ", dev_corpus.shape)
	print("entity-concat test corpus size: ", test_corpus.shape)

if args.corrupted:
	def mask_entity(u):
	    doc = spacy_model(u)
	    idxs = [0]
	    masks = []
	    ret = ""
	    for ent in doc.ents:
	        if ent.label_ in masking_map:
	            idxs.append(ent.start_char)
	            idxs.append(ent.end_char)
	            masks.append(masking_map[ent.label_])
	    if (len(idxs) > 1 and len(masks) > 0):
	        for i in range(0, len(idxs) - 1, 2):
	            ret = ret + u[idxs[i] : idxs[i + 1]] + masks[int(i / 2)]
	        ret += u[idxs[-1] :]
	        return ret
	    else :
	        return u
	train_corpus['utterance'] = train_corpus['utterance'].apply(mask_entity)
	dev_corpus['utterance'] = dev_corpus['utterance'].apply(mask_entity)
	test_corpus['utterance'] = test_corpus['utterance'].apply(mask_entity)
	print("corrupted train corpus size: ", train_corpus.shape)
	print("corrupted dev corpus size: ", dev_corpus.shape)
	print("corrupted test corpus size: ", test_corpus.shape)


# create training sentence pairs
assert len(train_corpus['label'].value_counts()) == len(train_data_labels)
train_pair_corpus = []
train_corpus = train_corpus.sample(frac=1) # randomize order first
# first, create similar pairs
for label in train_data_labels:
    same_label_data = train_corpus[train_corpus['label'] == label].reset_index(drop=True)
    utterance_list = same_label_data['utterance']
    for i in range(len(utterance_list) - 1):
        for j in range(i + 1, len(utterance_list)):
            utter1, utter2 = utterance_list[i], utterance_list[j]
            if (utter1 != utter2):
                inp_example = InputExample(texts=[utter1, utter2], label=1.0)   # TODO: should we use baseline model score as similarity score?
                train_pair_corpus.append(inp_example)
    print(len(train_pair_corpus))
num_sim_pairs = len(train_pair_corpus)
print("Number of similar pairs in training data: ", num_sim_pairs)
# then, create dissimilar pairs
for idx in range(len(train_data_labels) - 1):
    same_label_data = train_corpus[train_corpus['label'] == train_data_labels[idx]].reset_index(drop=True)[:180]
    diff_label_data = train_corpus[train_corpus['label'].isin(train_data_labels[idx + 1:])].reset_index(drop=True)[:180]
    for utter1 in same_label_data['utterance']:
        for utter2 in diff_label_data['utterance']:
            inp_example = InputExample(texts=[utter1, utter2], label=0.2)   # TODO: should we use baseline model score as similarity score?
            train_pair_corpus.append(inp_example)
train_pair_corpus = pd.Series(train_pair_corpus).sample(frac=1)
print("Number of dissimilar pairs in training data: ", len(train_pair_corpus) - num_sim_pairs)
print("total training data size: ", len(train_pair_corpus))

# create dev sentence pairs
dev_pair_corpus = []
dev_corpus = dev_corpus.sample(frac=1) # randomize order first
# first, create similar pairs
for label in test_data_labels:
    same_label_data = dev_corpus[dev_corpus['label'] == label].reset_index(drop=True)
    utterance_list = same_label_data['utterance']
    for i in range(len(utterance_list) - 1):
        for j in range(i + 1, len(utterance_list)):
            utter1, utter2 = utterance_list[i], utterance_list[j]
            if (utter1 != utter2):
                inp_example = InputExample(texts=[utter1, utter2], label=1.0)   # TODO: should we use baseline model score as similarity score?
                dev_pair_corpus.append(inp_example)
    print(len(dev_pair_corpus))
num_sim_dev_pairs = len(dev_pair_corpus)
print("Number of similar pairs in dev data: ", num_sim_dev_pairs)
# then, create dissimilar pairs
for idx in range(len(train_data_labels) - 1):
    same_label_data = dev_corpus[dev_corpus['label'] == test_data_labels[idx]].reset_index(drop=True)[:25]
    diff_label_data = dev_corpus[dev_corpus['label'].isin(test_data_labels[idx + 1:])].reset_index(drop=True)[:25]
    for utter1 in same_label_data['utterance']:
        for utter2 in diff_label_data['utterance']:
            inp_example = InputExample(texts=[utter1, utter2], label=0.2)   # TODO: should we use baseline model score as similarity score?
            dev_pair_corpus.append(inp_example)
dev_pair_corpus = pd.Series(dev_pair_corpus).sample(frac=1)
print("Number of dissimilar pairs in dev data: ", len(dev_pair_corpus) - num_sim_dev_pairs)
print("total validation data size: ", len(dev_pair_corpus))

# modeling
model_variant = args.variant
print("model variant is: ", model_variant)
path = args.output_path
print("output path: ", path)
# word_embedding_model = models.Transformer("roberta-base")
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
#                                pooling_mode_mean_tokens=True,
#                                pooling_mode_cls_token=False,
#                                pooling_mode_max_tokens=False)
# model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
model = SentenceTransformer(model_variant)  # fine-tune on top of SBERT
model.to(device)
train_dataloader = DataLoader(train_pair_corpus, shuffle=True, batch_size=32)
train_loss = losses.CosineSimilarityLoss(model=model)
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_pair_corpus, name='dev')
warmup_steps = math.ceil(len(train_dataloader) * 4  * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=4,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=path)
    