# Reference: https://github.com/gtfintechlab/fomc-hawkish-dovish/blob/main/code_model/bert_fine_tune_lm_hawkish_dovish_train_test.py


import pandas as pd

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
    BertTokenizer,
    BertForSequenceClassification,
    BertTokenizerFast,
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import pandas as pd
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

def load_data(train_data_path, test_data_path):
    # load training data
    data_df = pd.read_excel(train_data_path)
    sentences = data_df['sentence'].to_list()
    labels = data_df['label'].to_numpy()

    # load test data
    data_df_test = pd.read_excel(test_data_path)
    sentences_test = data_df_test['sentence'].to_list()
    labels_test = data_df_test['label'].to_numpy()

    return sentences, labels, sentences_test, labels_test


def load_tokenizer_model(language_model_to_use):
    if language_model_to_use == 'roberta':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', do_lower_case=True, do_basic_tokenize=True)
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3).to(device)
    elif language_model_to_use == 'flangroberta':
        tokenizer = AutoTokenizer.from_pretrained('SALT-NLP/FLANG-Roberta', do_lower_case=True, do_basic_tokenize=True)
        model = AutoModelForSequenceClassification.from_pretrained('SALT-NLP/FLANG-Roberta', num_labels=3).to(device)
    elif language_model_to_use == 'finbert':
        tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone', do_lower_case=True, do_basic_tokenize=True)
        model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3).to(device)
    elif language_model_to_use == 'flangbert':
        tokenizer = BertTokenizerFast.from_pretrained('SALT-NLP/FLANG-BERT', do_lower_case=True, do_basic_tokenize=True)
        model = BertForSequenceClassification.from_pretrained('SALT-NLP/FLANG-BERT', num_labels=3).to(device)
    elif language_model_to_use == 'roberta-large':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large', do_lower_case=True, do_basic_tokenize=True)
        model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=3).to(device)
    return tokenizer, model


def prepare_dataloader(language_model_to_use, sentences, labels, batch_size, return_train_dataloader):
    tokenizer, _ = load_tokenizer_model(language_model_to_use)

    sentence_input = []
    labels_output = []
    for i, sentence in enumerate(sentences):
        if isinstance(sentence, str):
            sentence_input.append(sentence)
            labels_output.append(labels[i])
        else:
            pass

    max_length=256

    if language_model_to_use == 'flangroberta':
        max_length=128

    tokens = tokenizer(sentence_input, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    labels = np.array(labels_output)
    labels = torch.LongTensor(labels)

    input_ids = tokens['input_ids']
    attention_masks = tokens['attention_mask']
    dataset = TensorDataset(input_ids, attention_masks, labels)

    if return_train_dataloader:
        val_length = int(len(dataset) * 0.2)
        train_length = len(dataset) - val_length
        print(f'Train Size: {train_length}, Validation Size: {val_length}')
        train, val = torch.utils.data.random_split(dataset=dataset, lengths=[train_length, val_length])
        dataloaders_dict = {'train': DataLoader(train, batch_size=batch_size, shuffle=True), 'val': DataLoader(val, batch_size=batch_size, shuffle=True)}
        return dataloaders_dict
    else:
        dataloaders_dict_test = {'test': DataLoader(dataset, batch_size=batch_size, shuffle=True)}
        print(f'Test Size: {len(dataset)}')
        return dataloaders_dict_test
