
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import RobertaTokenizer, RobertaModel
import math
import random
import sklearn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from transformers import AdamW, get_linear_schedule_with_warmup



evolution_disagree_train = pd.read_csv("./IAC/evolution/disagree_train.csv", delimiter = ",", header = 0, index_col = 0)
evolution_neutral_train = pd.read_csv("./IAC/evolution/neutral_train.csv", delimiter = ",", header = 0, index_col = 0)
evolution_agree_train = pd.read_csv("./IAC/evolution/agree_train.csv", delimiter = ",", header = 0, index_col = 0)

abortion_disagree_train = pd.read_csv("./IAC/abortion/disagree_train.csv", delimiter = ",", header = 0, index_col = 0)
abortion_neutral_train = pd.read_csv("./IAC/abortion/neutral_train.csv", delimiter = ",", header = 0, index_col = 0)
abortion_agree_train = pd.read_csv("./IAC/abortion/agree_train.csv", delimiter = ",", header = 0, index_col = 0)

gun_control_disagree_train = pd.read_csv("./IAC/gun_control/disagree_train.csv", delimiter = ",", header = 0, index_col = 0)
gun_control_neutral_train = pd.read_csv("./IAC/gun_control/neutral_train.csv", delimiter = ",", header = 0, index_col = 0)
gun_control_agree_train = pd.read_csv("./IAC/gun_control/agree_train.csv", delimiter = ",", header = 0, index_col = 0)

gay_marriage_disagree_train = pd.read_csv("./IAC/gay_marriage/disagree_train.csv", delimiter = ",", header = 0, index_col = 0)
gay_marriage_neutral_train = pd.read_csv("./IAC/gay_marriage/neutral_train.csv", delimiter = ",", header = 0, index_col = 0)
gay_marriage_agree_train = pd.read_csv("./IAC/gay_marriage/agree_train.csv", delimiter = ",", header = 0, index_col = 0)

existence_of_God_disagree_train = pd.read_csv("./IAC/existence_of_God/disagree_train.csv", delimiter = ",", header = 0, index_col = 0)
existence_of_God_neutral_train = pd.read_csv("./IAC/existence_of_God/neutral_train.csv", delimiter = ",", header = 0, index_col = 0)
existence_of_God_agree_train = pd.read_csv("./IAC/existence_of_God/agree_train.csv", delimiter = ",", header = 0, index_col = 0)

healthcare_disagree_train = pd.read_csv("./IAC/healthcare/disagree_train.csv", delimiter = ",", header = 0, index_col = 0)
healthcare_neutral_train = pd.read_csv("./IAC/healthcare/neutral_train.csv", delimiter = ",", header = 0, index_col = 0)
healthcare_agree_train = pd.read_csv("./IAC/healthcare/agree_train.csv", delimiter = ",", header = 0, index_col = 0)

climate_change_disagree_train = pd.read_csv("./IAC/climate_change/disagree_train.csv", delimiter = ",", header = 0, index_col = 0)
climate_change_neutral_train = pd.read_csv("./IAC/climate_change/neutral_train.csv", delimiter = ",", header = 0, index_col = 0)
climate_change_agree_train = pd.read_csv("./IAC/climate_change/agree_train.csv", delimiter = ",", header = 0, index_col = 0)

death_penalty_disagree_train = pd.read_csv("./IAC/death_penalty/disagree_train.csv", delimiter = ",", header = 0, index_col = 0)
death_penalty_neutral_train = pd.read_csv("./IAC/death_penalty/neutral_train.csv", delimiter = ",", header = 0, index_col = 0)
death_penalty_agree_train = pd.read_csv("./IAC/death_penalty/agree_train.csv", delimiter = ",", header = 0, index_col = 0)

marijuana_legalization_disagree_train = pd.read_csv("./IAC/marijuana_legalization/disagree_train.csv", delimiter = ",", header = 0, index_col = 0)
marijuana_legalization_neutral_train = pd.read_csv("./IAC/marijuana_legalization/neutral_train.csv", delimiter = ",", header = 0, index_col = 0)
marijuana_legalization_agree_train = pd.read_csv("./IAC/marijuana_legalization/agree_train.csv", delimiter = ",", header = 0, index_col = 0)

communism_vs_capitalism_disagree_train = pd.read_csv("./IAC/communism_vs_capitalism/disagree_train.csv", delimiter = ",", header = 0, index_col = 0)
communism_vs_capitalism_neutral_train = pd.read_csv("./IAC/communism_vs_capitalism/neutral_train.csv", delimiter = ",", header = 0, index_col = 0)
communism_vs_capitalism_agree_train = pd.read_csv("./IAC/communism_vs_capitalism/agree_train.csv", delimiter = ",", header = 0, index_col = 0)


source_disagree_train = pd.concat([evolution_disagree_train, abortion_disagree_train, gun_control_disagree_train, gay_marriage_disagree_train, existence_of_God_disagree_train, healthcare_disagree_train, climate_change_disagree_train, death_penalty_disagree_train, marijuana_legalization_disagree_train, communism_vs_capitalism_disagree_train], axis = 0)
source_neutral_train = pd.concat([evolution_neutral_train, abortion_neutral_train, gun_control_neutral_train, gay_marriage_neutral_train, existence_of_God_neutral_train, healthcare_neutral_train, climate_change_neutral_train, death_penalty_neutral_train, marijuana_legalization_neutral_train, communism_vs_capitalism_neutral_train], axis = 0)
source_agree_train = pd.concat([evolution_agree_train, abortion_agree_train, gun_control_agree_train, gay_marriage_agree_train, existence_of_God_agree_train, healthcare_agree_train, climate_change_agree_train, death_penalty_agree_train, marijuana_legalization_agree_train, communism_vs_capitalism_agree_train], axis = 0)

target_disagree_train = pd.read_csv("./AWTP/disagree_train.csv", delimiter = ",", header = 0)
target_neutral_train = pd.read_csv("./AWTP/neutral_train.csv", delimiter = ",", header = 0)
target_agree_train = pd.read_csv("./AWTP/agree_train.csv", delimiter = ",", header = 0)
target_disagree_dev = pd.read_csv("./AWTP/disagree_dev.csv", delimiter = ",", header = 0)
target_neutral_dev = pd.read_csv("./AWTP/neutral_dev.csv", delimiter = ",", header = 0)
target_agree_dev = pd.read_csv("./AWTP/agree_dev.csv", delimiter = ",", header = 0)
target_disagree_test = pd.read_csv("./AWTP/disagree_test.csv", delimiter = ",", header = 0)
target_neutral_test = pd.read_csv("./AWTP/neutral_test.csv", delimiter = ",", header = 0)
target_agree_test = pd.read_csv("./AWTP/agree_test.csv", delimiter = ",", header = 0)








MAX_LEN = 128
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


class custom_dataset(Dataset):
    def __init__(self, dataframe):
        self.custom_dataset = dataframe
        self.custom_dataset_quote = list(self.custom_dataset['quote_text'])
        self.custom_dataset_response = list(self.custom_dataset['response_text'])
        self.custom_dataset_label = list(self.custom_dataset['relation'])

    def __len__(self):
        return self.custom_dataset.shape[0]

    def __getitem__(self, idx):
        quote_text = self.custom_dataset_quote[idx]
        response_text = self.custom_dataset_response[idx]
        label = self.custom_dataset_label[idx]

        quote_encoding = tokenizer.encode_plus(quote_text, return_tensors='pt', add_special_tokens = True, max_length = MAX_LEN, padding = 'max_length', truncation = True, return_token_type_ids = True, return_attention_mask = True)
        quote_input_ids = quote_encoding['input_ids'].view(MAX_LEN)
        quote_token_type_ids = quote_encoding['token_type_ids'].view(MAX_LEN)
        quote_attention_mask = quote_encoding['attention_mask'].view(MAX_LEN)

        response_encoding = tokenizer.encode_plus(response_text, return_tensors='pt', add_special_tokens = True, max_length = MAX_LEN, padding = 'max_length', truncation = True, return_token_type_ids = True, return_attention_mask = True)
        response_input_ids = response_encoding['input_ids'].view(MAX_LEN)
        response_token_type_ids = response_encoding['token_type_ids'].view(MAX_LEN)
        response_attention_mask = response_encoding['attention_mask'].view(MAX_LEN)

        diction = {"quote_input_ids": quote_input_ids, "quote_token_type_ids": quote_token_type_ids, "quote_attention_mask": quote_attention_mask,
                   "response_input_ids": response_input_ids, "response_token_type_ids": response_token_type_ids, "response_attention_mask": response_attention_mask,
                   "label": label}

        return diction



evolution_disagree_train_dataset = custom_dataset(evolution_disagree_train)
evolution_neutral_train_dataset = custom_dataset(evolution_neutral_train)
evolution_agree_train_dataset = custom_dataset(evolution_agree_train)

abortion_disagree_train_dataset = custom_dataset(abortion_disagree_train)
abortion_neutral_train_dataset = custom_dataset(abortion_neutral_train)
abortion_agree_train_dataset = custom_dataset(abortion_agree_train)

gun_control_disagree_train_dataset = custom_dataset(gun_control_disagree_train)
gun_control_neutral_train_dataset = custom_dataset(gun_control_neutral_train)
gun_control_agree_train_dataset = custom_dataset(gun_control_agree_train)

gay_marriage_disagree_train_dataset = custom_dataset(gay_marriage_disagree_train)
gay_marriage_neutral_train_dataset = custom_dataset(gay_marriage_neutral_train)
gay_marriage_agree_train_dataset = custom_dataset(gay_marriage_agree_train)

existence_of_God_disagree_train_dataset = custom_dataset(existence_of_God_disagree_train)
existence_of_God_neutral_train_dataset = custom_dataset(existence_of_God_neutral_train)
existence_of_God_agree_train_dataset = custom_dataset(existence_of_God_agree_train)

healthcare_disagree_train_dataset = custom_dataset(healthcare_disagree_train)
healthcare_neutral_train_dataset = custom_dataset(healthcare_neutral_train)
healthcare_agree_train_dataset = custom_dataset(healthcare_agree_train)

climate_change_disagree_train_dataset = custom_dataset(climate_change_disagree_train)
climate_change_neutral_train_dataset = custom_dataset(climate_change_neutral_train)
climate_change_agree_train_dataset = custom_dataset(climate_change_agree_train)

death_penalty_disagree_train_dataset = custom_dataset(death_penalty_disagree_train)
death_penalty_neutral_train_dataset = custom_dataset(death_penalty_neutral_train)
death_penalty_agree_train_dataset = custom_dataset(death_penalty_agree_train)

marijuana_legalization_disagree_train_dataset = custom_dataset(marijuana_legalization_disagree_train)
marijuana_legalization_neutral_train_dataset = custom_dataset(marijuana_legalization_neutral_train)
marijuana_legalization_agree_train_dataset = custom_dataset(marijuana_legalization_agree_train)

communism_vs_capitalism_disagree_train_dataset = custom_dataset(communism_vs_capitalism_disagree_train)
communism_vs_capitalism_neutral_train_dataset = custom_dataset(communism_vs_capitalism_neutral_train)
communism_vs_capitalism_agree_train_dataset = custom_dataset(communism_vs_capitalism_agree_train)


target_disagree_train_dataset = custom_dataset(target_disagree_train)
target_neutral_train_dataset = custom_dataset(target_neutral_train)
target_agree_train_dataset = custom_dataset(target_agree_train)

target_disagree_dev_dataset = custom_dataset(target_disagree_dev)
target_neutral_dev_dataset = custom_dataset(target_neutral_dev)
target_agree_dev_dataset = custom_dataset(target_agree_dev)

target_disagree_test_dataset = custom_dataset(target_disagree_test)
target_neutral_test_dataset = custom_dataset(target_neutral_test)
target_agree_test_dataset = custom_dataset(target_agree_test)






def form_sets_threeclasses(tasklist_disagree, tasklist_neutral, tasklist_agree, idx):

    disagree_quote_input_ids = tasklist_disagree[idx]['quote_input_ids'] # 5 * 64
    disagree_quote_token_type_ids = tasklist_disagree[idx]['quote_token_type_ids']
    disagree_quote_attention_mask = tasklist_disagree[idx]['quote_attention_mask']
    disagree_response_input_ids = tasklist_disagree[idx]['response_input_ids']
    disagree_response_token_type_ids = tasklist_disagree[idx]['response_token_type_ids']
    disagree_response_attention_mask = tasklist_disagree[idx]['response_attention_mask']
    disagree_label = tasklist_disagree[idx]['label']

    neutral_quote_input_ids = tasklist_neutral[idx]['quote_input_ids'] # 5 * 64
    neutral_quote_token_type_ids = tasklist_neutral[idx]['quote_token_type_ids']
    neutral_quote_attention_mask = tasklist_neutral[idx]['quote_attention_mask']
    neutral_response_input_ids = tasklist_neutral[idx]['response_input_ids']
    neutral_response_token_type_ids = tasklist_neutral[idx]['response_token_type_ids']
    neutral_response_attention_mask = tasklist_neutral[idx]['response_attention_mask']
    neutral_label = tasklist_neutral[idx]['label']

    agree_quote_input_ids = tasklist_agree[idx]['quote_input_ids'] # 5 * 64
    agree_quote_token_type_ids = tasklist_agree[idx]['quote_token_type_ids']
    agree_quote_attention_mask = tasklist_agree[idx]['quote_attention_mask']
    agree_response_input_ids = tasklist_agree[idx]['response_input_ids']
    agree_response_token_type_ids = tasklist_agree[idx]['response_token_type_ids']
    agree_response_attention_mask = tasklist_agree[idx]['response_attention_mask']
    agree_label = tasklist_agree[idx]['label']

    quote_input_ids = torch.cat((disagree_quote_input_ids, neutral_quote_input_ids, agree_quote_input_ids), dim = 0)
    quote_token_type_ids = torch.cat((disagree_quote_token_type_ids, neutral_quote_token_type_ids, agree_quote_token_type_ids), dim = 0)
    quote_attention_mask = torch.cat((disagree_quote_attention_mask, neutral_quote_attention_mask, agree_quote_attention_mask), dim = 0)
    response_input_ids = torch.cat((disagree_response_input_ids, neutral_response_input_ids, agree_response_input_ids), dim = 0)
    response_token_type_ids = torch.cat((disagree_response_token_type_ids, neutral_response_token_type_ids, agree_response_token_type_ids), dim = 0)
    response_attention_mask = torch.cat((disagree_response_attention_mask, neutral_response_attention_mask, agree_response_attention_mask), dim = 0)
    label = torch.cat((disagree_label, neutral_label, agree_label))

    return quote_input_ids, quote_token_type_ids, quote_attention_mask, response_input_ids, response_token_type_ids, response_attention_mask, label




def form_sets_single(tasklist, idx):

    quote_input_ids = tasklist[idx]['quote_input_ids'] # 5 * 64
    quote_token_type_ids = tasklist[idx]['quote_token_type_ids']
    quote_attention_mask = tasklist[idx]['quote_attention_mask']
    response_input_ids = tasklist[idx]['response_input_ids']
    response_token_type_ids = tasklist[idx]['response_token_type_ids']
    response_attention_mask = tasklist[idx]['response_attention_mask']
    label = tasklist[idx]['label']

    return quote_input_ids, quote_token_type_ids, quote_attention_mask, response_input_ids, response_token_type_ids, response_attention_mask, label



def form_sets_diction(diction):

    quote_input_ids = diction['quote_input_ids'] # 5 * 64
    quote_token_type_ids = diction['quote_token_type_ids']
    quote_attention_mask = diction['quote_attention_mask']
    response_input_ids = diction['response_input_ids']
    response_token_type_ids = diction['response_token_type_ids']
    response_attention_mask = diction['response_attention_mask']
    label = diction['label']

    return quote_input_ids, quote_token_type_ids, quote_attention_mask, response_input_ids, response_token_type_ids, response_attention_mask, label



def form_dataloaders(disagree_dataset, neutral_dataset, agree_dataset, batch_size, shuffle_boolen, pop_boolen):

    disagree_dataloader = list(DataLoader(disagree_dataset, batch_size = batch_size, shuffle = shuffle_boolen))
    neutral_dataloader = list(DataLoader(neutral_dataset, batch_size = batch_size, shuffle = shuffle_boolen))
    agree_dataloader = list(DataLoader(agree_dataset, batch_size = batch_size, shuffle = shuffle_boolen))

    if pop_boolen:
        disagree_dataloader = disagree_dataloader[0:-1]
        neutral_dataloader = neutral_dataloader[0:-1]
        agree_dataloader = agree_dataloader[0:-1]

    return disagree_dataloader, neutral_dataloader, agree_dataloader






class Sentence_Words_Embeddings(nn.Module):
    # input: input_ids, token_type_ids, attention_mask
    # output: batch_size * number of tokens * 768
    def __init__(self):
        super(Sentence_Words_Embeddings, self).__init__()

        self.robertamodel = RobertaModel.from_pretrained("roberta-base", output_hidden_states = True, )

    def forward(self, input_ids, token_type_ids, attention_mask):

        outputs = self.robertamodel(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        hidden_states = outputs[2]
        token_embeddings_batch = torch.stack(hidden_states, dim=0) # 13 layer * batch_size * number of tokens * 768

        feature_matrix_batch = list()

        for i in range(input_ids.shape[0]):
            token_embeddings = token_embeddings_batch[:,i,:,:]
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
            token_embeddings = token_embeddings.permute(1,0,2) # number of tokens * 13 layer * 768

            token_vecs_sum = []
            for token in token_embeddings:
                sum_vec = torch.sum(token[-4:], dim=0)
                token_vecs_sum.append(sum_vec)
            token_vecs_sum = torch.stack(token_vecs_sum, dim=0) # number of tokens * 768, token embeddings within a sentence

            feature_matrix_batch.append(token_vecs_sum)

        feature_matrix_batch = torch.stack(feature_matrix_batch, dim = 0) # batch_size * number of tokens * 768

        return feature_matrix_batch



class FewShot_Model(nn.Module):
    def __init__(self):
        super(FewShot_Model, self).__init__()

        self.sentence_words_embeddings = Sentence_Words_Embeddings()

        self.bilstm = nn.LSTM(input_size=768, hidden_size=384, batch_first=True, bidirectional=True)

        self.query_dim_reduction = nn.Linear(768*2, 768, bias = True)
        nn.init.xavier_uniform_(self.query_dim_reduction.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.query_dim_reduction.bias)

        self.disagree_class_level = nn.Linear(768*2, 768, bias = True)
        nn.init.xavier_uniform_(self.disagree_class_level.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.disagree_class_level.bias)

        self.neutral_class_level = nn.Linear(768*2, 768, bias = True)
        nn.init.xavier_uniform_(self.neutral_class_level.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.neutral_class_level.bias)

        self.agree_class_level = nn.Linear(768*2, 768, bias = True)
        nn.init.xavier_uniform_(self.agree_class_level.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.agree_class_level.bias)

        self.query_att = nn.Linear(768, 256, bias = True)
        nn.init.xavier_uniform_(self.query_att.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.query_att.bias)

        self.support_att = nn.Linear(768, 256, bias = True)
        nn.init.xavier_uniform_(self.support_att.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.support_att.bias)

        self.att_final = nn.Linear(256, 1, bias = True)
        nn.init.xavier_uniform_(self.att_final.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.att_final.bias)

        self.relation_layer1 = nn.Linear(3075, 768, bias = True)
        nn.init.xavier_uniform_(self.relation_layer1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.relation_layer1.bias)

        self.relation_layer2 = nn.Linear(768, 256, bias = True)
        nn.init.xavier_uniform_(self.relation_layer2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.relation_layer2.bias)

        self.relation_layer3 = nn.Linear(256, 1, bias = True)
        nn.init.xavier_uniform_(self.relation_layer3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.relation_layer3.bias)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()

        self.few_shot_criterion = nn.MSELoss(reduction = 'sum')


    def qr_naive_embed_concat(self, quote_input_ids, quote_token_type_ids, quote_attention_mask, response_input_ids, response_token_type_ids, response_attention_mask):

        quote_words_embeddings = self.sentence_words_embeddings(quote_input_ids, quote_token_type_ids, quote_attention_mask) # batch_size * number of tokens * 768
        response_words_embeddings = self.sentence_words_embeddings(response_input_ids, response_token_type_ids, response_attention_mask) # batch_size * number of tokens * 768

        h0 = torch.zeros(2, quote_words_embeddings.shape[0], 384).cuda().requires_grad_()
        c0 = torch.zeros(2, quote_words_embeddings.shape[0], 384).cuda().requires_grad_()

        quote_words_embeddings_bilstm_output, (_,_) = self.bilstm(quote_words_embeddings, (h0, c0)) # batch_size * number of tokens * 768
        response_words_embeddings_bilstm_output, (_,_) = self.bilstm(response_words_embeddings, (h0, c0)) # batch_size * number of tokens * 768

        quote_bilstm_embed = quote_words_embeddings_bilstm_output[:, 0, :]
        response_bilstm_embed = response_words_embeddings_bilstm_output[:, 0, :]
        qr_naive_embed_concat = torch.cat((quote_bilstm_embed, response_bilstm_embed), dim=1)  # (batch_size, 768*2=1536) dim

        return qr_naive_embed_concat, quote_words_embeddings, response_words_embeddings

    def forward(self, support_quote_input_ids, support_quote_token_type_ids, support_quote_attention_mask, support_response_input_ids, support_response_token_type_ids, support_response_attention_mask, support_label, query_quote_input_ids, query_quote_token_type_ids, query_quote_attention_mask, query_response_input_ids, query_response_token_type_ids, query_response_attention_mask, query_label):

        support_disagree, support_disagree_quote_words_embeddings, support_disagree_response_words_embeddings = self.qr_naive_embed_concat(support_quote_input_ids[support_label == 0], support_quote_token_type_ids[support_label == 0], support_quote_attention_mask[support_label == 0], support_response_input_ids[support_label == 0], support_response_token_type_ids[support_label == 0], support_response_attention_mask[support_label == 0])
        support_neutral, support_neutral_quote_words_embeddings, support_neutral_response_words_embeddings = self.qr_naive_embed_concat(support_quote_input_ids[support_label == 1], support_quote_token_type_ids[support_label == 1], support_quote_attention_mask[support_label == 1], support_response_input_ids[support_label == 1], support_response_token_type_ids[support_label == 1], support_response_attention_mask[support_label == 1])
        support_agree, support_agree_quote_words_embeddings, support_agree_response_words_embeddings = self.qr_naive_embed_concat(support_quote_input_ids[support_label == 2], support_quote_token_type_ids[support_label == 2], support_quote_attention_mask[support_label == 2], support_response_input_ids[support_label == 2], support_response_token_type_ids[support_label == 2], support_response_attention_mask[support_label == 2])
        query, query_quote_words_embeddings, query_response_words_embeddings = self.qr_naive_embed_concat(query_quote_input_ids, query_quote_token_type_ids, query_quote_attention_mask, query_response_input_ids, query_response_token_type_ids, query_response_attention_mask)

        support_disagree_transformed = self.disagree_class_level(support_disagree) # K * 768
        support_neutral_transformed = self.neutral_class_level(support_neutral)
        support_agree_transformed = self.agree_class_level(support_agree)
        query_transformed = self.query_dim_reduction(query) # 3K * 768

        support_disagree_transformed = support_disagree_transformed.view(1, support_disagree_transformed.shape[0], support_disagree_transformed.shape[1]) # 1 * K * 768
        support_neutral_transformed = support_neutral_transformed.view(1, support_neutral_transformed.shape[0], support_neutral_transformed.shape[1])
        support_agree_transformed = support_agree_transformed.view(1, support_agree_transformed.shape[0], support_agree_transformed.shape[1]) # 1 * K * 768
        query_transformed = query_transformed.view(query_transformed.shape[0], 1, query_transformed.shape[1]) # 3K * 1 * 768
        query_transformed = query_transformed.repeat(1, support_disagree_transformed.shape[1], 1) # 3K * K * 768

        disagree_att = self.softmax(self.att_final(self.relu(self.support_att(support_disagree_transformed) + self.query_att(query_transformed)))) # 3K * K * 1
        disagree_class_embed = torch.mm(disagree_att[:,:,0], support_disagree_transformed.view(support_disagree_transformed.shape[1], support_disagree_transformed.shape[2])) # 3K * 768
        disagree_relation_feature = torch.cat((query_transformed[:,0,:], disagree_class_embed, torch.sub(disagree_class_embed, query_transformed[:,0,:]), torch.linalg.norm(torch.sub(disagree_class_embed, query_transformed[:,0,:]), dim = 1).view(disagree_class_embed.shape[0], 1), torch.mul(disagree_class_embed, query_transformed[:,0,:]), torch.sum(torch.mul(disagree_class_embed, query_transformed[:,0,:]), dim = 1).view(disagree_class_embed.shape[0], 1), torch.nn.functional.cosine_similarity(query_transformed[:,0,:], disagree_class_embed, dim = 1).view(disagree_class_embed.shape[0], 1)), dim = 1) # 3K * (768 * 4 + 3 = 3075)

        neutral_att = self.softmax(self.att_final(self.relu(self.support_att(support_neutral_transformed) + self.query_att(query_transformed)))) # 3K * K * 1
        neutral_class_embed = torch.mm(neutral_att[:,:,0], support_neutral_transformed.view(support_neutral_transformed.shape[1], support_neutral_transformed.shape[2])) # 3K * 768
        neutral_relation_feature = torch.cat((query_transformed[:,0,:], neutral_class_embed, torch.sub(neutral_class_embed, query_transformed[:,0,:]), torch.linalg.norm(torch.sub(neutral_class_embed, query_transformed[:,0,:]), dim = 1).view(neutral_class_embed.shape[0], 1), torch.mul(neutral_class_embed, query_transformed[:,0,:]), torch.sum(torch.mul(neutral_class_embed, query_transformed[:,0,:]), dim = 1).view(neutral_class_embed.shape[0], 1), torch.nn.functional.cosine_similarity(query_transformed[:,0,:], neutral_class_embed, dim = 1).view(neutral_class_embed.shape[0], 1)), dim = 1) # 3K * (768 * 4 + 3 = 3075)

        agree_att = self.softmax(self.att_final(self.relu(self.support_att(support_agree_transformed) + self.query_att(query_transformed)))) # 3K * K * 1
        agree_class_embed = torch.mm(agree_att[:,:,0], support_agree_transformed.view(support_agree_transformed.shape[1], support_agree_transformed.shape[2])) # 3K * 768
        agree_relation_feature = torch.cat((query_transformed[:,0,:], agree_class_embed, torch.sub(agree_class_embed, query_transformed[:,0,:]), torch.linalg.norm(torch.sub(agree_class_embed, query_transformed[:,0,:]), dim = 1).view(agree_class_embed.shape[0], 1), torch.mul(agree_class_embed, query_transformed[:,0,:]), torch.sum(torch.mul(agree_class_embed, query_transformed[:,0,:]), dim = 1).view(agree_class_embed.shape[0], 1), torch.nn.functional.cosine_similarity(query_transformed[:,0,:], agree_class_embed, dim = 1).view(agree_class_embed.shape[0], 1)), dim = 1) # 3K * (768 * 4 + 3 = 3075)

        disagree_relation_feature = disagree_relation_feature.view(disagree_relation_feature.shape[0], 1, disagree_relation_feature.shape[1])
        neutral_relation_feature = neutral_relation_feature.view(neutral_relation_feature.shape[0], 1, neutral_relation_feature.shape[1])
        agree_relation_feature = agree_relation_feature.view(agree_relation_feature.shape[0], 1, agree_relation_feature.shape[1]) # 3K * 1 * (768 * 4 + 3 = 3075)

        relation_feature = torch.cat((disagree_relation_feature, neutral_relation_feature, agree_relation_feature), dim = 1) # 3K * 3 * (768 * 4 + 3 = 3075)
        relation_scores = self.sigmoid(self.relation_layer3(self.sigmoid(self.relation_layer2(self.sigmoid(self.relation_layer1(relation_feature)))))) # 3K * 3 * 1
        relation_scores = relation_scores[:,:,0] # 3K * 3

        query_label_onehot = torch.nn.functional.one_hot(query_label, 3) # 3K * 3
        query_label_onehot = query_label_onehot.to(torch.float32)
        few_shot_loss = self.few_shot_criterion(relation_scores, query_label_onehot)

        return relation_scores, few_shot_loss, \
               support_disagree_quote_words_embeddings, support_disagree_response_words_embeddings, \
               support_neutral_quote_words_embeddings, support_neutral_response_words_embeddings, \
               support_agree_quote_words_embeddings, support_agree_response_words_embeddings, \
               query_quote_words_embeddings, query_response_words_embeddings










import time
import datetime

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))




def evaluate(fewshot_model, target_disagree_eval_dataloader, target_neutral_eval_dataloader, target_agree_eval_dataloader, num_supports, verbose):

    fewshot_model.eval()

    if num_supports > min(len(target_disagree_train_dataloader), len(target_neutral_train_dataloader), len(target_agree_train_dataloader)):
        num_supports = min(len(target_disagree_train_dataloader), len(target_neutral_train_dataloader), len(target_agree_train_dataloader))

    accuracy_dev, disagree_precision_dev, disagree_recall_dev, disagree_F_dev, neutral_precision_dev, neutral_recall_dev, neutral_F_dev, agree_precision_dev, agree_recall_dev, agree_F_dev, macro_F_dev, macro_precision_dev, macro_recall_dev = [],[],[],[],[],[],[],[],[],[],[],[],[]

    for j in range(num_supports):

        support_quote_input_ids, support_quote_token_type_ids, support_quote_attention_mask, support_response_input_ids, support_response_token_type_ids, support_response_attention_mask, support_label = form_sets_threeclasses(target_disagree_train_dataloader, target_neutral_train_dataloader, target_agree_train_dataloader, j)
        support_quote_input_ids, support_quote_token_type_ids, support_quote_attention_mask, support_response_input_ids, support_response_token_type_ids, support_response_attention_mask, support_label = support_quote_input_ids.to(device), support_quote_token_type_ids.to(device), support_quote_attention_mask.to(device), support_response_input_ids.to(device), support_response_token_type_ids.to(device), support_response_attention_mask.to(device), support_label.to(device)

        for k in range(min(len(target_disagree_eval_dataloader), len(target_neutral_eval_dataloader), len(target_agree_eval_dataloader))):
            query_quote_input_ids, query_quote_token_type_ids, query_quote_attention_mask, query_response_input_ids, query_response_token_type_ids, query_response_attention_mask, query_label = form_sets_threeclasses(target_disagree_eval_dataloader, target_neutral_eval_dataloader, target_agree_eval_dataloader, k)
            query_quote_input_ids, query_quote_token_type_ids, query_quote_attention_mask, query_response_input_ids, query_response_token_type_ids, query_response_attention_mask, query_label = query_quote_input_ids.to(device), query_quote_token_type_ids.to(device), query_quote_attention_mask.to(device), query_response_input_ids.to(device), query_response_token_type_ids.to(device), query_response_attention_mask.to(device), query_label.to(device)

            with torch.no_grad():
                relation_scores,_,_,_,_,_,_,_,_,_ = fewshot_model(support_quote_input_ids, support_quote_token_type_ids, support_quote_attention_mask, support_response_input_ids, support_response_token_type_ids, support_response_attention_mask, support_label, query_quote_input_ids, query_quote_token_type_ids, query_quote_attention_mask, query_response_input_ids, query_response_token_type_ids, query_response_attention_mask, query_label)

            decision = torch.argmax(relation_scores, dim = 1).view(relation_scores.shape[0], 1)
            true_label = query_label.view(relation_scores.shape[0], 1)

            if k == 0:
                decision_onetest = decision
                true_label_onetest = true_label
            else:
                decision_onetest = torch.cat((decision_onetest, decision), dim = 0)
                true_label_onetest = torch.cat((true_label_onetest, true_label), dim = 0)

        decision_onetest = decision_onetest.to('cpu').numpy()
        true_label_onetest = true_label_onetest.to('cpu').numpy()

        accuracy = accuracy_score(true_label_onetest, decision_onetest)
        macro_metrics = precision_recall_fscore_support(true_label_onetest, decision_onetest, average='macro')
        macro_precision = macro_metrics[0]; macro_recall = macro_metrics[1]; macro_F = macro_metrics[2]
        metrics = precision_recall_fscore_support(true_label_onetest, decision_onetest, average=None)
        disagree_precision = metrics[0][0]; neutral_precision = metrics[0][1]; agree_precision = metrics[0][2]
        disagree_recall = metrics[1][0]; neutral_recall = metrics[1][1]; agree_recall = metrics[1][2]
        disagree_F = metrics[2][0]; neutral_F = metrics[2][1]; agree_F = metrics[2][2]

        accuracy_dev.append(accuracy); macro_F_dev.append(macro_F); macro_precision_dev.append(macro_precision); macro_recall_dev.append(macro_recall)
        disagree_precision_dev.append(disagree_precision); disagree_recall_dev.append(disagree_recall); disagree_F_dev.append(disagree_F)
        neutral_precision_dev.append(neutral_precision); neutral_recall_dev.append(neutral_recall); neutral_F_dev.append(neutral_F)
        agree_precision_dev.append(agree_precision); agree_recall_dev.append(agree_recall); agree_F_dev.append(agree_F)

    if verbose:
        print('accuracy: {:.3f}, macro precision: {:.3f}, macro recall: {:.3f}, macro F: {:.3f}'.format(np.nanmean(np.array(accuracy_dev)), np.nanmean(np.array(macro_precision_dev)), np.nanmean(np.array(macro_recall_dev)), np.nanmean(np.array(macro_F_dev))))
        print('disagree precision: {:.3f}, disagree recall: {:.3f}, disagree F: {:.3f}'.format(np.nanmean(np.array(disagree_precision_dev)), np.nanmean(np.array(disagree_recall_dev)), np.nanmean(np.array(disagree_F_dev))))
        print('neutral precision: {:.3f}, neutral recall: {:.3f}, neutral F: {:.3f}'.format(np.nanmean(np.array(neutral_precision_dev)), np.nanmean(np.array(neutral_recall_dev)), np.nanmean(np.array(neutral_F_dev))))
        print('agree precision: {:.3f}, agree recall: {:.3f}, agree F: {:.3f}'.format(np.nanmean(np.array(agree_precision_dev)), np.nanmean(np.array(agree_recall_dev)), np.nanmean(np.array(agree_F_dev))))

    return np.nanmean(np.array(macro_F_dev))






def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()




batch_size = 5
num_epochs = 2
no_decay = ['bias', 'LayerNorm.weight']
bert_weight_decay = 1e-2
non_bert_weight_decay = 1e-2
warmup_proportion = 0.1
non_bert_lr = 1e-4
bert_lr = 2e-5


seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

fewshot_model = FewShot_Model()
fewshot_model.cuda()

param_all = list(fewshot_model.named_parameters())
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_all if ((not any(nd in n for nd in no_decay)) and ('bert' in n)) ], 'lr': bert_lr, 'weight_decay': bert_weight_decay},
    {'params': [p for n, p in param_all if ((not any(nd in n for nd in no_decay)) and (not 'bert' in n)) ],  'lr': non_bert_lr, 'weight_decay': non_bert_weight_decay},
    {'params': [p for n, p in param_all if ((any(nd in n for nd in no_decay)) and ('bert' in n)) ], 'lr': bert_lr, 'weight_decay': 0.0},
    {'params': [p for n, p in param_all if ((any(nd in n for nd in no_decay)) and (not 'bert' in n))], 'lr': non_bert_lr, 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, eps = 1e-8)
num_train_steps = num_epochs * 372 # num_epochs * len(tasks)
warmup_steps = int(warmup_proportion * num_train_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps = num_train_steps)

target_disagree_train_dataloader, target_neutral_train_dataloader, target_agree_train_dataloader = form_dataloaders(target_disagree_train_dataset, target_neutral_train_dataset, target_agree_train_dataset, batch_size=batch_size, shuffle_boolen=False, pop_boolen=True)
target_disagree_dev_dataloader, target_neutral_dev_dataloader, target_agree_dev_dataloader = form_dataloaders(target_disagree_dev_dataset, target_neutral_dev_dataset, target_agree_dev_dataset, batch_size=batch_size, shuffle_boolen=False, pop_boolen=False)
target_disagree_test_dataloader, target_neutral_test_dataloader, target_agree_test_dataloader = form_dataloaders(target_disagree_test_dataset, target_neutral_test_dataset, target_agree_test_dataset, batch_size=batch_size, shuffle_boolen=False, pop_boolen=False)


best_macro_F = 0


for epoch_i in range(num_epochs):

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i, num_epochs))
    print('Training...')



    evolution_disagree_train_dataloader, evolution_neutral_train_dataloader, evolution_agree_train_dataloader = form_dataloaders(evolution_disagree_train_dataset, evolution_neutral_train_dataset, evolution_agree_train_dataset, batch_size=batch_size, shuffle_boolen=True, pop_boolen=True)
    abortion_disagree_train_dataloader, abortion_neutral_train_dataloader, abortion_agree_train_dataloader = form_dataloaders(abortion_disagree_train_dataset, abortion_neutral_train_dataset, abortion_agree_train_dataset, batch_size=batch_size, shuffle_boolen=True, pop_boolen=True)
    gun_control_disagree_train_dataloader, gun_control_neutral_train_dataloader, gun_control_agree_train_dataloader = form_dataloaders(gun_control_disagree_train_dataset, gun_control_neutral_train_dataset, gun_control_agree_train_dataset, batch_size=batch_size, shuffle_boolen=True, pop_boolen=True)
    gay_marriage_disagree_train_dataloader, gay_marriage_neutral_train_dataloader, gay_marriage_agree_train_dataloader = form_dataloaders(gay_marriage_disagree_train_dataset, gay_marriage_neutral_train_dataset, gay_marriage_agree_train_dataset, batch_size=batch_size, shuffle_boolen=True, pop_boolen=True)
    existence_of_God_disagree_train_dataloader, existence_of_God_neutral_train_dataloader, existence_of_God_agree_train_dataloader = form_dataloaders(existence_of_God_disagree_train_dataset, existence_of_God_neutral_train_dataset, existence_of_God_agree_train_dataset, batch_size=batch_size, shuffle_boolen=True, pop_boolen=True)
    healthcare_disagree_train_dataloader, healthcare_neutral_train_dataloader, healthcare_agree_train_dataloader = form_dataloaders(healthcare_disagree_train_dataset, healthcare_neutral_train_dataset, healthcare_agree_train_dataset, batch_size=batch_size, shuffle_boolen=True, pop_boolen=True)
    climate_change_disagree_train_dataloader, climate_change_neutral_train_dataloader, climate_change_agree_train_dataloader = form_dataloaders(climate_change_disagree_train_dataset, climate_change_neutral_train_dataset, climate_change_agree_train_dataset, batch_size=batch_size, shuffle_boolen=True, pop_boolen=True)
    death_penalty_disagree_train_dataloader, death_penalty_neutral_train_dataloader, death_penalty_agree_train_dataloader = form_dataloaders(death_penalty_disagree_train_dataset, death_penalty_neutral_train_dataset, death_penalty_agree_train_dataset, batch_size=batch_size, shuffle_boolen=True, pop_boolen=False)
    marijuana_legalization_disagree_train_dataloader, marijuana_legalization_neutral_train_dataloader, marijuana_legalization_agree_train_dataloader = form_dataloaders(marijuana_legalization_disagree_train_dataset, marijuana_legalization_neutral_train_dataset, marijuana_legalization_agree_train_dataset, batch_size=batch_size, shuffle_boolen=True, pop_boolen=False)
    communism_vs_capitalism_disagree_train_dataloader, communism_vs_capitalism_neutral_train_dataloader, communism_vs_capitalism_agree_train_dataloader = form_dataloaders(communism_vs_capitalism_disagree_train_dataset, communism_vs_capitalism_neutral_train_dataset, communism_vs_capitalism_agree_train_dataset, batch_size=batch_size, shuffle_boolen=True, pop_boolen=False)


    tasks = [] # training tasks

    for idx in range(len(evolution_agree_train_dataloader)):

        support_idx = idx
        query_idx = support_idx + 1
        if query_idx == len(evolution_agree_train_dataloader):
            query_idx = 0

        support_quote_input_ids, support_quote_token_type_ids, support_quote_attention_mask, support_response_input_ids, support_response_token_type_ids, support_response_attention_mask, support_label = form_sets_threeclasses(evolution_disagree_train_dataloader, evolution_neutral_train_dataloader, evolution_agree_train_dataloader, support_idx)
        query_quote_input_ids, query_quote_token_type_ids, query_quote_attention_mask, query_response_input_ids, query_response_token_type_ids, query_response_attention_mask, query_label = form_sets_threeclasses(evolution_disagree_train_dataloader, evolution_neutral_train_dataloader, evolution_agree_train_dataloader, query_idx)

        support_diction = {"quote_input_ids": support_quote_input_ids, "quote_token_type_ids": support_quote_token_type_ids, "quote_attention_mask": support_quote_attention_mask,
                           "response_input_ids": support_response_input_ids, "response_token_type_ids": support_response_token_type_ids, "response_attention_mask": support_response_attention_mask,
                           "label": support_label}
        query_diction = {"quote_input_ids": query_quote_input_ids, "quote_token_type_ids": query_quote_token_type_ids, "quote_attention_mask": query_quote_attention_mask,
                         "response_input_ids": query_response_input_ids, "response_token_type_ids": query_response_token_type_ids, "response_attention_mask": query_response_attention_mask,
                         "label": query_label}

        diction = {"support_diction": support_diction, "query_diction": query_diction}

        tasks.append(diction)
    for idx in range(len(abortion_agree_train_dataloader)):

        support_idx = idx
        query_idx = support_idx + 1
        if query_idx == len(abortion_agree_train_dataloader):
            query_idx = 0

        support_quote_input_ids, support_quote_token_type_ids, support_quote_attention_mask, support_response_input_ids, support_response_token_type_ids, support_response_attention_mask, support_label = form_sets_threeclasses(abortion_disagree_train_dataloader, abortion_neutral_train_dataloader, abortion_agree_train_dataloader, support_idx)
        query_quote_input_ids, query_quote_token_type_ids, query_quote_attention_mask, query_response_input_ids, query_response_token_type_ids, query_response_attention_mask, query_label = form_sets_threeclasses(abortion_disagree_train_dataloader, abortion_neutral_train_dataloader, abortion_agree_train_dataloader, query_idx)

        support_diction = {"quote_input_ids": support_quote_input_ids, "quote_token_type_ids": support_quote_token_type_ids, "quote_attention_mask": support_quote_attention_mask,
                           "response_input_ids": support_response_input_ids, "response_token_type_ids": support_response_token_type_ids, "response_attention_mask": support_response_attention_mask,
                           "label": support_label}
        query_diction = {"quote_input_ids": query_quote_input_ids, "quote_token_type_ids": query_quote_token_type_ids, "quote_attention_mask": query_quote_attention_mask,
                         "response_input_ids": query_response_input_ids, "response_token_type_ids": query_response_token_type_ids, "response_attention_mask": query_response_attention_mask,
                         "label": query_label}

        diction = {"support_diction": support_diction, "query_diction": query_diction}

        tasks.append(diction)
    for idx in range(len(gun_control_agree_train_dataloader)):

        support_idx = idx
        query_idx = support_idx + 1
        if query_idx == len(gun_control_agree_train_dataloader):
            query_idx = 0

        support_quote_input_ids, support_quote_token_type_ids, support_quote_attention_mask, support_response_input_ids, support_response_token_type_ids, support_response_attention_mask, support_label = form_sets_threeclasses(gun_control_disagree_train_dataloader, gun_control_neutral_train_dataloader, gun_control_agree_train_dataloader, support_idx)
        query_quote_input_ids, query_quote_token_type_ids, query_quote_attention_mask, query_response_input_ids, query_response_token_type_ids, query_response_attention_mask, query_label = form_sets_threeclasses(gun_control_disagree_train_dataloader, gun_control_neutral_train_dataloader, gun_control_agree_train_dataloader, query_idx)

        support_diction = {"quote_input_ids": support_quote_input_ids, "quote_token_type_ids": support_quote_token_type_ids, "quote_attention_mask": support_quote_attention_mask,
                           "response_input_ids": support_response_input_ids, "response_token_type_ids": support_response_token_type_ids, "response_attention_mask": support_response_attention_mask,
                           "label": support_label}
        query_diction = {"quote_input_ids": query_quote_input_ids, "quote_token_type_ids": query_quote_token_type_ids, "quote_attention_mask": query_quote_attention_mask,
                         "response_input_ids": query_response_input_ids, "response_token_type_ids": query_response_token_type_ids, "response_attention_mask": query_response_attention_mask,
                         "label": query_label}

        diction = {"support_diction": support_diction, "query_diction": query_diction}

        tasks.append(diction)

    for idx in range(len(gay_marriage_agree_train_dataloader)):

        support_idx = idx
        query_idx = support_idx + 1
        if query_idx == len(gay_marriage_agree_train_dataloader):
            query_idx = 0

        support_quote_input_ids, support_quote_token_type_ids, support_quote_attention_mask, support_response_input_ids, support_response_token_type_ids, support_response_attention_mask, support_label = form_sets_threeclasses(gay_marriage_disagree_train_dataloader, gay_marriage_neutral_train_dataloader, gay_marriage_agree_train_dataloader, support_idx)
        query_quote_input_ids, query_quote_token_type_ids, query_quote_attention_mask, query_response_input_ids, query_response_token_type_ids, query_response_attention_mask, query_label = form_sets_threeclasses(gay_marriage_disagree_train_dataloader, gay_marriage_neutral_train_dataloader, gay_marriage_agree_train_dataloader, query_idx)

        support_diction = {"quote_input_ids": support_quote_input_ids, "quote_token_type_ids": support_quote_token_type_ids, "quote_attention_mask": support_quote_attention_mask,
                           "response_input_ids": support_response_input_ids, "response_token_type_ids": support_response_token_type_ids, "response_attention_mask": support_response_attention_mask,
                           "label": support_label}
        query_diction = {"quote_input_ids": query_quote_input_ids, "quote_token_type_ids": query_quote_token_type_ids, "quote_attention_mask": query_quote_attention_mask,
                         "response_input_ids": query_response_input_ids, "response_token_type_ids": query_response_token_type_ids, "response_attention_mask": query_response_attention_mask,
                         "label": query_label}

        diction = {"support_diction": support_diction, "query_diction": query_diction}

        tasks.append(diction)

    for idx in range(len(existence_of_God_agree_train_dataloader)):

        support_idx = idx
        query_idx = support_idx + 1
        if query_idx == len(existence_of_God_agree_train_dataloader):
            query_idx = 0

        support_quote_input_ids, support_quote_token_type_ids, support_quote_attention_mask, support_response_input_ids, support_response_token_type_ids, support_response_attention_mask, support_label = form_sets_threeclasses(existence_of_God_disagree_train_dataloader, existence_of_God_neutral_train_dataloader, existence_of_God_agree_train_dataloader, support_idx)
        query_quote_input_ids, query_quote_token_type_ids, query_quote_attention_mask, query_response_input_ids, query_response_token_type_ids, query_response_attention_mask, query_label = form_sets_threeclasses(existence_of_God_disagree_train_dataloader, existence_of_God_neutral_train_dataloader, existence_of_God_agree_train_dataloader, query_idx)

        support_diction = {"quote_input_ids": support_quote_input_ids, "quote_token_type_ids": support_quote_token_type_ids, "quote_attention_mask": support_quote_attention_mask,
                           "response_input_ids": support_response_input_ids, "response_token_type_ids": support_response_token_type_ids, "response_attention_mask": support_response_attention_mask,
                           "label": support_label}
        query_diction = {"quote_input_ids": query_quote_input_ids, "quote_token_type_ids": query_quote_token_type_ids, "quote_attention_mask": query_quote_attention_mask,
                         "response_input_ids": query_response_input_ids, "response_token_type_ids": query_response_token_type_ids, "response_attention_mask": query_response_attention_mask,
                         "label": query_label}

        diction = {"support_diction": support_diction, "query_diction": query_diction}

        tasks.append(diction)
    for idx in range(len(healthcare_agree_train_dataloader)):

        support_idx = idx
        query_idx = support_idx + 1
        if query_idx == len(healthcare_agree_train_dataloader):
            query_idx = 0

        support_quote_input_ids, support_quote_token_type_ids, support_quote_attention_mask, support_response_input_ids, support_response_token_type_ids, support_response_attention_mask, support_label = form_sets_threeclasses(healthcare_disagree_train_dataloader, healthcare_neutral_train_dataloader, healthcare_agree_train_dataloader, support_idx)
        query_quote_input_ids, query_quote_token_type_ids, query_quote_attention_mask, query_response_input_ids, query_response_token_type_ids, query_response_attention_mask, query_label = form_sets_threeclasses(healthcare_disagree_train_dataloader, healthcare_neutral_train_dataloader, healthcare_agree_train_dataloader, query_idx)

        support_diction = {"quote_input_ids": support_quote_input_ids, "quote_token_type_ids": support_quote_token_type_ids, "quote_attention_mask": support_quote_attention_mask,
                           "response_input_ids": support_response_input_ids, "response_token_type_ids": support_response_token_type_ids, "response_attention_mask": support_response_attention_mask,
                           "label": support_label}
        query_diction = {"quote_input_ids": query_quote_input_ids, "quote_token_type_ids": query_quote_token_type_ids, "quote_attention_mask": query_quote_attention_mask,
                         "response_input_ids": query_response_input_ids, "response_token_type_ids": query_response_token_type_ids, "response_attention_mask": query_response_attention_mask,
                         "label": query_label}

        diction = {"support_diction": support_diction, "query_diction": query_diction}

        tasks.append(diction)
    for idx in range(len(climate_change_agree_train_dataloader)):

        support_idx = idx
        query_idx = support_idx + 1
        if query_idx == len(climate_change_agree_train_dataloader):
            query_idx = 0

        support_quote_input_ids, support_quote_token_type_ids, support_quote_attention_mask, support_response_input_ids, support_response_token_type_ids, support_response_attention_mask, support_label = form_sets_threeclasses(climate_change_disagree_train_dataloader, climate_change_neutral_train_dataloader, climate_change_agree_train_dataloader, support_idx)
        query_quote_input_ids, query_quote_token_type_ids, query_quote_attention_mask, query_response_input_ids, query_response_token_type_ids, query_response_attention_mask, query_label = form_sets_threeclasses(climate_change_disagree_train_dataloader, climate_change_neutral_train_dataloader, climate_change_agree_train_dataloader, query_idx)

        support_diction = {"quote_input_ids": support_quote_input_ids, "quote_token_type_ids": support_quote_token_type_ids, "quote_attention_mask": support_quote_attention_mask,
                           "response_input_ids": support_response_input_ids, "response_token_type_ids": support_response_token_type_ids, "response_attention_mask": support_response_attention_mask,
                           "label": support_label}
        query_diction = {"quote_input_ids": query_quote_input_ids, "quote_token_type_ids": query_quote_token_type_ids, "quote_attention_mask": query_quote_attention_mask,
                         "response_input_ids": query_response_input_ids, "response_token_type_ids": query_response_token_type_ids, "response_attention_mask": query_response_attention_mask,
                         "label": query_label}

        diction = {"support_diction": support_diction, "query_diction": query_diction}

        tasks.append(diction)
    for idx in range(len(death_penalty_agree_train_dataloader)):

        support_idx = idx
        query_idx = support_idx + 1
        if query_idx == len(death_penalty_agree_train_dataloader):
            query_idx = 0

        support_quote_input_ids, support_quote_token_type_ids, support_quote_attention_mask, support_response_input_ids, support_response_token_type_ids, support_response_attention_mask, support_label = form_sets_threeclasses(death_penalty_disagree_train_dataloader, death_penalty_neutral_train_dataloader, death_penalty_agree_train_dataloader, support_idx)
        query_quote_input_ids, query_quote_token_type_ids, query_quote_attention_mask, query_response_input_ids, query_response_token_type_ids, query_response_attention_mask, query_label = form_sets_threeclasses(death_penalty_disagree_train_dataloader, death_penalty_neutral_train_dataloader, death_penalty_agree_train_dataloader, query_idx)

        support_diction = {"quote_input_ids": support_quote_input_ids, "quote_token_type_ids": support_quote_token_type_ids, "quote_attention_mask": support_quote_attention_mask,
                           "response_input_ids": support_response_input_ids, "response_token_type_ids": support_response_token_type_ids, "response_attention_mask": support_response_attention_mask,
                           "label": support_label}
        query_diction = {"quote_input_ids": query_quote_input_ids, "quote_token_type_ids": query_quote_token_type_ids, "quote_attention_mask": query_quote_attention_mask,
                         "response_input_ids": query_response_input_ids, "response_token_type_ids": query_response_token_type_ids, "response_attention_mask": query_response_attention_mask,
                         "label": query_label}

        diction = {"support_diction": support_diction, "query_diction": query_diction}

        tasks.append(diction)
    for idx in range(len(marijuana_legalization_agree_train_dataloader)):

        support_idx = idx
        query_idx = support_idx + 1
        if query_idx == len(marijuana_legalization_agree_train_dataloader):
            query_idx = 0

        support_quote_input_ids, support_quote_token_type_ids, support_quote_attention_mask, support_response_input_ids, support_response_token_type_ids, support_response_attention_mask, support_label = form_sets_threeclasses(marijuana_legalization_disagree_train_dataloader, marijuana_legalization_neutral_train_dataloader, marijuana_legalization_agree_train_dataloader, support_idx)
        query_quote_input_ids, query_quote_token_type_ids, query_quote_attention_mask, query_response_input_ids, query_response_token_type_ids, query_response_attention_mask, query_label = form_sets_threeclasses(marijuana_legalization_disagree_train_dataloader, marijuana_legalization_neutral_train_dataloader, marijuana_legalization_agree_train_dataloader, query_idx)

        support_diction = {"quote_input_ids": support_quote_input_ids, "quote_token_type_ids": support_quote_token_type_ids, "quote_attention_mask": support_quote_attention_mask,
                           "response_input_ids": support_response_input_ids, "response_token_type_ids": support_response_token_type_ids, "response_attention_mask": support_response_attention_mask,
                           "label": support_label}
        query_diction = {"quote_input_ids": query_quote_input_ids, "quote_token_type_ids": query_quote_token_type_ids, "quote_attention_mask": query_quote_attention_mask,
                         "response_input_ids": query_response_input_ids, "response_token_type_ids": query_response_token_type_ids, "response_attention_mask": query_response_attention_mask,
                         "label": query_label}

        diction = {"support_diction": support_diction, "query_diction": query_diction}

        tasks.append(diction)
    for idx in range(len(communism_vs_capitalism_agree_train_dataloader)):

        support_idx = idx
        query_idx = support_idx + 1
        if query_idx == len(communism_vs_capitalism_agree_train_dataloader):
            query_idx = 0

        support_quote_input_ids, support_quote_token_type_ids, support_quote_attention_mask, support_response_input_ids, support_response_token_type_ids, support_response_attention_mask, support_label = form_sets_threeclasses(communism_vs_capitalism_disagree_train_dataloader, communism_vs_capitalism_neutral_train_dataloader, communism_vs_capitalism_agree_train_dataloader, support_idx)
        query_quote_input_ids, query_quote_token_type_ids, query_quote_attention_mask, query_response_input_ids, query_response_token_type_ids, query_response_attention_mask, query_label = form_sets_threeclasses(communism_vs_capitalism_disagree_train_dataloader, communism_vs_capitalism_neutral_train_dataloader, communism_vs_capitalism_agree_train_dataloader, query_idx)

        support_diction = {"quote_input_ids": support_quote_input_ids, "quote_token_type_ids": support_quote_token_type_ids, "quote_attention_mask": support_quote_attention_mask,
                           "response_input_ids": support_response_input_ids, "response_token_type_ids": support_response_token_type_ids, "response_attention_mask": support_response_attention_mask,
                           "label": support_label}
        query_diction = {"quote_input_ids": query_quote_input_ids, "quote_token_type_ids": query_quote_token_type_ids, "quote_attention_mask": query_quote_attention_mask,
                         "response_input_ids": query_response_input_ids, "response_token_type_ids": query_response_token_type_ids, "response_attention_mask": query_response_attention_mask,
                         "label": query_label}

        diction = {"support_diction": support_diction, "query_diction": query_diction}

        tasks.append(diction)



    random.shuffle(tasks)  # training tasks




    t0 = time.time()
    total_train_loss = 0
    num_batch = 0

    for i in range(len(tasks)):

        if i % (10 * num_epochs) == 0 and not i == 0:

            elapsed = format_time(time.time() - t0)
            avg_train_loss = total_train_loss / num_batch

            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Training Loss Average: {:.3f}'.format(i, len(tasks), elapsed, avg_train_loss))

            total_train_loss = 0
            num_batch = 0

            # test on target dev set

            macro_F = evaluate(fewshot_model, target_disagree_dev_dataloader, target_neutral_dev_dataloader, target_agree_dev_dataloader, num_supports = 10, verbose = 1)
            if macro_F > best_macro_F:
                torch.save(fewshot_model.state_dict(), './saved_models/fewshot_model_divide_domain_best.ckpt')
                best_macro_F = macro_F


        fewshot_model.train()

        support_quote_input_ids, support_quote_token_type_ids, support_quote_attention_mask, support_response_input_ids, support_response_token_type_ids, support_response_attention_mask, support_label = form_sets_diction(tasks[i]['support_diction'])
        query_quote_input_ids, query_quote_token_type_ids, query_quote_attention_mask, query_response_input_ids, query_response_token_type_ids, query_response_attention_mask, query_label = form_sets_diction(tasks[i]['query_diction'])
        support_quote_input_ids, support_quote_token_type_ids, support_quote_attention_mask, support_response_input_ids, support_response_token_type_ids, support_response_attention_mask, support_label = support_quote_input_ids.to(device), support_quote_token_type_ids.to(device), support_quote_attention_mask.to(device), support_response_input_ids.to(device), support_response_token_type_ids.to(device), support_response_attention_mask.to(device), support_label.to(device)
        query_quote_input_ids, query_quote_token_type_ids, query_quote_attention_mask, query_response_input_ids, query_response_token_type_ids, query_response_attention_mask, query_label = query_quote_input_ids.to(device), query_quote_token_type_ids.to(device), query_quote_attention_mask.to(device), query_response_input_ids.to(device), query_response_token_type_ids.to(device), query_response_attention_mask.to(device), query_label.to(device)

        optimizer.zero_grad()
        relation_scores, few_shot_loss,_,_,_,_,_,_,_,_ = fewshot_model(support_quote_input_ids, support_quote_token_type_ids, support_quote_attention_mask, support_response_input_ids, support_response_token_type_ids, support_response_attention_mask, support_label, query_quote_input_ids, query_quote_token_type_ids, query_quote_attention_mask, query_response_input_ids, query_response_token_type_ids, query_response_attention_mask, query_label)
        batch_loss = few_shot_loss
        total_train_loss += batch_loss.item()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(fewshot_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        num_batch = num_batch + 1

    training_time = format_time(time.time() - t0)
    print("")
    print("  Training epoch took: {:}".format(training_time))



    # test on target dev set

    macro_F = evaluate(fewshot_model, target_disagree_dev_dataloader, target_neutral_dev_dataloader, target_agree_dev_dataloader, num_supports = 10, verbose = 1)
    if macro_F > best_macro_F:
        torch.save(fewshot_model.state_dict(), './saved_models/fewshot_model_divide_domain_best.ckpt')
        best_macro_F = macro_F



# test on target test set

fewshot_model = FewShot_Model()
fewshot_model.cuda()

fewshot_model.load_state_dict(torch.load("./saved_models/fewshot_model_divide_domain_best.ckpt", map_location=device))

print("saved best model performance on test set")
macro_F = evaluate(fewshot_model, target_disagree_test_dataloader, target_neutral_test_dataloader, target_agree_test_dataloader, num_supports = 30, verbose = 1)














# stop here
