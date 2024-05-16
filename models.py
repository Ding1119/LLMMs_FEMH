import torch.nn as nn
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)
from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForSeq2SeqLM

class BertEncoder(nn.Module):
    def __init__(self, num_labels, numerical_feature_dim, models_type, attention_dim=4):    
        super(BertEncoder, self).__init__()
        # self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels, return_dict=True)
        # import pdb;pdb.set_trace()
        if models_type == 'robert':
            self.model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
        elif models_type == 'bert':
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        elif models_type == 'clinicalBERT':
            self.model = BertForSequenceClassification.from_pretrained('medicalai/ClinicalBERT', num_labels=num_labels)
        elif models_type == 'BiomedBERT':
            self.model = AutoModelForSequenceClassification.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", num_labels=num_labels)
        elif models_type == 'SciFive':
            self.model = AutoModelForSequenceClassification.from_pretrained("razent/SciFive-base-Pubmed", num_labels=num_labels)
        else:
            raise ValueError("Unsupported model type. Choose 'robert' or 'bert'.")
        
        self.dnn_layer = nn.Linear(numerical_feature_dim, 2)
        self.clf1 = nn.Linear(8, 2)
        self.self_attention = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=1)  # 调整embed_dim

    def forward(self, use_attention, early_prediction, input_ids, attention_mask, numerical_features):
        model_outputs = self.model(input_ids, attention_mask=attention_mask)
        model_last_hidden_state = model_outputs.logits
        if early_prediction == 'False':
            numerical_features = numerical_features.squeeze()
            dnn_output = self.dnn_layer(numerical_features)
        elif early_prediction == 'True':
            pass
        if use_attention == 'True':
            attention_input = torch.cat((model_last_hidden_state, dnn_output), dim=1).unsqueeze(0)
            attention_output, _ = self.self_attention(attention_input, attention_input, attention_input)
            attention_output = attention_output.squeeze(0)
            concatenated_output = torch.cat((model_last_hidden_state, dnn_output, attention_output), dim=1)
            out = self.clf1(concatenated_output)
        else:
            out = model_last_hidden_state
        return out