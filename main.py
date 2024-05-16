import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import DistilBertForSequenceClassification, DistilBertConfig
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from collections import Counter
import torch
from sklearn.metrics import classification_report
import re
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, f1_score, precision_score, recall_score, classification_report
import numpy as np
import torch.nn as nn
from utils import *
from models import *
from model_T5 import *
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)

def train_eval(use_attention, early_prediction,predict_task, token_type, models_type, batch_size, epochs, pred_days):
    flag = torch.cuda.is_available()
    if flag:
        print("CUDA is available")
    else:
        print("CUDA is not available")

    ngpu = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    torch.cuda.set_device(0)
    print("Device:", device)
    print("GPU Model:", torch.cuda.get_device_name(0))

    # Load your training data
    if early_prediction == 'False':
       
       #================== C =================
        df = pd.read_csv("./final_textual_with_lab_values.csv")
        df[predict_task] = df['過去病史'].astype(str).apply(lambda x: 1 if '糖尿病' in x else 0)
        df = df.drop(columns=['現在病史'], axis=1)
        df = df.rename(columns={'objective_data': '現在病史'})
        import pdb;pdb.set_trace()
        # df = df['現在病史'].tolist()
        # import pdb;pdb.set_trace()
        # df = df.rename(columns={'diabetes': 'label_diabetes'})
        # df = df.rename(columns={'label_multi_x': 'label_multi'})
        #================== C =================

    elif early_prediction == 'True':
        df = lab_to_text_df()
        print(f"Prediction days ======={pred_days}===========")
        # df = lab_to_text_combine(pred_days)
        # df = df.rename(columns={'objective_data': '現在病史'})
        df = df.rename(columns={'objective': '現在病史'})
      
        import pdb;pdb.set_trace()
        df['label_diabetes'] = df['diabetes']

    numerical_col = lab_test_values()

    if predict_task == 'label_diabetes':
        if early_prediction == 'False':
            # df[predict_task] = df['過去病史'].astype(str).apply(lambda x: 1 if '糖尿病' in x else 0)
            label_map = {label: idx for idx, label in enumerate(df[predict_task].unique())}
            df[predict_task] = df[predict_task].map(label_map)
        elif early_prediction == 'True':
            label_map = {label: idx for idx, label in enumerate(df['diabetes'].unique())}
            df['diabetes'] = df['diabetes'].map(label_map)
            

    elif predict_task == 'label_multi':
        df[predict_task] = df['過去病史'].astype(str).apply(label_condition)
        label_map = {label: idx for idx, label in enumerate(df[predict_task].unique())}
        df[predict_task] = df[predict_task].map(label_map)
        
    if early_prediction == 'False':
        scaler = StandardScaler()
        df[numerical_col] = scaler.fit_transform(df[numerical_col])
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        print('train_df labels Counter:', Counter(train_df[predict_task]))
        print('test_df labels Counter:', Counter(test_df[predict_task]))    

    elif early_prediction == 'True':
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        print('train_df labels Counter:', Counter(train_df['diabetes']))
        print('test_df labels Counter:', Counter(test_df['diabetes']))    
    # import pdb;pdb.set_trace()
   
    if token_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif token_type == 'robert':
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    elif token_type == 'clinicalBERT':
        tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    elif token_type == 'BiomedBERT':
        tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" )
    elif token_type == 'SciFive':
        tokenizer = AutoTokenizer.from_pretrained("razent/SciFive-base-Pubmed")  
    elif token_type == 'T5':
        tokenizer = AutoTokenizer.from_pretrained('t5-large')
        
    if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

    if models_type == 'bert':
        model = BertEncoder(num_labels=len(label_map), models_type=models_type, numerical_feature_dim = len(numerical_col))
        model.to(device)

    elif models_type == 'robert':
        model = BertEncoder(num_labels=len(label_map), models_type=models_type, numerical_feature_dim = len(numerical_col))
        model.to(device)

    elif models_type == 'clinicalBERT':
       
        model = BertEncoder(num_labels=len(label_map), models_type=models_type, numerical_feature_dim = len(numerical_col))
        model.to(device)

    elif models_type == 'BiomedBERT':
       
        model = BertEncoder(num_labels=len(label_map), models_type=models_type, numerical_feature_dim = len(numerical_col))
        model.to(device)

    elif models_type == 'SciFive':
        
        model = BertEncoder(num_labels=len(label_map), models_type=models_type, numerical_feature_dim = len(numerical_col))
        model.to(device)

    elif models_type == 'T5':
        model = T5Encoder(num_labels=len(label_map), models_type=models_type, numerical_feature_dim = len(numerical_col))
        model.to(device)
    
    max_length = 128
    learning_rate = 2e-5

    if early_prediction == 'False':
        train_data_loader = create_data_loader(train_df['現在病史'].tolist(), np.array(train_df[numerical_col],dtype=np.float32),
                                        train_df[predict_task].tolist(), tokenizer, max_length, batch_size ,device)
        test_data_loader = create_data_loader(test_df['現在病史'].tolist(),
                                        np.array(test_df[numerical_col],dtype=np.float32), test_df[predict_task].tolist(), tokenizer, max_length, batch_size, device)
    elif early_prediction == 'True':
        train_data_loader = create_data_loader_early(train_df['現在病史'].tolist(),
                                       train_df['diabetes'].tolist(), tokenizer, max_length, batch_size, device)
        test_data_loader = create_data_loader_early(test_df['現在病史'].tolist(), test_df['diabetes'].tolist(), tokenizer, max_length, batch_size, device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = CrossEntropyLoss()

    n_epochs = epochs
    training_range = tqdm(range(n_epochs))

    for epoch in training_range:
        model.train()
        total_loss = 0.0
        for batch in train_data_loader:
            # input_ids, attention_mask, labels = batch
            if early_prediction == 'False':
                input_ids, attention_mask, numerical_features, labels = batch
                numerical_features = numerical_features.to(device)
            elif early_prediction == 'True':
                input_ids, attention_mask, labels = batch
                numerical_features =None
            optimizer.zero_grad()
            # input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            if use_attention == 'True': 
                if models_type == 'T5':
                
                    outputs = model(use_attention, early_prediction,input_ids, attention_mask=attention_mask,decoder_input_ids=input_ids, numerical_features=numerical_features, )
                else:
                
                    outputs = model(use_attention, early_prediction, input_ids, attention_mask=attention_mask, numerical_features=numerical_features)
            else:
                if models_type == 'T5':
               
                    outputs = model(use_attention, early_prediction, input_ids, attention_mask=attention_mask,decoder_input_ids=input_ids, numerical_features=numerical_features, )
                else:
        
                    outputs = model(use_attention, early_prediction, input_ids, attention_mask=attention_mask, numerical_features=numerical_features)

            # logits = outputs.logits
            logits = outputs
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            loss.backward()

            optimizer.step()

        avg_loss = total_loss / len(train_data_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_loss}')

    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_data_loader:

            if early_prediction == 'False':
                input_ids, attention_mask, numerical_features, labels = batch
                numerical_features = numerical_features.to(device)
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            elif early_prediction == 'True':
                input_ids, attention_mask, labels = batch
                numerical_features =None
            # Move data to device
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            if use_attention == 'True':
                
                if models_type == 'T5':
                    outputs = model(use_attention, early_prediction ,input_ids, attention_mask=attention_mask,decoder_input_ids=input_ids, numerical_features=numerical_features, )
                else:
                    outputs = model(use_attention, early_prediction, input_ids, attention_mask=attention_mask, numerical_features=numerical_features)
            else:
                if models_type == 'T5':
                    outputs = model(use_attention, early_prediction, input_ids, attention_mask=attention_mask,decoder_input_ids=input_ids, numerical_features=numerical_features, )
                else:
                    outputs = model(use_attention, early_prediction, input_ids, attention_mask=attention_mask, numerical_features=numerical_features)

            
            preds = torch.argmax(outputs, dim=1).tolist()
            probs = F.softmax(outputs, dim=1)[:, 1].tolist()
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.tolist())



    print('=======Model is:', models_type)
    # calculate AUC
    # auc_value = roc_auc_score(all_labels, all_probs)
    # print(f'AUC: {auc_value}')
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Accuracy: {accuracy}')
    # import pdb;pdb.set_trace()
    # Calculate F1 score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f'F1 Score: {f1}')

    # Calculate precision
    precision = precision_score(all_labels, all_preds, average='weighted')
    print(f'Precision: {precision}')

    # Calculate recall
    recall = recall_score(all_labels, all_preds, average='weighted')
    print(f'Recall: {recall}')

    classification_rep = classification_report(all_labels, all_preds)
    print(f'Classification Report:\n{classification_rep}')

    print('====================================================================')

    if early_prediction == 'True':
        auc_value = roc_auc_score(all_labels, all_probs)
        print(f'AUC: {auc_value}')

        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        auprc_value = auc(recall, precision)
        print(f'AUPRC: {auprc_value}')
        # import pdb;pdb.set_trace()
        # plot_roc_curves(all_labels, all_probs, pred_days, 'roc_curves.png')
        
    else:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_attention', type=str)
    parser.add_argument('--early_prediction', type=str)
    parser.add_argument('--predict_task', type=str)
    parser.add_argument('--token_type', type=str)
    parser.add_argument('--models_type', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)   
    parser.add_argument('--pred_days')   
    args = parser.parse_args()
    train_eval(args.use_attention, args.early_prediction, args.predict_task, args.token_type, args.models_type, args.batch_size, args.epochs, args.pred_days)


if __name__ == '__main__':
    main()
