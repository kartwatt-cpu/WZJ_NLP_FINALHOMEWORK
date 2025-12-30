import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# 配置
MODEL_NAME = 'bert-base-chinese'
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =================================================================
# 1. 数据处理和数据集定义
# =================================================================

class FraudDetectionDataset(Dataset):
    """自定义欺诈对话数据集"""
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_preprocess_data(file_path):
    # 保持原有的读取逻辑
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("不支持的文件格式")

    # --- 关键修改：匹配你图片中的实际列名 ---
    # 根据你的截图，文本列是 specific_interaction，标签列是 is_fraud
   # --- 最终修正：匹配你终端打印出的实际列名 ---
    text_col = 'specific_dialogue_content' # 👈 修改这里，从 specific_interaction 改为 specific_dialogue_content
    label_col = 'is_fraud'

    if text_col in df.columns and label_col in df.columns:
        texts = df[text_col].astype(str).tolist()
        # 标签转换逻辑保持兼容
        labels = df[label_col].apply(lambda x: 1 if str(x).upper() in ['TRUE', '1', '是'] else 0).tolist()
    else:
        print(f"表格实际列名为: {df.columns.tolist()}") 
        raise KeyError(f"找不到列: '{text_col}' 或 '{label_col}'")

    # 拆分训练集和测试集 (用于后续攻击)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 进一步拆分测试集中的欺诈样本，作为后续 AP 攻击的原始样本
    fraud_texts_test = [t for t, l in zip(test_texts, test_labels) if l == 1]
    
    return train_texts, train_labels, test_texts, test_labels, fraud_texts_test


# =================================================================
# 2. 目标检测器 D: BERT-base 模型
# =================================================================

class FraudDetectorBERT(nn.Module):
    """基于 BERT 的欺诈对话检测器 (D)"""
    def __init__(self, num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.1)
        # 类别数量: 0: 正常, 1: 欺诈
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 使用 [CLS] token 的输出作为句子表示
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# =================================================================
# 3. 训练与加载函数
# =================================================================

def train_detector(data_path, model_path):
    """训练 BERT-base 模型并保存"""
    print(f"--- 正在加载和预处理数据: {data_path} ---")
    train_texts, train_labels, _, _, _ = load_and_preprocess_data(data_path)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = FraudDetectionDataset(train_texts, train_labels, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = FraudDetectorBERT().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    print(f"--- 开始训练 BERT-base (设备: {DEVICE}) ---")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1} 完成，平均损失: {total_loss / len(train_dataloader):.4f}")

    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"BERT Detector 已保存至 {model_path}")
    return model, tokenizer

def load_detector(model_path):
    """加载 BERT 模型"""
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = FraudDetectorBERT()
    
    # --- 修改开始: 添加 weights_only=False 以消除警告 ---
    # 注意：如果你的 torch 版本非常老，不支持这个参数，请忽略此修改
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
    except TypeError:
        # 兼容旧版本 torch
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    # --- 修改结束 ---
    
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

def evaluate_model(model, tokenizer, texts, labels):
    """评估模型准确率，返回 Acc 和 预测概率 (标签 0: 正常)"""
    model.eval()
    all_preds = []
    all_probs_normal = []
    
    test_dataset = FraudDetectionDataset(texts, labels, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=1) # [batch_size, 2]
            
            predicted_classes = torch.argmax(probs, dim=1).cpu().numpy()
            
            # 提取 '正常' 标签 (标签 0) 的概率，用于对抗性评分
            probs_normal = probs[:, 0].cpu().numpy()

            all_preds.extend(predicted_classes)
            all_probs_normal.extend(probs_normal)

    acc = accuracy_score(labels, all_preds)
    f1 = f1_score(labels, all_preds, average='binary') # 针对二分类 (欺诈/正常)
    
    return acc, all_probs_normal, all_preds


# =================================================================
# 4. 对比基线模型: TextCNN 框架
# =================================================================

class TextCNN(nn.Module):
    """用于对比基线的 TextCNN 模型框架 (此处仅为占位)"""
    def __init__(self, vocab_size, embed_dim, num_class, num_filters=100, filter_sizes=[3, 4, 5]):
        super(TextCNN, self).__init__()
        # 此处需要使用 Embedding 层，而不是 BERT
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=h) 
            for h in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_class)

    def forward(self, x):
        # ... (TextCNN 的前向传播逻辑)
        # (需要用户在实际复现时，结合词向量和 DataLoader 进行实现)
        return torch.randn(x.size(0), 2).to(DEVICE) # 返回随机结果作为占位

if __name__ == '__main__':
    # 示例运行: 训练 BERT 模型并保存
    # ！！！请将 'your_data.csv' 替换为您的实际训练数据文件路径 ！！！
    DATA_FILE = './fraud_dialogue_data.csv' 
    MODEL_SAVE_PATH = './models/bert_fraud_detector.pt'
    
    if not os.path.exists(DATA_FILE):
        print(f"错误: 找不到数据文件 {DATA_FILE}。请替换为您的实际文件路径。")
    else:
        # 第一步：训练并保存 BERT-base 检测器
        train_detector(DATA_FILE, MODEL_SAVE_PATH)







# [model_training.py] 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib

def train_and_save_svm(train_texts, train_labels, save_path):
    """训练一个基于词频特征的 SVM 模型"""
    print("--- 正在训练传统基线模型 (TF-IDF + SVM) ---")
    # 设置 ngram_range=(1,2) 增加它对短语的捕捉能力
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    
    # 启用 probability=True 以便后续观察置信度
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, train_labels)
    
    # 保存向量化器和模型
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump((vectorizer, svm_model), save_path)
    print(f"SVM 模型已成功保存至: {save_path}")
    return vectorizer, svm_model

def load_svm_model(model_path):
    """加载保存的 SVM 组件"""
    return joblib.load(model_path)