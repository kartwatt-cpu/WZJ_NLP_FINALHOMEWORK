import os
import pandas as pd
from tqdm import tqdm
import torch

# 从 model_training 导入所有必要函数
from model_training import (
    load_detector, 
    evaluate_model, 
    train_detector, 
    train_and_save_svm, 
    load_svm_model,
    load_and_preprocess_data
)
from ap_generator import APGenerator

def run_full_experiment(detector_path, svm_path, llm_model, test_data):
    """
    完整的对比实验逻辑
    """
    # 1. 加载模型
    print("--- 正在加载模型 ---")
    model_bert, tokenizer = load_detector(detector_path)
    vectorizer, svm_model = load_svm_model(svm_path)
    
    # 2. 提取欺诈样本进行攻击测试
    fraud_texts = [t for t, l in test_data if l == 1]
    if not fraud_texts:
        print("❌ 错误：测试集中没有欺诈样本！")
        return

    # 3. 初始化攻击发生器
    ap_attacker = APGenerator(detector_path, llm_model, max_candidates=3)
    
    bert_successful_attacks = 0
    svm_successful_attacks = 0
    total_samples = len(fraud_texts)
    
    # 4. 开始攻击循环
    print(f"--- 开始对比攻击实验 (共 {total_samples} 条欺诈样本) ---")
    for text in tqdm(fraud_texts, desc="对比攻击中"):
        # 生成改写候选句 (LLM 产生的变体)
        candidates = ap_attacker.generate_paraphrase_candidates(text)
        if not candidates:
            continue
            
        # --- 测 BERT ---
        # 只要生成的候选句中有一个能让 BERT 判定为正常(0)，则 BERT 攻击成功
        _, _, bert_preds = evaluate_model(model_bert, tokenizer, candidates, [1]*len(candidates))
        if 0 in bert_preds:
            bert_successful_attacks += 1
            
        # --- 测 SVM ---
        # 同样的候选句，看 SVM 是否会判定为正常(0)
        vec_candidates = vectorizer.transform(candidates)
        svm_preds = svm_model.predict(vec_candidates)
        if 0 in svm_preds:
            svm_successful_attacks += 1

    # 5. 打印对比实验结果
    print("\n" + "="*40)
    print("      对抗攻击成功率 (ASR) 对比报告")
    print("="*40)
    print(f"实验样本总数: {total_samples}")
    print(f"BERT ASR (大模型): {bert_successful_attacks / total_samples:.4f}")
    print(f"SVM  ASR (传统模型): {svm_successful_attacks / total_samples:.4f}")
    print("-" * 40)
    print("结果分析提示：SVM ASR 通常显著高于 BERT，说明传统模型更依赖字面词频。")
    print("="*40)



# [main_experiment.py]

def run_ablation_study(detector_path, ll_model, fraud_texts):
    print("\n--- 正在运行实验二：消融实验 ---")
    ap_attacker = APGenerator(detector_path, ll_model)
    modes = ['synonym', 'noise', 'full']
    results = {}

    for mode in modes:
        success_count = 0
        for text in tqdm(fraud_texts, desc=f"测试模式: {mode}"):
            # 生成对应模式的候选
            candidates = ap_attacker.generate_ablation_candidates(text, mode=mode)
            # 测 BERT
            _, _, preds = evaluate_model(ap_attacker.detector, ap_attacker.tokenizer, candidates, [1])
            if 0 in preds:
                success_count += 1
        results[mode] = success_count / len(fraud_texts)
    
    print("\n消融实验结果 (ASR):")
    for mode, asr in results.items():
        print(f"模式 {mode}: {asr:.4f}")
    return results




if __name__ == '__main__':
    # 路径配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, '训练集结果.csv')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    DETECTOR_MODEL_PATH = os.path.join(MODEL_DIR, 'bert_fraud_detector.pt')
    SVM_MODEL_PATH = os.path.join(MODEL_DIR, 'svm_model.pkl')
    LLM_MODEL_NAME = "qwen3:8b" 

    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误: 找不到数据文件 {DATA_PATH}")
    else:
        # 1. 预处理并加载数据
        print("--- 正在准备数据 ---")
        train_texts, train_labels, test_texts, test_labels, _ = load_and_preprocess_data(DATA_PATH)
        
        # 2. 检查并训练 BERT
        if not os.path.exists(DETECTOR_MODEL_PATH):
            print("--- 未发现 BERT 模型，开始训练 ---")
            train_detector(DATA_PATH, DETECTOR_MODEL_PATH)
        
        # 3. 检查并训练 SVM
        if not os.path.exists(SVM_MODEL_PATH):
            print("--- 未发现 SVM 模型，开始训练 ---")
            train_and_save_svm(train_texts, train_labels, SVM_MODEL_PATH)

        # 4. 构造用于实验的测试集 (取前 50 条测试集数据)
        real_test_data = list(zip(test_texts, test_labels))[:10]
        
        # 5. 执行对比实验
        run_full_experiment(DETECTOR_MODEL_PATH, SVM_MODEL_PATH, LLM_MODEL_NAME, real_test_data)