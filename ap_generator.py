import torch
import numpy as np
from model_training import load_detector, evaluate_model, FraudDetectorBERT
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import ollama

# =================================================================
# 1. 评分函数与 LLM 交互
# =================================================================

class APGenerator:
    def __init__(self, detector_path, llm_model_name="qwen3:8b", max_candidates=5):
        # 加载目标检测器 D
        self.detector, self.tokenizer = load_detector(detector_path)
        self.llm_model_name = llm_model_name
        self.max_candidates = max_candidates
        # 加载语义相似度模型 (用于确保语义保持性)
        self.sim_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2') # 多语言模型，适用于中文
        if torch.cuda.is_available():
            self.sim_model = self.sim_model.to('cuda')
        

    
    def generate_paraphrase_candidates(self, text):
        # 1. 预处理：去掉可能存在的对话标签 (left:/right:) 以便 LLM 专注内容
        clean_text = text.replace("left:", "").replace("right:", "").replace("specific_dialogue_content", "").strip()
        
        # 2. 攻击性更强的 Prompt
        prompt = (
            f"你是一个绕过AI审查的黑话专家。请通过以下手段改写这句话，目标是让AI看不出这是诈骗，但人能懂：\n"
            f"1. 【关键词毁灭】：绝对禁止出现'银行'、'钱'、'转账'、'公安'。必须替换为'银.杏'、'米'、'搬砖'、'叔叔'。\n"
            f"2. 【拼音化】：重要词汇用拼音缩写，如 'yinhang'、'zhuanzhang'。\n"
            f"3. 【噪声注入】：每隔两个字随机插入一个特殊符号如 # 或 ~。\n"
            f"原句：{clean_text}\n"
            f"直接输出3个改写版本，不要解释。"
        )
        
        candidates = []
        try:
            # 使用 generate 接口
            response = ollama.generate(model=self.llm_model_name, prompt=prompt)
            raw_output = response['response'].strip()
            
            # 按行分割
            lines = raw_output.split('\n')
            for line in lines:
                line = line.strip()
                # 过滤掉过短的或显然是编号的行
                if len(line) > 2 and not line[0].isdigit():
                    # 如果原句有标签，这里稍微加一点随机性把它拼回去，或者直接忽略标签
                    # 为了骗过BERT，我们尽量破坏原有结构
                    candidates.append(line)
                    
        except Exception as e:
            print(f"Ollama 调用失败: {e}")
        
        # 兜底：如果生成失败，手动加噪声
        if not candidates:
            candidates.append(clean_text.replace("", " ").strip()) # 加空格
            candidates.append(clean_text.replace("，", "#").replace("。", "*")) # 换标点
            
        return candidates[:3] # 限制返回数量，节省后续推理时间
        
        candidates = []
        for _ in range(self.max_candidates):
            try:
                response = ollama.generate(
                    model=self.llm_model_name,
                    prompt=f"{system_prompt}\n\n原始对话: {text}",
                    # 调整参数以增加多样性 (例如 temperature)
                    options={'temperature': 0.8, 'num_predict': 128} 
                )
                candidates.append(response['response'].strip())
            except Exception as e:
                # 捕获 Ollama 错误s
                # print(f"Ollama error: {e}")
                pass
        
        return [c for c in candidates if c] # 过滤空字符串

    def get_semantic_similarity(self, text1, text2):
        """计算语义相似度 (余弦相似度)"""
        embeddings = self.sim_model.encode([text1, text2], convert_to_tensor=True)
        return util.cos_sim(embeddings[0], embeddings[1]).item()

    def attack(self, original_text, original_label, sim_threshold=0.1):
        candidates = self.generate_paraphrase_candidates(original_text)
        if not candidates:
            return None, False

        # 批量预测
        _, all_probs, all_preds = evaluate_model(self.detector, self.tokenizer, candidates, [1]*len(candidates))

        for i, candidate in enumerate(candidates):
            sim_score = self.get_semantic_similarity(original_text, candidate)
            
            # 打印调试信息：看看模型到底给了多少分，相似度是多少
            # detection_score 是被判定为“正常”的概率
            print(f"-> 尝试改写: {candidate[:30]}... | 相似度: {sim_score:.2f} | 正常概率: {all_probs[i]:.4f}")

            # 核心判断：如果正常概率 > 0.5 或者 预测结果为 0
            if all_preds[i] == 0 and sim_score >= sim_threshold:
                return candidate, True
                
        return None, False
    

    # [ap_generator.py] 增加如下方法

    def generate_ablation_candidates(self, text, mode="full"):
        """
        mode: 
        - 'synonym': 仅同义词替换
        - 'noise': 仅插入噪声符号
        - 'full': 完整 AP 攻击
        """
        clean_text = text.replace("left:", "").replace("right:", "").strip()
        
        if mode == "synonym":
            prompt = f"请改写以下句子，仅使用同义词替换敏感词，保持句式和字符结构不变：\n{clean_text}"
        elif mode == "noise":
            prompt = f"请在以下句子的每个字之间随机插入 @、# 或空格，不要修改文字内容：\n{clean_text}"
        else: # full
            prompt = f"请改写以下信息：使用黑话、谐音词替换敏感词，并在字间随机加入干扰符号和空格：\n{clean_text}"

        # 调用 Ollama 生成结果 (参考原有逻辑)
        response = ollama.generate(model=self.llm_model_name, prompt=prompt)
        return [response['response'].strip()]