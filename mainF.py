import os
import json
import torch
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import deque
from datasets import Dataset, load_from_disk, concatenate_datasets
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

# 配置参数
class Config:
    # 模型配置
    BASE_MODEL_PATH = "./model_dir"
    MODEL_A_PATH = "./models/model_a"
    MODEL_B_PATH = "./models/model_b"
    MODEL_C_PATH = "./models/model_c"
    MODEL_D_PATH = "./models/model_d"
    
    # 数据配置
    DATA_DIR = "./data"
    CHECKPOINT_DIR = "./checkpoints"
    MAX_HISTORY = 5  # 保存的最大对话轮次
    MIN_TURNS_FOR_TRAINING = 3  # 开始训练所需的最小对话轮次
    
    # 训练参数
    BATCH_SIZE = 2
    LEARNING_RATE = 2e-5
    EPOCHS = 3
    EARLY_STOPPING_PATIENCE = 3
    REWARD_SCALE = 0.1  # 奖励缩放因子
    PPO_CLIP = 0.2  # PPO clip参数
    KL_PENALTY = 0.01  # KL散度惩罚系数
    
    # 奖励权重
    INTENT_REWARD_WEIGHT = 0.7
    USER_REWARD_WEIGHT = 0.3
    DIVERSITY_BONUS = 0.2  # 多样性奖励
    
    # 混合专家配置
    EXPERT_SWITCH_THRESHOLD = 0.7  # 专家切换阈值
    EXPERT_UPDATE_INTERVAL = 5  # 专家更新间隔(对话次数)
    
    # 初始化目录
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./experts", exist_ok=True)

# 专家系统类
class ExpertSystem:
    def __init__(self, base_model_path: str, num_experts: int = 3):
        self.num_experts = num_experts
        self.experts = []
        self.expert_performance = []
        self.last_update = 0
        
        # 初始化专家模型
        for i in range(num_experts):
            expert_path = f"./experts/expert_{i}"
            if os.path.exists(expert_path):
                model = AutoModelForCausalLM.from_pretrained(expert_path)
            else:
                model = AutoModelForCausalLM.from_pretrained(base_model_path)
                model.save_pretrained(expert_path)
            self.experts.append(model)
            self.expert_performance.append(1.0)  # 初始性能评分
    
    def select_expert(self, prompt: str, tokenizer: AutoTokenizer) -> Tuple[int, AutoModelForCausalLM]:
        # 简单实现: 根据性能加权随机选择
        weights = np.array(self.expert_performance) / sum(self.expert_performance)
        selected_idx = np.random.choice(range(self.num_experts), p=weights)
        return selected_idx, self.experts[selected_idx]
    
    def update_expert_performance(self, expert_idx: int, reward: float):
        # 更新专家性能记录 (指数移动平均)
        self.expert_performance[expert_idx] = (
            0.9 * self.expert_performance[expert_idx] + 0.1 * reward
        )
    
    def maybe_update_experts(self, current_step: int, tokenizer: AutoTokenizer):
        if current_step - self.last_update >= Config.EXPERT_UPDATE_INTERVAL:
            self._update_experts(tokenizer)
            self.last_update = current_step
    
    def _update_experts(self, tokenizer: AutoTokenizer):
        # 用表现最好的专家更新其他专家
        best_expert_idx = np.argmax(self.expert_performance)
        best_expert = self.experts[best_expert_idx]
        
        for i in range(self.num_experts):
            if i != best_expert_idx and self.expert_performance[i] < Config.EXPERT_SWITCH_THRESHOLD:
                # 复制最佳专家到当前专家
                self.experts[i] = AutoModelForCausalLM.from_pretrained(
                    f"./experts/expert_{best_expert_idx}"
                )
                self.experts[i].save_pretrained(f"./experts/expert_{i}")
                self.expert_performance[i] = 0.8  # 重置为中等性能

# 增强的对话系统类
class EnhancedChatSystem(ChatSystem):
    def __init__(self):
        super().__init__()
        
        # 初始化专家系统
        self.expert_system = ExpertSystem(Config.BASE_MODEL_PATH)
        self.conversation_count = 0
        
        # 奖励调整参数
        self.reward_baseline = 0.0
        self.reward_normalization_factor = 1.0
        self.reward_history = deque(maxlen=100)
    
    def generate_responses(self, prompt: str, num_responses: int = 3) -> List[str]:
        # 使用混合专家模式生成响应
        responses = []
        scores = []
        
        for _ in range(num_responses):
            expert_idx, expert_model = self.expert_system.select_expert(prompt, self.tokenizer)
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            outputs = expert_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 评估响应质量
            score = self._evaluate_response_quality(prompt, response)
            
            responses.append(response)
            scores.append(score)
            
            # 记录专家表现
            self.expert_system.update_expert_performance(expert_idx, score)
        
        # 添加多样性奖励
        diversity_bonus = self._calculate_diversity_bonus(responses)
        scores = [s + diversity_bonus for s in scores]
        
        return responses, scores
    
    def _evaluate_response_quality(self, prompt: str, response: str) -> float:
        # 使用模型B评估基本质量
        inputs = self.tokenizer(
            f"Prompt: {prompt}\nResponse: {response}",
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        with torch.no_grad():
            base_score = self.model_b(**inputs).logits.item()
        
        # 添加长度归一化 (避免过长或过短响应)
        optimal_length = 50  # 假设最优长度
        length = len(response.split())
        length_penalty = 1.0 - abs(length - optimal_length) / (optimal_length + 10)
        
        return base_score * length_penalty
    
    def _calculate_diversity_bonus(self, responses: List[str]) -> float:
        # 计算响应之间的多样性
        if len(responses) < 2:
            return 0.0
        
        # 使用简单的词重叠度量
        unique_words = set()
        total_words = 0
        
        for response in responses:
            words = response.split()
            unique_words.update(words)
            total_words += len(words)
        
        diversity_ratio = len(unique_words) / (total_words + 1e-6)
        return Config.DIVERSITY_BONUS * diversity_ratio
    
    def _adjust_reward(self, raw_reward: float) -> float:
        # 动态调整奖励基准
        self.reward_history.append(raw_reward)
        if len(self.reward_history) > 10:  # 有足够历史数据后开始调整
            self.reward_baseline = np.mean(self.reward_history)
            self.reward_normalization_factor = np.std(self.reward_history) + 1e-6
        
        # 标准化奖励
        normalized_reward = (raw_reward - self.reward_baseline) / self.reward_normalization_factor
        return normalized_reward * Config.REWARD_SCALE
    
    def process_turn(self, prompt: str):
        if not self.current_conversation:
            self.start_new_conversation(prompt)
        
        # 生成响应
        generations, scores = self.generate_responses(prompt)
        
        # 调整奖励
        adjusted_scores = [self._adjust_reward(s) for s in scores]
        
        # 排序并选择最佳响应
        scored_responses = sorted(zip(generations, adjusted_scores), key=lambda x: x[1], reverse=True)
        sorted_generations, sorted_scores = zip(*scored_responses)
        
        # 更新当前对话
        self.current_conversation["generations"].append(list(sorted_generations))
        self.current_conversation["scores"].append(list(sorted_scores))
        self.current_conversation["selected_index"] = 0  # 选择得分最高的
        
        # 返回最佳响应
        return sorted_generations[0]
    
    def end_conversation(self, user_rating: float = None):
        super().end_conversation(user_rating)
        self.conversation_count += 1
        
        # 定期更新专家系统
        self.expert_system.maybe_update_experts(self.conversation_count, self.tokenizer)
    
    def train_models(self, dataset: Dataset):
        # 检查是否有足够的数据进行训练
        if len(dataset) < Config.MIN_TURNS_FOR_TRAINING:
            print(f"Not enough data for training (have {len(dataset)}, need {Config.MIN_TURNS_FOR_TRAINING}). Skipping.")
            return
        
        print("Training models with enhanced pipeline...")
        
        # 1. 首先训练奖励模型 (模型B)
        print("Phase 1: Training Reward Model (B)...")
        self._train_with_early_stopping(
            model=self.model_b,
            model_path=Config.MODEL_B_PATH,
            dataset=dataset,
            is_reward_model=True
        )
        
        # 2. 然后训练意图模型 (模型C)
        print("Phase 2: Training Intent Model (C)...")
        self._train_with_early_stopping(
            model=self.model_c,
            model_path=Config.MODEL_C_PATH,
            dataset=dataset,
            is_reward_model=True
        )
        
        # 3. 训练对话评估模型 (模型D)
        print("Phase 3: Training Conversation Model (D)...")
        self._train_with_early_stopping(
            model=self.model_d,
            model_path=Config.MODEL_D_PATH,
            dataset=dataset,
            is_reward_model=True
        )
        
        # 4. 最后用PPO训练生成模型 (模型A)
        print("Phase 4: Training Generation Model (A) with PPO...")
        self._train_generation_model_with_enhanced_ppo(dataset)
    
    def _train_with_early_stopping(self, model: PreTrainedModel, model_path: str, 
                                 dataset: Dataset, is_reward_model: bool = True):
        tokenized_dataset = DataHandler.preprocess_data(dataset, self.tokenizer)
        
        # 拆分训练评估集
        if len(tokenized_dataset) > 10:
            train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
            train_dataset = train_test_split["train"]
            eval_dataset = train_test_split["test"]
        else:
            train_dataset = tokenized_dataset
            eval_dataset = tokenized_dataset
        
        training_args = TrainingArguments(
            output_dir=os.path.join(Config.CHECKPOINT_DIR, os.path.basename(model_path)),
            learning_rate=Config.LEARNING_RATE,
            per_device_train_batch_size=Config.BATCH_SIZE,
            num_train_epochs=Config.EPOCHS,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="loss" if is_reward_model else "accuracy",
            greater_is_better=not is_reward_model
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer
        )
        
        trainer.train()
        model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
    
    def _train_generation_model_with_enhanced_ppo(self, dataset: Dataset):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_a.to(device)
        self.model_b.to(device)  # 奖励模型
        
        # 准备数据加载器
        def collate_fn(batch):
            prompts = [item["prompt"] for item in batch]
            responses = [item["generations"][item["selected_index"]] for item in batch]
            rewards = [item["final_score"] for item in batch]
            
            # Tokenize prompts and responses
            prompt_tokens = [self.tokenizer(p, return_tensors="pt", truncation=True, max_length=256)["input_ids"][0] for p in prompts]
            response_tokens = [self.tokenizer(r, return_tensors="pt", truncation=True, max_length=256)["input_ids"][0] for r in responses]
            
            # Pad sequences
            prompt_tokens = pad_sequence(prompt_tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            response_tokens = pad_sequence(response_tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            
            return {
                "prompt_ids": prompt_tokens.to(device),
                "response_ids": response_tokens.to(device),
                "rewards": torch.tensor(rewards, dtype=torch.float32).to(device)
            }
        
        dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        
        # 优化器和调度器
        optimizer = AdamW(self.model_a.parameters(), lr=Config.LEARNING_RATE)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0, 
            num_training_steps=len(dataloader)*Config.EPOCHS
        )
        
        # 训练循环
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(Config.EPOCHS):
            epoch_loss = 0.0
            self.model_a.train()
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            for batch in progress_bar:
                optimizer.zero_grad()
                
                # 获取旧策略的概率
                with torch.no_grad():
                    old_logits = self.model_a(
                        input_ids=batch["prompt_ids"],
                        attention_mask=(batch["prompt_ids"] != self.tokenizer.pad_token_id).float()
                    ).logits
                
                # 前向传播
                outputs = self.model_a(
                    input_ids=batch["prompt_ids"],
                    attention_mask=(batch["prompt_ids"] != self.tokenizer.pad_token_id).float(),
                    labels=batch["response_ids"]
                )
                
                # 计算新策略的概率
                new_logits = outputs.logits
                new_log_probs = torch.log_softmax(new_logits, dim=-1)
                selected_log_probs = torch.gather(
                    new_log_probs[:, :-1, :], 
                    dim=2, 
                    index=batch["response_ids"][:, 1:].unsqueeze(-1)
                ).squeeze(-1)
                
                # 计算旧策略的概率
                old_log_probs = torch.log_softmax(old_logits[:, :-1, :], dim=-1)
                old_selected_log_probs = torch.gather(
                    old_log_probs, 
                    dim=2, 
                    index=batch["response_ids"][:, 1:].unsqueeze(-1)
                ).squeeze(-1)
                
                # 计算比率和PPO损失
                ratios = torch.exp(selected_log_probs - old_selected_log_probs.detach())
                clipped_ratios = torch.clamp(ratios, 1-Config.PPO_CLIP, 1+Config.PPO_CLIP)
                
                # 奖励加权
                rewards = batch["rewards"].unsqueeze(1).expand(-1, ratios.size(1))
                policy_loss = -torch.min(ratios * rewards, clipped_ratios * rewards).mean()
                
                # KL散度惩罚
                kl_div = (old_log_probs.exp() * (old_log_probs - new_log_probs[:, :-1, :])).sum(-1).mean()
                
                # 总损失
                loss = policy_loss + Config.KL_PENALTY * kl_div
                
                # 反向传播
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item(), "kl_div": kl_div.item()})
            
            # 早停检查
            avg_loss = epoch_loss / len(dataloader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # 保存最佳模型
                self.model_a.save_pretrained(Config.MODEL_A_PATH)
                self.tokenizer.save_pretrained(Config.MODEL_A_PATH)
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

# 增强的主交互循环
def enhanced_main():
    chat_system = EnhancedChatSystem()
    print("Enhanced chat system initialized. Type 'exit' to end the conversation.")
    
    while True:
        try:
            user_input = input("You: ")
            
            if user_input.lower() == 'exit':
                # 结束对话并请求评分
                while True:
                    rating = input("Please rate this conversation (1-10, or 'skip' to skip): ")
                    if rating.lower() == 'skip':
                        chat_system.end_conversation()
                        print("Conversation ended without rating.")
                        break
                    try:
                        rating = float(rating)
                        if 1 <= rating <= 10:
                            chat_system.end_conversation(user_rating=rating)
                            print("Conversation ended. Thank you for your feedback!")
                            break
                        print("Please enter a number between 1 and 10.")
                    except ValueError:
                        print("Invalid input. Please enter a number or 'skip'.")
                break
            
            # 处理用户输入
            if not chat_system.current_conversation:
                print("Starting new conversation...")
                response = chat_system.process_turn(user_input)
                print(f"Assistant: {response}")
            else:
                # 先处理上一轮的反馈
                chat_system.process_feedback(user_input)
                
                # 然后处理新提示
                response = chat_system.process_turn(user_input)
                print(f"Assistant: {response}")
        
        except KeyboardInterrupt:
            print("\nInterrupted. Ending conversation...")
            chat_system.end_conversation()
            break
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print("Resetting conversation...")
            chat_system.end_conversation()

if __name__ == "__main__":
    enhanced_main()
