#!/usr/bin/env python3
import json
import os
import numpy as np
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging
)
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 设置transformers日志级别
logging.set_verbosity_info()

class RLHFSystem:
    def __init__(self):
        # 初始化TensorBoard记录器
        self.writer = SummaryWriter(log_dir="./runs")
        self.current_run = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # 下载或加载基础模型
        if not os.path.exists("./model_dir"):
            print("Downloading base model...")
            from modelscope import snapshot_download
            snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir="./model_dir")
        
        # 设备设置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # 初始化模型
        self.initialize_models()
        
        # 训练数据存储
        self.training_data = deque(maxlen=10)
        self.load_training_data()
        
        # 训练监控数据
        self.monitor_data = {
            "reward_mean": [],
            "reward_std": [],
            "entropy": [],
            "kl_divergence": [],
            "generation_diversity": [],
            "training_loss": [],
            "epsilon": []
        }
        
        # 强化学习参数
        self.initialize_rl_parameters()
        
        # 预训练
        self.pretrain(5)
        print("System initialized successfully!")
    
    def initialize_models(self):
        """初始化所有模型"""
        # 生成模型
        self.generator = AutoModelForCausalLM.from_pretrained("./model_dir").to(self.device)
        self.generator_tokenizer = AutoTokenizer.from_pretrained("./model_dir")
        self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token
        
        # 奖励模型
        self.reward_model_A = AutoModelForSequenceClassification.from_pretrained(
            "./model_dir", num_labels=1).to(self.device)
        self.reward_tokenizer_A = AutoTokenizer.from_pretrained("./model_dir")
        self.reward_tokenizer_A.pad_token = self.reward_tokenizer_A.eos_token
        
        self.reward_model_B = AutoModelForSequenceClassification.from_pretrained(
            "./model_dir", num_labels=1).to(self.device)
        self.reward_tokenizer_B = AutoTokenizer.from_pretrained("./model_dir")
        self.reward_tokenizer_B.pad_token = self.reward_tokenizer_B.eos_token
        
        # 优化器
        self.generator_optimizer = optim.AdamW(self.generator.parameters(), lr=1e-5)
    
    def initialize_rl_parameters(self):
        """初始化RL参数"""
        self.epsilon = 0.9
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.1
        self.ppo_epochs = 3
        self.clip_param = 0.2
        self.entropy_coef = 0.01
        self.kl_coef = 0.1
        self.train_counter = 0
    
    def load_training_data(self):
        """加载已有的训练数据"""
        if os.path.exists("training_data.json"):
            with open("training_data.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                self.training_data = deque(data, maxlen=10)
            print(f"Loaded {len(self.training_data)} training examples")
    
    def save_training_data(self):
        """保存训练数据"""
        with open("training_data.json", "w", encoding="utf-8") as f:
            json.dump(list(self.training_data), f, ensure_ascii=False, indent=2)
    
    def pretrain(self, epochs=5):
        """预训练奖励模型"""
        if len(self.training_data) == 0:
            return
            
        print(f"\nStarting pretraining for {epochs} epochs...")
        
        # 准备数据集
        dataset = Dataset.from_dict({
            "question": [item["question"] for item in self.training_data],
            "answer": [item["answer"] for item in self.training_data],
            "reward": [item["reward"] for item in self.training_data]
        })
        
        # 预处理函数
        def preprocess_function(examples):
            inputs = [f"问题：{q}\n答案：{a}" for q, a in zip(examples["question"], examples["answer"])]
            model_inputs = self.reward_tokenizer_A(inputs, truncation=True, padding=True, max_length=512)
            model_inputs["labels"] = examples["reward"]
            return model_inputs
        
        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        
        # 训练奖励模型A
        training_args = TrainingArguments(
            output_dir="./results_A",
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            num_train_epochs=epochs,
            save_total_limit=2,
            logging_dir='./logs',
            logging_steps=10,
            report_to="tensorboard"
        )
        
        trainer_A = Trainer(
            model=self.reward_model_A,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.reward_tokenizer_A
        )
        trainer_A.train()
        
        # 训练奖励模型B
        trainer_B = Trainer(
            model=self.reward_model_B,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.reward_tokenizer_B
        )
        trainer_B.train()
        
        print("Pretraining completed!")
        self.save_models()
    
    def save_models(self):
        """保存所有模型"""
        self.generator.save_pretrained("./trained_generator")
        self.generator_tokenizer.save_pretrained("./trained_generator")
        
        self.reward_model_A.save_pretrained("./trained_reward_A")
        self.reward_tokenizer_A.save_pretrained("./trained_reward_A")
        
        self.reward_model_B.save_pretrained("./trained_reward_B")
        self.reward_tokenizer_B.save_pretrained("./trained_reward_B")
        print("Models saved to disk")
    
    def generate_answers(self, question, num_answers=3):
        """生成多个答案"""
        answers = []
        inputs = self.generator_tokenizer(f"问题：{question}\n答案：", return_tensors="pt").to(self.device)
        
        for _ in range(num_answers):
            # 使用epsilon-greedy策略
            if np.random.random() < self.epsilon:
                # 探索: 使用随机采样
                outputs = self.generator.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    pad_token_id=self.generator_tokenizer.eos_token_id
                )
            else:
                # 利用: 使用贪婪搜索
                outputs = self.generator.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    pad_token_id=self.generator_tokenizer.eos_token_id
                )
            
            answer = self.generator_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            answers.append(answer)
        
        # 衰减epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return answers
    
    def evaluate_with_reward_A(self, question, answer):
        """使用奖励模型A评估答案质量"""
        inputs = self.reward_tokenizer_A(
            f"问题：{question}\n答案：{answer}",
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.reward_model_A(**inputs)
        
        reward = outputs.logits.item()
        return reward
    
    def evaluate_with_reward_B(self, user_feedback):
        """使用奖励模型B评估用户反馈质量"""
        inputs = self.reward_tokenizer_B(
            user_feedback,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.reward_model_B(**inputs)
        
        reward = outputs.logits.item()
        return reward
    
    def calculate_diversity(self, texts):
        """计算生成文本的多样性"""
        if len(texts) < 2:
            return 0.0
        
        tokens = [set(text.split()) for text in texts]
        similarities = []
        
        for i in range(len(tokens)):
            for j in range(i+1, len(tokens)):
                intersection = len(tokens[i] & tokens[j])
                union = len(tokens[i] | tokens[j])
                similarities.append(intersection / union if union > 0 else 0)
        
        avg_similarity = np.mean(similarities) if similarities else 0
        return 1 - avg_similarity  # 多样性 = 1 - 平均相似度
    
    def calculate_entropy(self, texts):
        """计算生成文本的熵"""
        word_counts = {}
        total_words = 0
        
        for text in texts:
            words = text.split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            total_words += len(words)
        
        if total_words == 0:
            return 0
        
        entropy = 0.0
        for count in word_counts.values():
            p = count / total_words
            entropy -= p * np.log(p + 1e-10)
            
        return entropy / np.log(2)  # 转换为bits
    
    def calculate_kl_divergence(self, questions, answers):
        """计算KL散度（简化版）"""
        if len(self.training_data) == 0:
            return 0.0
        
        train_words = []
        for item in self.training_data:
            train_words.extend(item["answer"].split())
        
        gen_words = []
        for ans in answers:
            gen_words.extend(ans.split())
        
        all_words = set(train_words + gen_words)
        p = {word: train_words.count(word)/len(train_words) for word in all_words}
        q = {word: gen_words.count(word)/len(gen_words) for word in all_words}
        
        kl = 0.0
        for word in all_words:
            p_word = p.get(word, 1e-10)
            q_word = q.get(word, 1e-10)
            kl += p_word * np.log(p_word / q_word)
            
        return kl
    
    def monitor_generation_quality(self, questions, answers, rewards):
        """监控生成质量"""
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)
        entropy = self.calculate_entropy(answers)
        kl_div = self.calculate_kl_divergence(questions, answers)
        diversity = self.calculate_diversity(answers)
        
        # 保存监控数据
        self.monitor_data["reward_mean"].append(reward_mean)
        self.monitor_data["reward_std"].append(reward_std)
        self.monitor_data["entropy"].append(entropy)
        self.monitor_data["kl_divergence"].append(kl_div)
        self.monitor_data["generation_diversity"].append(diversity)
        self.monitor_data["epsilon"].append(self.epsilon)
        
        # 记录到TensorBoard
        self.writer.add_scalar("Quality/Reward_Mean", reward_mean, self.train_counter)
        self.writer.add_scalar("Quality/Reward_Std", reward_std, self.train_counter)
        self.writer.add_scalar("Quality/Entropy", entropy, self.train_counter)
        self.writer.add_scalar("Quality/KL_Divergence", kl_div, self.train_counter)
        self.writer.add_scalar("Quality/Diversity", diversity, self.train_counter)
        self.writer.add_scalar("Params/Epsilon", self.epsilon, self.train_counter)
        
        # 检查模式坍塌
        if diversity < 0.2:  # 多样性阈值
            print("Warning: Potential mode collapse detected! Increasing exploration...")
            self.epsilon = min(0.9, self.epsilon + 0.1)
            self.entropy_coef *= 1.5
    
    def ppo_update(self, questions, answers, rewards):
        """使用PPO算法更新生成模型"""
        old_probs = []
        inputs_list = []
        target_ids_list = []
        
        # 1. 计算旧策略的概率
        with torch.no_grad():
            for q, a in zip(questions, answers):
                inputs = self.generator_tokenizer(f"问题：{q}\n答案：", return_tensors="pt").to(self.device)
                target_ids = self.generator_tokenizer(a, return_tensors="pt").input_ids.to(self.device)
                
                outputs = self.generator(**inputs, labels=target_ids)
                old_probs.append(torch.exp(-outputs.loss))
                inputs_list.append(inputs)
                target_ids_list.append(target_ids)
        
        # 标准化奖励
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        
        # 2. 多轮PPO更新
        for _ in range(self.ppo_epochs):
            policy_losses = []
            value_losses = []
            entropy_losses = []
            kl_penalties = []
            
            for i, (inputs, target_ids, old_p, r) in enumerate(zip(
                inputs_list, target_ids_list, old_probs, rewards)):
                
                # 计算新策略的概率
                outputs = self.generator(**inputs, labels=target_ids)
                new_p = torch.exp(-outputs.loss)
                
                # 计算概率比和surrogate loss
                ratio = new_p / old_p
                surr1 = ratio * r
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * r
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值函数损失
                value_loss = 0.1 * (outputs.logits.mean() - r) ** 2
                
                # 熵奖励
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                entropy_loss = -self.entropy_coef * entropy
                
                # KL惩罚
                kl_penalty = self.kl_coef * torch.log(new_p / old_p) ** 2
                
                # 总损失
                loss = policy_loss + value_loss + entropy_loss + kl_penalty
                
                # 记录各项损失
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                kl_penalties.append(kl_penalty.item())
                
                # 执行优化步骤
                self.generator_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 0.5)
                self.generator_optimizer.step()
            
            # 记录训练损失
            avg_policy_loss = np.mean(policy_losses)
            avg_value_loss = np.mean(value_losses)
            avg_entropy_loss = np.mean(entropy_losses)
            avg_kl_penalty = np.mean(kl_penalties)
            total_loss = avg_policy_loss + avg_value_loss + avg_entropy_loss + avg_kl_penalty
            
            self.monitor_data["training_loss"].append(total_loss)
            self.writer.add_scalar("Training/Policy_Loss", avg_policy_loss, self.train_counter)
            self.writer.add_scalar("Training/Value_Loss", avg_value_loss, self.train_counter)
            self.writer.add_scalar("Training/Entropy_Loss", avg_entropy_loss, self.train_counter)
            self.writer.add_scalar("Training/KL_Penalty", avg_kl_penalty, self.train_counter)
            self.writer.add_scalar("Training/Total_Loss", total_loss, self.train_counter)
        
        # 打印训练信息
        print(f"\nTraining Step {self.train_counter}:")
        print(f"  Avg Reward: {np.mean(rewards.cpu().numpy()):.2f}")
        print(f"  Diversity: {self.monitor_data['generation_diversity'][-1]:.2f}")
        print(f"  Entropy: {self.monitor_data['entropy'][-1]:.2f}")
        print(f"  KL Div: {self.monitor_data['kl_divergence'][-1]:.2f}")
        print(f"  Policy Loss: {avg_policy_loss:.4f}")
        print(f"  Value Loss: {avg_value_loss:.4f}")
        print(f"  Entropy Loss: {avg_entropy_loss:.4f}")
        print(f"  KL Penalty: {avg_kl_penalty:.4f}")
        
        # 保存详细训练日志
        with open(f"./training_log_{self.current_run}.txt", "a") as f:
            f.write(f"\n=== Step {self.train_counter} ===\n")
            f.write(f"Questions: {questions}\n")
            f.write(f"Answers: {answers}\n")
            f.write(f"Rewards: {rewards.cpu().numpy().tolist()}\n")
            for metric, values in self.monitor_data.items():
                if values:
                    f.write(f"{metric}: {values[-1]}\n")
    
    def save_checkpoint(self):
        """保存检查点"""
        checkpoint_dir = f"./checkpoints/checkpoint_{self.train_counter}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存模型
        self.generator.save_pretrained(f"{checkpoint_dir}/generator")
        self.reward_model_A.save_pretrained(f"{checkpoint_dir}/reward_A")
        self.reward_model_B.save_pretrained(f"{checkpoint_dir}/reward_B")
        
        # 保存tokenizer
        self.generator_tokenizer.save_pretrained(f"{checkpoint_dir}/generator")
        self.reward_tokenizer_A.save_pretrained(f"{checkpoint_dir}/reward_A")
        self.reward_tokenizer_B.save_pretrained(f"{checkpoint_dir}/reward_B")
        
        # 保存训练数据
        with open(f"{checkpoint_dir}/training_data.json", "w") as f:
            json.dump(list(self.training_data), f)
        
        # 保存监控数据
        with open(f"{checkpoint_dir}/monitor_data.json", "w") as f:
            json.dump(self.monitor_data, f)
        
        print(f"\nCheckpoint saved at step {self.train_counter}")
    
    def process_user_input(self, question):
        """处理用户输入"""
        # 生成多个答案
        answers = self.generate_answers(question)
        
        # 评估每个答案
        rewards = []
        for answer in answers:
            reward = self.evaluate_with_reward_A(question, answer)
            rewards.append(reward)
        
        # 选择最佳答案
        best_idx = np.argmax(rewards)
        best_answer = answers[best_idx]
        best_reward = rewards[best_idx]
        
        # 监控生成质量
        self.monitor_generation_quality([question]*len(answers), answers, rewards)
        
        return best_answer, best_reward, answers, rewards
    
    def process_user_feedback(self, user_feedback, question, best_answer):
        """处理用户反馈"""
        # 评估用户反馈质量
        feedback_reward = self.evaluate_with_reward_B(user_feedback)
        
        # 获取本轮生成的所有答案和奖励
        _, _, all_answers, all_rewards = self.process_user_input(question)
        
        # 使用PPO更新生成模型
        questions = [question] * len(all_answers)
        self.ppo_update(questions, all_answers, all_rewards)
        
        # 用反馈奖励来训练奖励模型A
        self.training_data.append({
            "question": question,
            "answer": best_answer,
            "reward": feedback_reward
        })
        self.save_training_data()
        
        # 增加训练计数器
        self.train_counter += 1
        
        # 每5步保存检查点
        if self.train_counter % 5 == 0:
            self.save_checkpoint()
        
        return feedback_reward

def main():
    system = RLHFSystem()
    
    print("\n欢迎使用强化学习对话系统! 输入'退出'结束对话。")
    print("系统已初始化，可以开始提问了!\n")
    
    while True:
        # 用户输入问题
        question = input("\n请输入您的问题: ").strip()
        if question.lower() in ["退出", "exit", "quit"]:
            break
        
        # 系统生成并选择最佳答案
        best_answer, best_reward, all_answers, all_rewards = system.process_user_input(question)
        
        print(f"\n最佳答案 (得分: {best_reward:.2f}):")
        print(best_answer)
        
        # 显示其他生成的答案和分数
        print("\n其他生成的答案:")
        for i, (answer, reward) in enumerate(zip(all_answers, all_rewards)):
            if answer != best_answer:
                print(f"选项 {i+1} (得分: {reward:.2f}):")
                print(answer)
                print("---")
        
        # 用户反馈
        feedback = input("\n您对这个回答满意吗? 可以输入您的反馈或直接按回车继续: ").strip()
        if feedback:
            feedback_reward = system.process_user_feedback(feedback, question, best_answer)
            print(f"感谢您的反馈! 反馈得分: {feedback_reward:.2f}")
    
    # 最终保存
    system.save_models()
    system.save_monitoring_plots()
    print("\n对话结束，所有模型和数据已保存。")

if __name__ == "__main__":
    main()
