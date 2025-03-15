# main.py
import os
import json
import logging
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import numpy as np
from datetime import datetime
from modelscope import snapshot_download

# 初始化日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Main")

# 全局配置参数
CONFIG = {
    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "local_model_dir": "./my_model_dir",
    "max_new_tokens": 50,
    "max_seq_length": 512,
    "batch_size": 2,
    "training_epochs": 1,
    "rl_rounds": 2,
    "data_dir": "data",
    "sample_file": "samples.json",
    "reward_sample_file": "reward_samples.json",
    "training_log_file": "training_logs.json"
}

# 初始化数据目录
os.makedirs(CONFIG["data_dir"], exist_ok=True)

class TrainingDataset(Dataset):
    def __init__(self, encodings, scores=None):
        self.encodings = encodings
        self.scores = scores

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.scores is not None:
            item['scores'] = torch.tensor(self.scores[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_logs = []
        
    def log(self, logs):
        super().log(logs)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "epoch": self.state.epoch,
            "step": self.state.global_step,
            "metrics": {
                "loss": logs.get("loss", None),
                "learning_rate": logs.get("learning_rate", None)
            }
        }
        self._save_training_log(log_entry)
        logger.info(f"Step {self.state.global_step} | Loss: {log_entry['metrics']['loss']:.4f}")

    def _save_training_log(self, log_entry):
        log_path = os.path.join(CONFIG["data_dir"], CONFIG["training_log_file"])
        try:
            # 读取现有日志
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    existing_logs = json.load(f)
            else:
                existing_logs = []
            
            # 添加新日志
            existing_logs.append(log_entry)
            
            # 保存更新后的日志
            with open(log_path, "w") as f:
                json.dump(existing_logs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save training log: {str(e)}")

class ModelTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_models()
        
    def _init_models(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(CONFIG['local_model_dir'])
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            model_config = {
                "pad_token_id": self.tokenizer.eos_token_id,
                "attention_mask": True
            }
            
            self.policy_model = AutoModelForCausalLM.from_pretrained(
                CONFIG['local_model_dir'], 
                **model_config
            ).to(self.device)
            
            self.reward_model = AutoModelForCausalLM.from_pretrained(
                CONFIG['local_model_dir'],
                **model_config
            ).to(self.device)
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise

    def _load_json_data(self, filename):
        """加载JSON格式数据"""
        filepath = os.path.join(CONFIG["data_dir"], filename)
        try:
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Failed to load {filename}: {str(e)}")
            return []

    def _prepare_dataset(self, data):
        """准备训练数据集"""
        try:
            texts = [item["text"] for item in data]
            scores = [item["score"] for item in data]
            
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=CONFIG['max_seq_length']
            )
            return TrainingDataset(encodings, scores)
        except Exception as e:
            logger.error(f"Dataset preparation failed: {str(e)}")
            raise

    def train_policy(self):
        """训练策略模型"""
        try:
            training_data = self._load_json_data(CONFIG["sample_file"])
            if not training_data:
                logger.warning("No training data available")
                return
                
            dataset = self._prepare_dataset(training_data)
            logger.info("Starting policy model training...")
            
            training_args = TrainingArguments(
                output_dir='./policy_results',
                num_train_epochs=CONFIG['training_epochs'],
                per_device_train_batch_size=CONFIG['batch_size'],
                logging_dir='./policy_logs',
                report_to="none",
                remove_unused_columns=True,
                logging_steps=1
            )
            
            trainer = CustomTrainer(
                model=self.policy_model,
                args=training_args,
                train_dataset=dataset,
            )
            
            trainer.train()
            self.policy_model.save_pretrained("./policy_model")
            logger.info("Policy model training completed.")
        except Exception as e:
            logger.error(f"Policy training failed: {str(e)}")
            raise

    def train_reward_model(self):
        """训练奖励模型"""
        try:
            reward_data = self._load_json_data(CONFIG["reward_sample_file"])
            if not reward_data:
                logger.warning("No reward data available")
                return
                
            dataset = self._prepare_dataset(reward_data)
            logger.info("Starting reward model training...")
            
            training_args = TrainingArguments(
                output_dir='./reward_results',
                num_train_epochs=CONFIG['training_epochs'],
                per_device_train_batch_size=CONFIG['batch_size'],
                logging_dir='./reward_logs',
                report_to="none",
                remove_unused_columns=True,
                logging_steps=1
            )
            
            trainer = CustomTrainer(
                model=self.reward_model,
                args=training_args,
                train_dataset=dataset,
            )
            
            trainer.train()
            self.reward_model.save_pretrained("./reward_model")
            logger.info("Reward model training completed.")
        except Exception as e:
            logger.error(f"Reward training failed: {str(e)}")
            raise

class RLAgent:
    def __init__(self, trainer):
        self.trainer = trainer
        self.generated_samples = []
    
    def generate_question(self):
        return "Explain machine learning in simple terms."
    
    def _save_generated_data(self):
        """保存生成的训练数据"""
        filepath = os.path.join(CONFIG["data_dir"], CONFIG["sample_file"])
        try:
            existing_data = []
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    existing_data = json.load(f)
            
            new_data = []
            for sample in self.generated_samples:
                for answer in sample["answers"]:
                    new_data.append({
                        "text": answer["text"],
                        "score": answer["score"]
                    })
            
            existing_data.extend(new_data)
            with open(filepath, "w") as f:
                json.dump(existing_data, f, indent=2)
            logger.info(f"Saved {len(new_data)} new samples")
        except Exception as e:
            logger.error(f"Failed to save samples: {str(e)}")

    def _generate_answer(self, question):
        """生成答案"""
        try:
            inputs = self.trainer.tokenizer(
                question, 
                return_tensors="pt", 
                max_length=CONFIG['max_seq_length'],
                truncation=True
            ).to(self.trainer.device)
            
            with torch.no_grad():
                generated = self.trainer.policy_model.generate(
                    **inputs,
                    max_new_tokens=CONFIG['max_new_tokens'],
                    pad_token_id=self.trainer.tokenizer.eos_token_id,
                    attention_mask=inputs['attention_mask']
                )
            return self.trainer.tokenizer.decode(generated[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return ""

    def get_reward(self, text):
        """获取评分"""
        try:
            inputs = self.trainer.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=CONFIG['max_seq_length'],
                padding='max_length'
            ).to(self.trainer.device)
            
            with torch.no_grad():
                outputs = self.trainer.reward_model(**inputs)
            return outputs.logits.mean().item()
        except Exception as e:
            logger.error(f"Reward calculation failed: {str(e)}")
            return 0.0

    def run_rl_cycle(self):
        """执行强化学习循环"""
        try:
            logger.info("Starting RL training...")
            for _ in range(CONFIG['rl_rounds']):
                question = self.generate_question()
                answers = []
                
                for _ in range(2):
                    answer_text = self._generate_answer(question)
                    if not answer_text:
                        continue
                    
                    score = self.get_reward(answer_text)
                    answers.append({
                        "text": answer_text,
                        "score": score
                    })
                    logger.info(f"Generated answer | Score: {score:.2f}")
                
                if answers:
                    self.generated_samples.append({
                        "question": question,
                        "answers": answers
                    })
            
            self._save_generated_data()
            logger.info("RL cycle completed")
        except Exception as e:
            logger.error(f"RL cycle failed: {str(e)}")
            raise

def main():
    try:
        trainer = ModelTrainer()
        rl_agent = RLAgent(trainer)
        
        # 训练奖励模型
        reward_file = os.path.join(CONFIG["data_dir"], CONFIG["reward_sample_file"])
        if os.path.exists(reward_file):
            logger.info("Training reward model...")
            trainer.train_reward_model()
        
        # 训练策略模型
        trainer.train_policy()
        
        # 执行强化学习
        rl_agent.run_rl_cycle()
        
        logger.info("Process completed successfully")
    except Exception as e:
        logger.error(f"Main process failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    try:
        # 模型下载检查
        if not os.path.exists(CONFIG['local_model_dir']):
            os.makedirs(CONFIG['local_model_dir'], exist_ok=True)
            snapshot_download(CONFIG['model_name'], 
                             local_dir=CONFIG['local_model_dir'])
        
        # 初始化CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        exit(1)
