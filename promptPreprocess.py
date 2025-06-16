import numpy as np
import pandas as pd
from datasets import load_dataset
import promtConfig as pConf
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import requests
torch.manual_seed(30)

class DataPreparation:
    def __init__(self):
        self.weebit_path = pConf.weebit_path
        self.onestopenglish_path = pConf.onestopenglish_path
        self.UniversalCEFR_path = pConf.UniversalCEFR_path

    def load_corpus(self, corpus_name):
        """Load and standardize corpus format"""
        if corpus_name == "weebit":
            return self.load_weebit()
        elif corpus_name == "onestopenglish":
            return self.load_onestopenglish()
        elif corpus_name == "UniversalCEFR":
            return self.load_UniversalCEFR()

    def load_weebit(self):
        # WeeBit has 5 levels: 7-8, 8-9, 9-10, 10-11, 11-12 age groups
        df = pd.read_csv(self.weebit_path,
                         sep='\t',
                         header=0,  # First row as header
                         names=['text', 'label'],  # Custom column names
                        encoding='utf-8',
                        na_values=['NA', 'null'])  # Define what counts as NaN)
        return df

    def load_UniversalCEFR(self):
        # OneStopEnglish has 3 levels: Elementary, Intermediate, Advanced
        ds = load_dataset(self.UniversalCEFR_path,split="train")
        df = pd.DataFrame(ds)
        df = df.rename(columns={"cefr_level": "label"})
        return df

    def load_onestopenglish(self):
        # OneStopEnglish has 3 levels: Elementary, Intermediate, Advanced
        ds = load_dataset(self.onestopenglish_path,split="train")
        return pd.DataFrame(ds)

    def standardize_levels(self, data, corpus_name):
        """Map different readability scales to unified system"""
        # Map to common scale (e.g., 1-5 or Easy/Medium/Hard)
        pass


class CoTPromptGenerator:
    def __init__(self):
        self.base_prompt_template = pConf.promptName

    def generate_prompt(self, *args):
        return self.base_prompt_template.format(*args)

    def generate_few_shot_prompt(self, text, examples):
        """Optional: Add few examples for better performance"""
        pass


class LLMInterface:
    def __init__(self, model_name):
        self.model_name = model_name
        self.api_endpoint = self.get_api_endpoint(model_name)
        self.tokenizer = None
        self.model = None

    def get_api_endpoint(self, model_name):
        endpoints = {
            "deepseek": "deepseek-ai/DeepSeek-Prover-V2-7B",
            "qwen": "huggingface_api_url",
            "llama": "huggingface_api_url"
        }
        return endpoints.get(model_name)

    def setModel(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.api_endpoint)
        self.model = AutoModelForCausalLM.from_pretrained(self.api_endpoint, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

    def query_llm(self, prompt):
        """
        Query LLM with CoT prompt
        Low temperature for consistent reasoning
        """
        chat = [
            {"role": "system", "content": "You are an expert in text readability assessment."},
            {"role": "user", "content": prompt}
        ]
        prompts = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True
        )
        return self.tokenizer(prompts, return_tensors="pt").to(self.model.device)

    def output_llm(self, prompt):
        start = time.time()
        inputs = self.query_llm(prompt)
        outputs = self.model.generate(inputs['input_ids'], max_new_tokens=8192)
        print(f'runing time: {time.time() - start}')
        return self.tokenizer.batch_decode(outputs)

    def parse_response(self, response):
        """Extract reasoning steps and final readability level"""
        reasoning_steps = self.extract_reasoning(response)
        readability_level = self.extract_level(response)
        return {
            "reasoning": reasoning_steps,
            "level": readability_level
        }

class LLMOllamaInterface:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        """
        Initialize Ollama CoT-ARA client

        Args:
            base_url: Ollama server URL (default: http://localhost:11434)
            model: Model name to use (e.g., 'llama3', 'qwen', 'deepseek-r1')
        """
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()

    def check_connection(self) -> bool:
        """Check if Ollama server is running and model is available"""
        try:
            # Check server status
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                print("connect faild")
                return False

            # Check if model is available
            models = response.json().get('models', [])
            available_models = [m['name'] for m in models]

            if not any(self.model in name for name in available_models):
                print(f"Model '{self.model}' not found. Available models: {available_models}")
                return False

            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def generate_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> dict:
        """
        Send prompt to Ollama and get response

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary containing response and metadata
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/api/generate", json=payload)
            end_time = time.time()

            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "response": result.get("response", ""),
                    "model": result.get("model", self.model),
                    "total_duration": result.get("total_duration", 0) / 1e9,  # Convert to seconds
                    "load_duration": result.get("load_duration", 0) / 1e9,
                    "prompt_eval_count": result.get("prompt_eval_count", 0),
                    "eval_count": result.get("eval_count", 0),
                    "response_time": end_time - start_time
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

class EvaluationMetrics:
    def __init__(self):
        self.results = []

    def calculate_accuracy(self, predictions, ground_truth):
        """Calculate prediction accuracy"""
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        return correct / len(predictions)

    def calculate_eqs(self, reasoning_chains):
        """
        Calculate Explanation Quality Score (EQS)
        Based on your formula from the proposal
        """
        completeness_scores = []
        correctness_scores = []
        consistency_scores = []

        for chain in reasoning_chains:
            completeness = self.evaluate_completeness(chain)
            correctness = self.evaluate_correctness(chain)
            consistency = self.evaluate_consistency(chain)

            completeness_scores.append(completeness)
            correctness_scores.append(correctness)
            consistency_scores.append(consistency)

        # Weights as per your proposal
        w1, w2, w3 = 0.33, 0.33, 0.34  # Equal weights initially

        eqs = np.sqrt(
            (np.mean(completeness_scores) * w1) ** 2 +
            (np.mean(correctness_scores) * w2) ** 2 +
            (np.mean(consistency_scores) * w3) ** 2
        )
        return eqs

    def evaluate_completeness(self, reasoning_chain):
        """Check if all 4 factors are addressed"""
        factors = [
            "vocabulary", "sentence structure",
            "cohesion", "background knowledge"
        ]
        addressed = sum(1 for f in factors if f in reasoning_chain.lower())
        return addressed / len(factors)


