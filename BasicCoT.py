import json

from tqdm import tqdm

import promtConfig as pConf
from promptPreprocess import DataPreparation,CoTPromptGenerator,LLMInterface,EvaluationMetrics,LLMOllamaInterface
import sys
from promtConfig import zeroShot_prompt


class ZeroShotCoTPipeline:
    def __init__(self, model_name, corpus_name):
        self.data_prep = DataPreparation()
        self.prompt_gen = CoTPromptGenerator()
        self.llm = LLMInterface(model_name)
        self.evaluator = EvaluationMetrics()
        self.corpus_name = corpus_name
        self.ollamaModel = LLMOllamaInterface(model=model_name)

    def run_pipeline(self):
        # 1. Load and prepare data
        print(f"Loading {self.corpus_name} corpus...")
        data = self.data_prep.load_corpus(self.corpus_name)

        # 2. Process each text
        predictions = []
        reasoning_chains = []

        for ind,item in tqdm(data.iterrows()):
            row_dict = item.to_dict()
            text = row_dict['text']
            true_level = row_dict['label']
            '''
            # 3. Generate CoT prompt
            prompt = self.prompt_gen.generate_prompt(text)
            self.llm.setModel()

            # 4. Query LLM
            response = self.llm.output_llm(prompt)
            parsed = self.llm.parse_response(response)
            '''
            prompt = self.prompt_gen.generate_prompt(text,pConf.level)
            if not self.ollamaModel.check_connection():
                print("Failed to connect to Ollama or model not available")
                sys.exit(1)

            print(f"Connected to Ollama with model: {self.ollamaModel.model}")

            result = self.ollamaModel.generate_response(prompt)
            print(result['response'])




    def save_results(self, results, output_path):
        """Save results for analysis"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == '__main__':
    zeroShot_CoT = ZeroShotCoTPipeline(pConf.model,pConf.corpus)
    zeroShot_CoT.run_pipeline()