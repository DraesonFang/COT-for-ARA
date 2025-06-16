import json

from tqdm import tqdm

import promtConfig as pConf
from promptPreprocess import DataPreparation,CoTPromptGenerator,LLMInterface,EvaluationMetrics,LLMOllamaInterface
import sys
from promtConfig import zeroShot_prompt
import re
import pandas as pd

def extract_level(text):
    if pConf.promptName == pConf.newPrompt or pConf.promptName == pConf.zeroShot_prompt:
        pattern = r"level:\s*(\d+)"
    elif pConf.promptName == pConf.CEFR_prompt:
        pattern = r"(?:CEFR|Level)[:\s]*([ABC][12]|Pre-[ABC][12]|[ABC][12][+-]?)"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1) if match else None


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
        pd_result = pd.DataFrame(columns = ['predictions', 'true_level'])

        #3. check connect
        self.ollamaModel.check_connection()
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
            # prompt = self.prompt_gen.generate_prompt(text,pConf.level)
            prompt = self.prompt_gen.generate_prompt(text)
            if not self.ollamaModel.check_connection():
                print("Failed to connect to Ollama or model not available")
                sys.exit(1)

            print(f"Connected to Ollama with model: {self.ollamaModel.model}")

            result = self.ollamaModel.generate_response(prompt)
            print(result['response'])
            level = extract_level(result['response'])
            new_row = pd.DataFrame([{'predictions': level, 'true_level': true_level}])
            pd_result = pd.concat([pd_result, new_row], ignore_index=True)
            if ind == 2:
                break

        self.save_results(pd_result,pConf.acc_output_path)


    def save_results(self, results, output_path):
        """Save results for analysis"""
        results.to_csv(output_path, index=False)

    def calc_accuracy(self, result_path):
        '''return a percentage,ground_truth is true_level, label of prediction is predictions'''
        df = pd.read_csv(result_path)
        return self.evaluator.calculate_accuracy(df['predictions'],df['true_level'])


if __name__ == '__main__':
    zeroShot_CoT = ZeroShotCoTPipeline(pConf.model,pConf.corpus)
    zeroShot_CoT.run_pipeline()
