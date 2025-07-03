import json
import os

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
    elif pConf.promptName == pConf.CEFR_prompt or pConf.promptName == pConf.CEFR_prompt_2:
        pattern = r"(?:CEFR|Level)[:\s]*([ABC][12]|Pre-[ABC][12]|[ABC][12][+-]?)"
    else:
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
        pd_result = pd.DataFrame(columns = ['Essay ID','Text Essay','Gold Label', 'LLM Output','Pred Label'])

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

            result = self.ollamaModel.generate_response(prompt,temperature=pConf.temperature,top_p=pConf.top_p)
            print(result['response'])
            level = extract_level(result['response'])
            new_row = pd.DataFrame([{'Essay ID':ind, 'Text Essay':text, 'Gold Label': true_level,'LLM Output': result['response'],'Pred Label': level}])
            pd_result = pd.concat([pd_result, new_row], ignore_index=True)
            if ind == 5:
                break

        self.calc_accuracy(pd_result, pConf.classification_report_path)
        self.save_results(pd_result, pConf.acc_output_path)


    def save_results(self, results, output_path):
        """Save results for analysis"""
        results.to_csv(output_path, index=False)

    def calc_accuracy(self, dataframe,output_path):
        '''return a percentage,ground_truth is true_level, label of prediction is predictions'''
        target_names = ['A1','A2','B1','B2','C1','C2']
        report_dict = self.evaluator.calculate_accuracy(dataframe['Pred Label'], dataframe['Gold Label'],target_names)
        df_report = pd.DataFrame(report_dict).transpose()
        file_name = output_path

        for i in range(100):
            if os.path.exists(file_name):
                file_name = "classification_report"+str(i) +".csv"

        df_report.to_csv("classification_report.csv",index=False)



if __name__ == '__main__':
    zeroShot_CoT = ZeroShotCoTPipeline(pConf.model,pConf.corpus)
    zeroShot_CoT.run_pipeline()
    print(zeroShot_CoT.calc_accuracy(pConf.acc_output_path))
