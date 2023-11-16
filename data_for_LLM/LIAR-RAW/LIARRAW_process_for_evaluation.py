import json
import os
from typing import Optional
from datasets import load_dataset

class Parser:
    def __init__(self, root_dir: str, file_name: str) -> None:
        '''
        load JSON file from local
        '''
        self.root_dir = root_dir
        self.file_name = file_name
        if os.path.exists(self.root_dir):
            file_path = os.path.join(self.root_dir, self.file_name)
            if os.path.exists(file_path):
                self.raw_data = json.load(open(file=file_path, mode="r"))
            else:
                raise FileNotFoundError(f"file {file_path} not exists")
        else:
            raise FileNotFoundError(f"file {root_dir} not exists")
    
    def getRawData(self):
        return self.raw_data
    
    def __len__(self):
        return len(self.raw_data)
    
    def process(self):
        '''
        step1: load event one by one
        step2: extract event_id, claim, label, explain from data
        step3: modify label:
            The mapping function is as follows:
            'True' or 'true' is unified as 'True'.
            'pants-fire', 'hale-true', 'mostly-true', 'barely-true', 'half', 'False' are all unified as 'False'. That means "If the label is not true then it is assigned the value 'False'."
        step4: data collation
        '''
        True_label = ["True", "true", "TRUE"]
        False_label = ['pants-fire', 'half-true', 'mostly-true', 'barely-true', 'half', 'False', 'false']
        self.collated_dataset = []
        for event in self.raw_data:
            claim = event.get("claim", "")
            explain = event.get("explain", "").replace('"', '\\"')
            if event["label"] in True_label:
                label = 'True'
            elif event["label"] in False_label:
                label = 'False'
            else:
                raise ValueError(f"{event['label']} is not valid")

            collated_data = {}
            # evidence and response
            response = f"According to our knowledge and the given information, we think that the claim is {label}."
            collated_data["evidence"] = explain
            collated_data["claim"] = claim            
            collated_data["response"] = response
            self.collated_dataset.append(collated_data)

        return self.collated_dataset
    
    def to_json(self, mode: str):
        mode_choice = ["train", "test", "val"]
        assert mode in mode_choice

        with open("./data_for_LLM/LIAR-RAW/" + f"{mode}_use.jsonl", "w", encoding='utf-8') as file:
            json.dump(self.collated_dataset, file, ensure_ascii=False, indent=4)

if __name__=='__main__':

    test_parser = Parser(root_dir="./data_for_LLM/LIAR-RAW/raw_data/", file_name="test.json")
    raw_test_data = test_parser.getRawData()

    collated_test_data = test_parser.process()

    test_json_file = test_parser.to_json("test")