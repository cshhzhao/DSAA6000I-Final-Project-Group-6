import json
import os
from typing import Optional


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
            # prompt and chosen
            prompt = (f'Below is an instruction that describes a fake news detection task. '
                      f'Write a response that appropriately completes the request.\n\n'
                      f'### Instruction:\n'
                      f'If there are only True and False categories, based on your knowledge and the '
                      f'following information: {explain} Evaluate the following assertion: {claim} If possible, '
                      f'please also give the reasons. \n\n### Response:.')
            chosen = f"According to our knowledge and the given information, we think that the claim is {label}."
            collated_data["prompt"] = prompt
            collated_data["chosen"] = chosen
            collated_data["rejected"] = "I don't know."
            self.collated_dataset.append(collated_data)

        return self.collated_dataset
    
    def to_json(self, mode: str):
        mode_choice = ["train", "test", "val"]
        assert mode in mode_choice

        with open("./datasets/Input/LIAR-RAW/" + f"{mode}.json", "w", encoding='utf-8') as file:
            json.dump(self.collated_dataset, file, ensure_ascii=False, indent=4)

