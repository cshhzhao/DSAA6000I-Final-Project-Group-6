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
            collated_data = {}
            collated_data["id"] = event["event_id"]
            collated_data["claim"] = event["claim"]
            collated_data["explain"] = event["explain"]
            if event["label"] in True_label:
                collated_data["label"] = True
            elif event["label"] in False_label:
                collated_data["label"] = False
            else:
                raise ValueError(f"{event['label']} is not valid")

            self.collated_dataset.append(collated_data)

        return self.collated_dataset
    
    def to_json(self, mode: str):
        mode_choice = ["train", "test", "val"]
        assert mode in mode_choice

        self.processed_json = json.dumps(self.collated_dataset)
        with open("./datasets/Input/LIAR-RAW/" + f"{mode}.json", "w+") as file:
            file.write(self.processed_json)
        return self.processed_json