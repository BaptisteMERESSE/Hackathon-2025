from typing import Callable
import json

class cleaner:
    def __init__(self, var_name: str, function: Callable):
        self.var_name = var_name
        self.function = function
    
    def clean(self, data: dict):
        data[self.var_name] = self.function(data, self.var_name)



def deleter(data, var_name: str):
    data.pop(var_name, None)




cleaner_list = []

cleaner_list.append(cleaner("nom_variable", deleter))







def clean_metadata(metadata_path, output_path):
    with open(metadata_path, "r", encoding="utf-8") as fin, \
        open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            data = json.loads(line)
            for cleaner in cleaner_list:
                cleaner.clean(data)
            fout.write(json.dumps(data) + "\n")