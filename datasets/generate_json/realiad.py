'''
 # @ Author: Bin-Bin Gao
 # @ Create Time: 2025-07-25 10:19:17
 # @ Modified by: Your name
 # @ Modified time: 2025-07-25 22:11:44
 # @ Description: Detect and Describe Any Vision Anomalies with Large Vision-Language Models
 '''



import os
import glob
import json

from paths import dataset_dir


class IndustryDBSolver(object):
    def __init__(self, root_path=None, data_name="RealIAD", json_path=None):
        if root_path is None:
            root_path = dataset_dir()
        self.data_path = os.path.join(root_path, data_name)
        self.data_name = data_name
        self.meta_path = f'{self.data_path}/meta.json'
        self.json_path = json_path or dataset_dir("realiad_jsons")

        self.json_files = [json_file for json_file in os.listdir(f'{self.json_path}') if ".json" in json_file]

        self.cls_names = sorted([item.split('.json')[0] for item in self.json_files])
        

    def run(self):
          
        info = dict(
            categories = self.cls_names,
            number_normal = {item: 0 for item in self.cls_names},
            number_abnormal = {item: 0 for item in self.cls_names},
            train={}, 
            test={}
            )
       

        for json_file in self.json_files:
            cls_name = json_file.split('.json')[0]
            
            file_path = os.path.join(self.json_path, json_file)

            print(json_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f) 
                
            for phase in ['train', 'test']:
                info[phase][cls_name] = []
                cls_info = []

                for item in data[phase]:
                    view_id =  item["image_path"].split('/')[-1].split('_')[-2]  # Assuming the view is indicated in the filename

                    info_img = dict(
                        img_path = f'{cls_name}/{item["image_path"]}',
                        mask_path= f'{cls_name}/{item["mask_path"]}' if item['mask_path'] else "",
                        annotation_type='mask',
                        view_id = view_id,
                        cls_name=cls_name,
                        specie_name=item["anomaly_class"],
                        anomaly=0 if item['anomaly_class'] == 'OK' else 1,    
                    )
                    is_abnormal = True if item['anomaly_class'] != 'OK' else False
                    
                    if is_abnormal:
                        cls_info.append(info_img)
                        info['number_abnormal'][cls_name] += 1
                    else:
                        cls_info.append(info_img)
                        info['number_normal'][cls_name] += 1
               
                info[phase][cls_name] = cls_info

        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")

if __name__ == '__main__':
    runner = IndustryDBSolver(dataset_dir(), "RealIAD")
    runner.run()