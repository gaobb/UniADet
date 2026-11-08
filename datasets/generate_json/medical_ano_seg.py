import os
import json
import pandas as pd

from paths import dataset_dir


class ClinicDBSolver(object):
    CLSNAMES = ['ClinicDB',  'ColonDB', 'Endo',  'ISIC', 'Kvasir'] #,  'TN3K']

    def __init__(self, root='data/mvtec'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        all_anomaly_samples = 0
        all_normal_samples = 0
        for cls_name in self.CLSNAMES:
            class_anomaly_samples = 0
            class_normal_samples = 0
            cls_dir = f'{self.root}/{cls_name}'
            for phase in ['test', 'train']:
                cls_info = []
                if phase == 'test':
                    img_names = os.listdir(f'{cls_dir}/images')
                    mask_names = os.listdir(f'{cls_dir}/masks')
                    is_abnormal = True 
                else:
                    img_names = os.listdir(f'{cls_dir}/normals')
                    mask_names = None
                    is_abnormal = False

                img_names.sort()
                mask_names.sort() if mask_names is not None else None
                for idx, img_name in enumerate(img_names):
                   
                    info_img = dict(
                        img_path=f'{cls_name}/images/{img_name}' if phase == 'test' else f'{cls_name}/normals/{img_name}',
                        mask_path=f'{cls_name}/masks/{mask_names[idx]}' if mask_names is not None else '',
                        cls_name=cls_name,
                        specie_name='',
                        anomaly=1 if is_abnormal else 0
                    )
                    cls_info.append(info_img)
                    if phase == 'test':
                        if is_abnormal:
                            class_anomaly_samples += 1
                        else:
                            class_normal_samples += 1
                info[phase][cls_name] = cls_info
                print(f'{cls_name}_normal_samples', class_normal_samples, f'{cls_name}_anomaly_samples', class_anomaly_samples)
        
            all_anomaly_samples += class_anomaly_samples
            all_normal_samples += class_normal_samples
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        print('all_normal_samples', all_normal_samples, 'all_anomaly_samples', all_anomaly_samples)

if __name__ == '__main__':
    runner = ClinicDBSolver(root=dataset_dir("medical-seg"))
    runner.run()
