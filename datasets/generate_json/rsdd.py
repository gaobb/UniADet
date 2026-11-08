

import os
import json

from paths import dataset_dir
import glob

class KSDD2Solver(object):
    def __init__(self, root='data/mvtec'):
        self.root = root
        self.meta_path = f'{root}/meta.json'
        self.cls_name = 'rsdd'

    def run(self):
        info = dict(train={}, test={})
        anomaly_samples = 0
        normal_samples = 0
        phase = 'test'
        cls_name = self.cls_name
        cls_info = []
        for specie in ['pos', 'neg']:
            cls_dir = f'{self.root}/{specie}'
            
            species = os.listdir(cls_dir)
            
            data_path = os.path.join(self.root, specie)
            data_list = glob.glob(os.path.join(data_path,f"*.jpg"))
            data_list = sorted(data_list)

            is_abnormal = True if specie == 'pos' else False

            for idx, img_name in enumerate(data_list):
                img_path = f'{specie}/{os.path.basename(img_name)}'
                mask_path = img_path.replace('.jpg', '.png') if is_abnormal else ''
                print(img_path)
                info_img = dict(
                    img_path=img_path,
                    mask_path=mask_path,
                    cls_name=self.cls_name,
                    specie_name=specie,
                    anomaly=1 if is_abnormal else 0,
                )
                cls_info.append(info_img)
                
                if is_abnormal:
                    anomaly_samples = anomaly_samples + 1
                else:
                    normal_samples = normal_samples + 1
            info[phase][cls_name] = cls_info

        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)


if __name__ == '__main__':
    runner = KSDD2Solver(root=dataset_dir("RSDD"))
    runner.run()
