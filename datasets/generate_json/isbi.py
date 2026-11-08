import os
import json

from paths import dataset_dir


class IsbiSolver(object):
    CLSNAMES = ['ISIC']

    def __init__(self, root='data/mvtec'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        anomaly_samples = 0
        normal_samples = 0
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}'
            for phase in ['test', 'train']:
                cls_info = []
                if phase == 'test':
                    flag_fold = 'images'
                    is_abnormal = True
                else:
                    flag_fold = 'normals'
                    is_abnormal = False

        
                img_names = os.listdir(f'{cls_dir}/{flag_fold}')
                mask_names = os.listdir(f'{cls_dir}/masks') if is_abnormal else None
                img_names.sort()
                mask_names.sort() if mask_names is not None else None
                for idx, img_name in enumerate(img_names):
                    info_img = dict(
                        img_path=f'{flag_fold}/{img_name}',
                        mask_path=f'masks/{mask_names[idx]}' if is_abnormal else '',
                        cls_name=cls_name,
                        specie_name='ISIC',
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
    runner = IsbiSolver(root=dataset_dir("medical-seg", "ISIC"))
    runner.run()
