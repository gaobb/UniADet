import os
import json

from paths import dataset_dir

# cls: Br35H  BrainMRI HeadCT COVID19
# seg: ClinicDB  ColonDB  Endo  HeadCT  ISIC  Kvasir  TN3K
class MedicalSolver(object):
    CLSNAMES = ['Br35H', 'BrainMRI', 'HeadCT'] #, 'COVID19']

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
            for phase in ['test']:
                cls_info = []
                species = os.listdir(f'{cls_dir}')
                for specie in species:
                    is_abnormal = True if specie not in ['good', 'no'] else False
                    img_names = os.listdir(f'{cls_dir}/{specie}')

                    img_names.sort()

                    for idx, img_name in enumerate(img_names):
                        info_img = dict(
                            img_path=f'{cls_name}/{specie}/{img_name}',
                            mask_path="",
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
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
    runner = MedicalSolver(root=dataset_dir("medical-cls"))
    runner.run()
