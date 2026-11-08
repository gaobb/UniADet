"""t-SNE of normal and anomalous patches from two datasets.

Layer features are concatenated on the last dimension. Existing .npy
caches are reused. New figures are written under figures/tsne-rerun.
"""
import argparse
import os
from datetime import datetime
from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm
from tsnecuda import TSNE

from datasets import Dataset, get_transform
from models.backbone import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD, build_backbone, setup_seed
from utils.logger import get_logger


def tsne_plot(targets, outputs, class_names, label_map, custom_palette, image_name=None):
    print('generating t-SNE plot...')
 
    tsne_output = TSNE(n_components=2, perplexity=15, learning_rate=200).fit_transform(outputs)
   

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['targets'] = targets
    # 2. Create a new column with the descriptive labels using the map
    df['Category'] = df['targets'].map(label_map)
    
    # Debug: print unique categories in data
    print(f"Unique categories in data: {df['Category'].unique()}")
    print(f"Expected class_names: {class_names}")
    
    # 4. Define the order for the legend - only include categories that exist in data
    existing_categories = df['Category'].dropna().unique().tolist()
    hue_order = [c for c in class_names if c in existing_categories]
    
    # Filter palette to only include existing categories
    filtered_palette = {k: v for k, v in custom_palette.items() if k in existing_categories}
    
    # Create new figure
    fig, ax = plt.subplots(figsize=(20, 20))
    
    ax = sns.scatterplot(
        x='x', y='y',
        hue='Category',
        hue_order=hue_order if hue_order else None,     # <-- Enforce the legend order
        palette=filtered_palette if filtered_palette else None,  # <-- Use the custom colors
        data=df,
        marker='o',
        legend="full",
        alpha=1.0,
        s=200,
        ax=ax
    )

    # Check if legend exists before moving it
    if ax.get_legend() is not None:
        sns.move_legend(
            ax,
            loc="upper center",          # anchor
            ncol=2,                      # two columns
            title=None,                  # drop the legend title
            frameon=False,               # drop the legend frame
            fontsize=32,             #'medium'
            markerscale=3,
        )

    # The single, clean way to remove all borders, ticks, and labels
    ax.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    if image_name is not None:
        png_name = image_name[:-4] + '.png' if image_name.endswith('.eps') else image_name + '.png'
        if os.path.exists(image_name) or os.path.exists(png_name):
            print(f'skip existing figure: {image_name}')
        else:
            out_dir = os.path.dirname(image_name)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            plt.savefig(image_name, format='eps', bbox_inches='tight')
            plt.savefig(png_name, bbox_inches='tight')
    else:
        # Default to eps when no path is given.
        plt.savefig('./tsne.eps', format='eps', bbox_inches='tight')

    plt.close()
    print('done!')


def main(args):
    mode = 'test'
    log_file = f'{args.dataset}_{args.seed}seed_{mode}_log_{datetime.now().strftime("%Y-%m-%d")}.txt'
    logger = get_logger(args.save_path, log_file)

    logger.info('\n' + pformat(vars(args), indent=2, width=80))  
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, embed_dim, patch_size = build_backbone(args.pretrained_model, device)

    if args.pretrained_model in ('ViT-L-16-DINOv3', 'ViT-L-14-DINOv2Reg'):
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = CLIP_MEAN, CLIP_STD

    train_transform, test_transform, norm_transform = get_transform(args, mean=mean, std=std)

    test_data = Dataset(
        root=args.test_data_path,
        img_size=args.image_size,
        train_transform=train_transform,
        test_transform=test_transform,
        norm_transform=norm_transform,
        dataset_name=args.dataset,
        save_dir=args.save_path,
        mode=mode,
        seed=args.seed,
    )
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model.float()
    model.to(device)
    model.eval()

    patch_tokens, patch_label = [], []
    for items in tqdm(test_dataloader):
        query_image = items['query_img'].to(device)
        gt_mask = items['query_mask'] #[:, 0]
        gt_mask = F.interpolate(gt_mask.unsqueeze(1).float(), size=(args.image_size // patch_size, args.image_size // patch_size), mode='bilinear')

        #gt_mask = gt_mask.bool().float()
        gt_mask = gt_mask.to(device)
        b = gt_mask.shape[0]

        with torch.no_grad():
            query_feats = model(query_image, args.features_list)
            
        patch_tokens.append(torch.cat(query_feats, dim=-1)[:,1:,:])
        #patch_tokens.append(query_feats[-1][:,1:,:])
        patch_label.append(gt_mask.reshape(b, -1))

    patch_label = torch.cat(patch_label, dim=0).reshape(-1)
    patch_tokens = torch.cat(patch_tokens, dim=0).reshape(-1, embed_dim * len(query_feats))
  
    
 
    print(patch_tokens.shape, patch_label.shape)
    normal_patch_tokens = patch_tokens[patch_label < 0.5, :]
    abnormal_patch_tokens = patch_tokens[patch_label > 0.5, :]
    print(normal_patch_tokens.shape, abnormal_patch_tokens.shape)
   
    # randomly sample 1000 normal patch tokens
    normal_patch_tokens = normal_patch_tokens[torch.randperm(normal_patch_tokens.shape[0])[:5000], :]
    abnormal_patch_tokens = abnormal_patch_tokens[torch.randperm(abnormal_patch_tokens.shape[0])[:5000], :] 

    final_feat12 = torch.cat([normal_patch_tokens, abnormal_patch_tokens], dim=0) #.cpu().numpy()
    final_label12 = torch.cat([torch.zeros(normal_patch_tokens.shape[0]), torch.ones(abnormal_patch_tokens.shape[0])], dim=0) #.cpu().numpy()

 
    return final_feat12, final_label12 
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser("UniADet", add_help=True)
    # paths
    parser.add_argument("--test_data_path", type=str, default="./data", help="root directory that contains visa, mvtec, and medical-seg")
    parser.add_argument("--save_path", type=str, default="./results/", help="path to save logs")
    parser.add_argument("--pretrained_model", type=str, default="ViT-L/14@336px", help="pre-trained model name")
    parser.add_argument("--dataset_pair", type=str, default="visa-medical", choices=["mvtec-medical", "visa-mvtec", "visa-medical"])
    parser.add_argument("--features_list", type=int, nargs="+", default=[12, 15, 18, 21, 24], help="features used")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)


    
    dataset = args.dataset_pair
    dataset_runs = {
        'mvtec-medical': (
            ['MVTec Normal', 'MVTec Abormal', 'Medical Normal', 'Medical Abormal'],
            ['mvtec', 'medical-seg'],
        ),
        'visa-mvtec': (
            ['VisA Normal', 'VisA Abormal', 'MVTec Normal', 'MVTec Abormal'],
            ['visa', 'mvtec'],
        ),
        'visa-medical': (
            ['VisA Normal', 'VisA Abormal', 'Medical Normal', 'Medical Abormal'],
            ['visa', 'medical-seg'],
        ),
    }
    class_labels, runs = dataset_runs[dataset]
    class_colors = ['#377eb8', '#e41a1c', '#72a9d4', '#fb8072']

    safe_model_name = args.pretrained_model.replace('/', '-')
    image_name = f'./figures/tsne-dataset-pair/{safe_model_name}-{dataset}-normal-anomaly.eps'

    data_feat = f'./figures/tsne-dataset-pair/{safe_model_name}-{dataset}_normal_abnormal_embeddings.npy'
    data_label = f'./figures/tsne-dataset-pair/{safe_model_name}-{dataset}_normal_abnormal_labels.npy'

    label_map = {i: class_labels[i] for i in range(len(class_labels))}
    custom_palette = {class_label: class_color for class_label, class_color in zip(class_labels, class_colors)}

    if os.path.exists(data_feat) and os.path.exists(data_label):
        final_feat = np.load(data_feat)
        final_label = np.load(data_label)
        print(final_feat.shape, final_label.shape)
    else:
        data_root = args.test_data_path
        saved = []
        for name in runs:
            args.dataset = name
            args.test_data_path = os.path.join(data_root, name)
            saved.append(main(args))
        data1_feat, data1_label = saved[0]
        data2_feat, data2_label = saved[1]
        final_feat = torch.cat([data1_feat, data2_feat], dim=0).cpu().numpy()
        final_label = torch.cat([data1_label, data2_label + 2], dim=0).cpu().numpy()
        if not os.path.exists(data_feat):
            np.save(data_feat, final_feat)
        if not os.path.exists(data_label):
            np.save(data_label, final_label)

    png_name = image_name[:-4] + '.png' if image_name.endswith('.eps') else image_name + '.png'
    if os.path.exists(image_name) and os.path.exists(png_name):
        print(f'skip existing figure: {image_name}')
    else:
        print('tsne visualization start...')
        tsne_plot(final_label, final_feat, class_labels, label_map, custom_palette, image_name=image_name)