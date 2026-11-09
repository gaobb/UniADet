"""t-SNE of tokens within one class.

Compares patch tokens from the first and last entries of features_list,
and the last-layer CLS token against local patch tokens.
"""
import argparse
from datetime import datetime
from pprint import pformat

import matplotlib.pyplot as plt
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
   
    tsne_output = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(outputs)
    
    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['targets'] = targets
    # 2. Create a new column with the descriptive labels using the map
    df['Category'] = df['targets'].map(label_map)
    # 4. Define the order for the legend (ensures consistency)
    hue_order =  class_names 
    plt.rcParams['figure.figsize'] = 20, 20
    ax = sns.scatterplot(
        x='x', y='y',
        hue='Category',
        hue_order=hue_order,     # <-- Enforce the legend order
        palette=custom_palette,  # <-- Use the custom colors
        data=df,
        marker='o',
        legend="full",
        alpha=1.0,
        s=100 
    )

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
        plt.savefig(image_name, format='eps', bbox_inches='tight')
        plt.savefig(image_name.replace('eps', 'png'), bbox_inches='tight')
    else:
        # Default to eps when no path is given.
        plt.savefig('./tsne.eps', format='eps', bbox_inches='tight')

    plt.close()  # Release the figure after saving.
    '''
    if image_name is not None:
        plt.savefig(image_name, bbox_inches='tight')
    else:
        plt.savefig('./tsne.png', bbox_inches='tight')
    '''
    print('done!')

def sample_tokens(patch_tokens, patch_labels, sampled_pos_indices = None, sample_size=5000):
    neg_label_indices = torch.where(patch_labels == 1)[0]
    # Extract all features and labels for class 1
    neg_patch_tokens = patch_tokens[neg_label_indices]
    neg_patch_labels = patch_labels[neg_label_indices]

    # 3. Isolate and Sample Data for Label 0
    # Find all indices where the label is 0
    pos_label_indices = torch.where(patch_labels == 0)[0]
    # Extract all features and labels for class 0
    pos_patch_tokens = patch_tokens[pos_label_indices]
    pos_patch_labels = patch_labels[pos_label_indices]
    
    # Ensure we don't try to sample more than we have
    if sampled_pos_indices is None:
        sample_size = neg_patch_labels.shape[0]
        num_available_size = pos_patch_labels.shape[0]
        actual_sample_size = min(sample_size, num_available_size)

        # Generate random indices FOR THE LABEL 0 SUBSET
        perm = torch.randperm(num_available_size, device=patch_tokens.device)
        sampled_pos_indices = perm[:actual_sample_size]

    # Perform the sampling on the label 0 data
    sampled_pos_patch_labels = pos_patch_labels[sampled_pos_indices]
    sampled_pos_patch_tokens = pos_patch_tokens[sampled_pos_indices]
    
    final_tokens = torch.cat([neg_patch_tokens, sampled_pos_patch_tokens], dim=0)
    final_labels = torch.cat([neg_patch_labels, sampled_pos_patch_labels], dim=0)
    return final_tokens, final_labels, sampled_pos_indices


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
        class_name=args.class_name,
    )
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
  
    ###################### Loading Learners ###################### 
    model.float() 
    model.to(device)
    model.eval()
    
    ###################### Visual and Learner forward ###################### 
    image_feat, patch_feat, patch_feat12 = [], [], []
    image_label, patch_label = [], []
    for items in tqdm(test_dataloader): 
        query_image = items['query_img'].to(device)
        gt_anomaly = items['anomaly'].to(device)
        gt_mask = items['query_mask'] #[:, 0]
        gt_mask = F.interpolate(gt_mask.unsqueeze(1).float(), size=(args.image_size//patch_size, args.image_size//patch_size), mode='bilinear')

        gt_mask = gt_mask.bool().float()
        gt_mask = gt_mask.to(device)
        b = gt_mask.shape[0]

        with torch.no_grad():
            query_feats = model(query_image, args.features_list)

        image_feat.append(query_feats[-1][:, 0])
        patch_feat.append(query_feats[-1][:,1:])
        patch_feat12.append(query_feats[0][:,1:])

        image_label.append(gt_anomaly)
        patch_label.append(gt_mask.reshape(b, -1))
    
    image_feat = torch.cat(image_feat, dim=0)
    patch_feat = torch.cat(patch_feat, dim=0).reshape(-1, embed_dim)
    image_label = torch.cat(image_label, dim=0).long()
    patch_label = torch.cat(patch_label, dim=0).reshape(-1)
    patch_feat12 = torch.cat(patch_feat12, dim=0).reshape(-1, embed_dim)

    early_layer = args.features_list[0]
    late_layer = args.features_list[-1]
    safe_model_name = args.pretrained_model.replace('/', '-')
    class_colors = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3']

    ## early and late patch tokens
    final_feat24, final_label24, sample_indx24 = sample_tokens(patch_feat, patch_label)
    final_feat12, final_label12, sample_indx12 = sample_tokens(patch_feat12, patch_label, sampled_pos_indices = sample_indx24)

    final_feat  = torch.cat([final_feat24, final_feat12], dim=0).cpu().numpy()
    final_label = torch.cat([final_label24.long(), final_label12.long()+2], dim=0).cpu().numpy()
    class_labels = [f'{late_layer}th Normal', f'{late_layer}th Anomaly', f'{early_layer}th Normal', f'{early_layer}th Anomaly']
    label_map = {i: class_labels[i] for i in range(len(class_labels))}
    custom_palette = {class_label: class_color for class_label, class_color in zip(class_labels, class_colors)}
    print('tsne visualization start...')
    image_name = f'./figures/tsne-token/{safe_model_name}-{args.class_name}-{early_layer}-{late_layer}.eps'
    tsne_plot(final_label, final_feat, class_labels, label_map, custom_palette, image_name=image_name)

    ## global CLS and local patches from the last layer, same normal-patch sample
    final_feat24, final_label24, sample_indx24 = sample_tokens(patch_feat, patch_label, sampled_pos_indices=sample_indx24)

    final_feat  = torch.cat([image_feat, final_feat24], dim=0).cpu().numpy()
    final_label = torch.cat([image_label, final_label24.long()+2], dim=0).cpu().numpy()
    class_labels = ['Normal Images', 'Anomaly Images', 'Normal Patches', 'Anomaly Patches']
    label_map = {i: class_labels[i] for i in range(len(class_labels))}
    custom_palette = {class_label: class_color for class_label, class_color in zip(class_labels, class_colors)}
    print('tsne visualization start...')
    image_name = f'./figures/tsne-token/{safe_model_name}-{args.class_name}-global-local.eps'
    tsne_plot(final_label, final_feat, class_labels, label_map, custom_palette, image_name=image_name)
            
        
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser("UniADet", add_help=True)
    # paths
    parser.add_argument("--test_data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default="./results/", help="path to save logs")
    parser.add_argument("--pretrained_model", type=str, default="ViT-L-14@336px", help="pre-trained model name")
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--features_list", type=int, nargs="+", default=[12, 15, 18, 21, 24], help="features used")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    parser.add_argument("--class_name", type=str, default=None, help="class name to visualize")
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    main(args)