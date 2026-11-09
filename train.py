import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets import Dataset, get_transform
from models.backbone import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD, build_backbone, setup_seed
from models.msdcs import MSDCS
from utils.logger import get_logger
from utils.loss import FocalLoss, BinaryDiceLoss


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def build_train_loader(args):
    if args.pretrained_model  in ['ViT-L-16-DINOv3', 'ViT-L-14-DINOv2Reg']:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = CLIP_MEAN, CLIP_STD

    train_transform, test_transform, norm_transform = get_transform(args, mean=mean, std=std)

    train_data = Dataset(root=args.train_data_path, img_size = args.image_size, train_transform=train_transform, test_transform=test_transform, norm_transform = norm_transform, dataset_name = args.dataset, save_dir=args.save_path, mode='train', seed=args.seed)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    obj_list = train_data.obj_list
    return train_data_loader

def train(args):
    mode ='train'

    log_file = f'{args.dataset}_{args.seed}seed_{mode}_log.txt'
    logger = get_logger(args.save_path, log_file)

    logger.info('\n')
    logger.info(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, embed_dim, patch_size = build_backbone(args.pretrained_model, device)

    msdcs = MSDCS(args.image_size, patch_size, embed_dim=embed_dim, layers_num=len(args.features_list))
       
    print(msdcs)
    model.to(device)
    msdcs.to(device)
   
    model.eval()
    msdcs.train()

    msdcs_parameters = sum(p.numel() for p in msdcs.parameters())      
    fixed_parameters = sum(p.numel() for p in model.parameters())
    
    print(f"learned params:{(msdcs_parameters):.0f}",
          f"learned parameters:{(msdcs_parameters)/1e+6:.1f}M",
          f"fixed params:{(fixed_parameters)/1e+6:.1f}M",
          f"all params:{(msdcs_parameters+fixed_parameters)/1e+6:.1f}M")

    learnable_params = list(msdcs.parameters())

    optimizer = torch.optim.Adam(
        learnable_params, 
        lr=args.learning_rate, 
        betas=(0.5, 0.999)
        )  

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    ###################### Data  ######################
    train_data_loader = build_train_loader(args)
    
    ###################### forward and backward ######################   
    for epoch in tqdm(range(args.epoch)):
        local_loss_list, global_loss_list = [], []
        for items in tqdm(train_data_loader):
            image = items['query_img'].to(device)
            label =  items['anomaly']
            cls_id = items['cls_id']

            gt = items['query_mask'].squeeze().to(device)
            gt = gt.bool().int()
            
            query_feats = model(image, args.features_list)
            image_logits, patch_scores = msdcs(query_feats, window_list=args.window_list)
            
            global_loss, local_loss = 0, 0
            for gs in image_logits:
                global_loss += F.cross_entropy(gs, label.long().cuda())  
            for ps in patch_scores:
                local_loss += loss_focal(ps, gt)
                local_loss += loss_dice(ps[:, 1, :, :], gt)
                local_loss += loss_dice(ps[:, 0, :, :], 1-gt)

            optimizer.zero_grad()
            (global_loss + local_loss ).backward()
            optimizer.step()
            global_loss_list.append(global_loss.item())
            local_loss_list.append(local_loss.item())
           
        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], global_loss:{:.4f}, local_loss:{:.4f}'.format(
                epoch + 1, args.epoch, np.mean(global_loss_list), np.mean(local_loss_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({"msdcs": msdcs.state_dict()}, ckp_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("UniADet", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint', help='path to save results')
    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")
    parser.add_argument("--pretrained_model", type=str, default='ViT-L-14@336px', help="pre-trained model name")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--window_list", type=int, nargs="+", default=[1], help="features used")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
