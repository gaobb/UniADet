import argparse
from datetime import datetime
from pprint import pformat

import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

from datasets import Dataset, PromptDataset, get_transform
from models.backbone import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD, build_backbone, setup_seed
from models.msdcs import MSDCS, multi_scale_nearest_neighbor
from utils.effecient_metric import Evaluator
from utils.logger import get_logger
from utils.visualization import visualizer

def prompt_association(patch_memory, target_class_name):
    patch_level_num = len(patch_memory[target_class_name[0]])
    retrive_patch = [[] for i in range(patch_level_num)]

    for class_name in target_class_name:
        for l in range(patch_level_num):
            retrive_patch[l].append(patch_memory[class_name][l]) #

    for l in range(patch_level_num):
        retrive_patch[l] = torch.stack(retrive_patch[l]) #B*S*L*D
    return retrive_patch

def build_loaders(args):
    if args.pretrained_model in ('ViT-L-16-DINOv3', 'ViT-L-14-DINOv2Reg'):
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    else:
        mean, std = CLIP_MEAN, CLIP_STD
        
    class_name = args.class_name if args.dataset in ("RealIAD", "Real-IAD-Variety") else None
    train_transform, test_transform, norm_transform = get_transform(args, mean=mean, std=std)
    prompt_data = PromptDataset(
        root=args.test_data_path,
        img_size=args.image_size,
        train_transform=train_transform,
        test_transform=test_transform,
        norm_transform=norm_transform,
        dataset_name=args.dataset,
        k_shots=args.k_shot,
        save_dir=args.save_path,
        mode="test",
        seed=args.seed,
        class_name=class_name,
    )
    test_data = Dataset(
        root=args.test_data_path,
        img_size=args.image_size,
        train_transform=train_transform,
        test_transform=test_transform,
        norm_transform=norm_transform,
        dataset_name=args.dataset,
        save_dir=args.save_path,
        mode="test",
        seed=args.seed,
        class_name=class_name,
    )
    prompt_dataloader = torch.utils.data.DataLoader(prompt_data, batch_size=args.batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return prompt_dataloader, test_dataloader, test_data.obj_list, test_data.view_list

def build_prompt_memory(model, prompt_dataloader, features_list, obj_list, view_list, k_shot, device):
    feats_scale_num = len(features_list)
    prompt_memory = {}
    patch_temp = [[] for i in range(feats_scale_num)]

    cls_names_temp = []
    view_ids_temp = []

    for idx, items in enumerate(tqdm(prompt_dataloader)):
        cls_name = items['cls_name']
        prompt_image = items['prompt_img'].to(device) # B*s*c*h*w
        prompt_mask = items['prompt_mask'].to(device)
        view_id = items['view_id']

        with torch.no_grad():
            query_feats = model(prompt_image, features_list)

        cls_names_temp.extend(cls_name)
        view_ids_temp.extend(view_id)
        for i in range(feats_scale_num):
            patch_temp[i].append(query_feats[i])

    for i in range(feats_scale_num):
        patch_temp[i] = torch.cat(patch_temp[i], dim=0)

    for obj in obj_list:

        if len(view_list) > 1:
            for view_id in view_list:
                indice = (np.array(cls_names_temp) == obj) & (np.array(view_ids_temp) == view_id)
                #prompt_image_memory[obj + '_' + view_id] = image_temp[indice]
                prompt_memory[obj + '_' + view_id] = []

                for i in range(feats_scale_num):
                    if k_shot == "-1":
                        prompt_memory[obj+ '_' + view_id].append(patch_temp[i][indice].cpu())
                    else:
                        prompt_memory[obj+ '_' + view_id].append(patch_temp[i][indice])
        else:
            indice = np.array(cls_names_temp) == obj
            prompt_memory[obj] = []
            for i in range(feats_scale_num):
                if k_shot == "-1":
                    prompt_memory[obj].append(patch_temp[i][indice].cpu())
                else:
                    prompt_memory[obj].append(patch_temp[i][indice])

    return prompt_memory

def inference(model, adapter, prompt_dataloader, test_dataloader, obj_list, patch_size, view_list, cpu_eva, device, args):
    gt_masks, pr_masks, cls_names, gt_anomalys, pr_anomalys, query_paths = [], [], [], [], [], []

    if args.k_shot > 0 or args.k_shot == '-1':
        prompt_memory = build_prompt_memory(
            model, prompt_dataloader, args.features_list, obj_list, view_list, args.k_shot, device,
        )

    nums = 0
    total_time = 0
    for idx, items in enumerate(tqdm(test_dataloader)):
        query_image = items['query_img'].to(device)
        current_batchsize = query_image.shape[0]
        query_path = items['img_path']

        cls_name = items['cls_name']
        cls_id = items['cls_id']

        gt_anomaly = items['anomaly'].to(device)
        gt_mask = items['query_mask']
        gt_mask = gt_mask.bool().float()
        gt_mask = gt_mask.to(device)

        kernel_size = 2 * int(4.0 * args.sigma + 0.5) + 1

        # torch.cuda.synchronize()
        # start_time = time.time()

        with torch.no_grad():
            query_feats = model(query_image, args.features_list)

            # backbone_flops = FlopCountAnalysis(model, (query_image, args.features_list))
            # print(f"Total Backbone FLOPs: {(backbone_flops.total()):.3e}")

            if args.k_shot > 0 or args.k_shot == '-1':
                if len(view_list) > 1:
                    target_cls_name = [cls + '_' + view for cls, view in zip(cls_name, items['view_id'])]
                else:
                    target_cls_name = cls_name
                prompt_feats = prompt_association(prompt_memory, target_cls_name)
                prompt_feats = [feat.to(device) for feat in prompt_feats]

        #################################### adapter
        with torch.no_grad():
            # for zero-shot
            image_logits, patch_scores = adapter(query_feats, window_list=args.window_list)

            # adapter_flops = FlopCountAnalysis(adapter, (query_feats, args.window_list))
            # print(f"Total Adapter FLOPs: {(adapter_flops.total()):.3e}")
            # print(f"Total FLOPs: {(adapter_flops.total() + adapter_flops.total()):.3e}")
          
            zs_patch_score = [ls[:,1:].detach() for ls in patch_scores]
            zs_image_score = [gs.softmax(-1)[:,1:].detach() for gs in image_logits]

            zs_patch_score = torch.concat(zs_patch_score, dim=1).mean(dim=1)
            zs_image_score = torch.concat(zs_image_score, dim=1).mean(dim=1)

            if 'medical' not in args.dataset:
                zs_patch_score = 1.0/(1 + torch.exp(-15*(zs_patch_score-0.40)))

            pixel_anomaly_map = gaussian_blur(
                    zs_patch_score,
                    kernel_size=[kernel_size, kernel_size],
                    sigma=[args.sigma, args.sigma]
                )


        # torch.cuda.synchronize()
        # end_time = time.time()
        # run_time = end_time - start_time

        # if idx > 5:
        #     total_time += run_time
        #     nums += query_image.shape[0]
        # if idx > 200:
        #     break

        # for few-k_shot
        if args.k_shot > 0 or args.k_shot == '-1':
            fs_patch_score = multi_scale_nearest_neighbor(query_feats, prompt_feats, args.image_size, patch_size, window_list=args.window_list)


            fs_patch_score = gaussian_blur(
                fs_patch_score,
                kernel_size=[kernel_size, kernel_size],
                sigma=[args.sigma, args.sigma]
            )
            if 'medical' not in args.dataset:
                fs_patch_score = 1.0/(1 + torch.exp(-15*(fs_patch_score-0.30)))

            pixel_anomaly_map = gaussian_blur(
                (1-args.lamda_fs) * zs_patch_score + args.lamda_fs * fs_patch_score,
                kernel_size=[kernel_size, kernel_size],
                sigma=[args.sigma, args.sigma])


        resize_mask = 256
        if resize_mask is not None:
            pixel_anomaly_map = F.interpolate(pixel_anomaly_map[:,None], size=(resize_mask, resize_mask), mode='bilinear', align_corners=False)
            pixel_anomaly_map = pixel_anomaly_map[:,0]
            gt_mask = F.interpolate(gt_mask[:, None], size=(resize_mask, resize_mask), mode='nearest')
            gt_mask = gt_mask.bool().int()

        # get image level prediction
        if 'medical' in args.dataset:
            topk = int(resize_mask*resize_mask*0.02)
        else:
            topk = 1

        anomaly_map_topk, _ = torch.topk(pixel_anomaly_map.view(current_batchsize, -1), k=topk, dim=1)
        anomaly_map_topk = anomaly_map_topk.mean(dim=1)

        image_anomaly_pred = (1-args.lamda_pixel) * zs_image_score + args.lamda_pixel * anomaly_map_topk


        pixel_anomaly_map  = torch.nan_to_num(pixel_anomaly_map,  nan=0.0, posinf=0.0, neginf=0.0)
        image_anomaly_pred = torch.nan_to_num(image_anomaly_pred, nan=0.0, posinf=0.0, neginf=0.0)

        if cpu_eva:
            gt_masks.append(gt_mask.int().cpu())
            pr_masks.append(pixel_anomaly_map.cpu())
            cls_names.append(np.array(cls_name))
            query_paths.append(np.array(query_path))

            gt_anomalys.append(gt_anomaly.int().cpu())
            pr_anomalys.append(image_anomaly_pred.cpu())
        else:
            gt_masks.append(gt_mask.int())
            pr_masks.append(pixel_anomaly_map)
            cls_names.append(np.array(cls_name))
            query_paths.append(np.array(query_path))

            gt_anomalys.append(gt_anomaly.int())
            pr_anomalys.append(image_anomaly_pred)

        vis_flag = False
        vis_path = './framework-vis'
        if vis_flag:
            if resize_mask is not None:
                visualizer(items['img_path'], pixel_anomaly_map.detach().cpu().numpy(), resize_mask, vis_path, cls_name, args.k_shot)
            else:
                visualizer(items['img_path'], pixel_anomaly_map.detach().cpu().numpy(), args.image_size, vis_path, cls_name, args.k_shot)


    # print("total_times", total_time)
    # print("total_nums", nums)
    # print("average times", total_time/nums)
    # print("fps", nums/total_time)

    return gt_masks, pr_masks, cls_names, gt_anomalys, pr_anomalys, query_paths

def main(args):
    mode ='test'

    log_file = f'{args.dataset}_{args.seed}seed_{mode}_{args.k_shot}shot_log_{datetime.now().strftime("%Y-%m-%d")}.txt'
    logger = get_logger(args.save_path, log_file)

    logger.info('\n' + pformat(vars(args), indent=2, width=80))  
 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ###################### Initialize the evaluation metrics ###################### 
    if args.dataset == 'RealIAD':
        cpu_eva = True
    else:
        cpu_eva = False

    if cpu_eva:
        evaluator = Evaluator('cpu', metrics=args.eval_metrics)
    else:
        evaluator = Evaluator(device, metrics=args.eval_metrics)

    ###################### Build Model and Loaders ###################### 
    model, embed_dim, patch_size = build_backbone(args.pretrained_model, device)
    prompt_dataloader, test_dataloader, obj_list, view_list = build_loaders(args)
  
    ###################### Loading Learners ###################### 
    adapter = MSDCS(args.image_size, patch_size, embed_dim=embed_dim, layers_num=len(args.features_list))
    #adapter = BS1MSDCS(args.image_size, patch_size, embed_dim=embed_dim, layers_num=len(args.features_list))
   
    logger.info('\n' + "loading model from: " + args.checkpoint_path)
    checkpoint_adapter = torch.load(args.checkpoint_path)
    adapter.load_state_dict(checkpoint_adapter["adapter"])
    if hasattr(model, "transformer") and args.pretrained_model == 'ViT-L/14@336px':
       del model.transformer 

    model.to(device)
    adapter.to(device)

    model.eval()
    adapter.eval()

    adapter_parameters = sum(p.numel() for p in adapter.parameters()) 
    fixed_parameters = sum(p.numel() for p in model.parameters())
    

    print(f"adapter params:{(adapter_parameters):.0f}",
          f"learned all parameters:{(adapter_parameters)/1e+6:.3f}M",
          f"fixed params:{(fixed_parameters)/1e+6:.1f}M",
          f"all params:{(adapter_parameters+fixed_parameters)/1e+6:.1f}M"
     )

    ###################### Visual and Learner forward ###################### 
    gt_masks, pr_masks, cls_names, gt_anomalys, pr_anomalys, query_paths = inference(
        model, adapter, prompt_dataloader, test_dataloader, obj_list,
        patch_size, view_list, cpu_eva, device, args,
    )

    ###################### Evaluation ###################### 
    results_eval = dict(gt_masks=gt_masks, pr_masks=pr_masks, cls_names=cls_names, gt_anomalys=gt_anomalys, pr_anomalys=pr_anomalys, query_paths=query_paths)
    results_eval = {k: np.concatenate(v, axis=0) if k == 'cls_names' or k == 'query_paths' else torch.cat(v, dim=0) for k, v in results_eval.items()}

    # save results
    msg = {}
    for idx, cls_name in enumerate(tqdm(obj_list)): 
        metric_results = evaluator.run(results_eval, cls_name, logger)
        msg['Name'] = msg.get('Name', [])
        msg['Name'].append(cls_name)
        avg_act = True if len(obj_list) > 1 and idx == len(obj_list) - 1 else False
        msg['Name'].append('Avg') if avg_act else None
    
        for metric in args.eval_metrics:
            metric_result = metric_results[metric] * 100

            msg[metric] = msg.get(metric, [])
            msg[metric].append(metric_result)
            
            if avg_act:
                metric_result_avg = sum(msg[metric]) / len(msg[metric])
                msg[metric].append(metric_result_avg)
                
    tab = tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.1f', numalign="center", stralign="center", ) 
    logger.info('\n' + tab)  


if __name__ == '__main__':
    parser = argparse.ArgumentParser("UniADet", add_help=True)
    # paths
    parser.add_argument("--test_data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--pretrained_model", type=str, default='ViT-L/14@336px', help="pre-trained model name")
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/', help='path to checkpoint')
    # model
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[12, 15, 18, 21, 24], help="features used")
    parser.add_argument("--window_list", type=int, nargs="+", default=[1], help="features used")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--k_shot", type=int, default=0, help="k_shot")
    parser.add_argument("--lamda_fs", type=float, default=0.5, help="lamda_fs")
    parser.add_argument("--lamda_pixel", type=float, default=0.5, help="lamda_pixel")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="sigma")   
    parser.add_argument("--eval_metrics", type=str, nargs="+", default=['I-AUROC', 'I-AP', 'I-F1max', 'P-AUROC', 'P-AP', 'P-F1max', 'P-AUPRO'], help='evaluation metrics')   
    parser.add_argument("--class_name", type=str, default=None, help="class names for RealIAD dataset")
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    main(args)
