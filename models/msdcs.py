import torch
import torch.nn as nn
import torch.nn.functional as F


def sliding_window_smooth(patch_feature, window_size=3, stride=1):
    b, n, c = patch_feature.shape
    side = int(n**0.5)
    pad_size = (window_size - stride) // 2

    smooth_patch_feature = patch_feature.reshape(b, side, side, c).permute(0,3,1,2)
    smooth_patch_feature = F.unfold(smooth_patch_feature, kernel_size=window_size, stride=stride, padding=pad_size)

    smooth_patch_feature = smooth_patch_feature.view(b, c, window_size*window_size, -1)
    smooth_patch_feature = smooth_patch_feature.mean(dim=2) #.permute(0,2,1)  # [B, L, C]

    n_bar = smooth_patch_feature.shape[-1]
    if n_bar != n:
        side_bar = int(n_bar**0.5)
        smooth_patch_feature = F.interpolate(smooth_patch_feature.reshape(b, c, side_bar, side_bar), size=(side, side), mode='bilinear') 
        smooth_patch_feature = smooth_patch_feature.reshape(b,c,-1)

    smooth_patch_feature = smooth_patch_feature.permute(0,2,1)  # [B, L, C]

    return smooth_patch_feature

def bs1_compute_anomaly_logits(patch_feature, class_prompts, window_list=[1]):
    class_prompts = class_prompts/class_prompts.norm(dim=-1, keepdim=True)        
    norm_patch_feature = patch_feature/patch_feature.norm(dim = -1, keepdim = True) 

    patch_logits = []
    for window_size in window_list:
        if window_size > 1:
            smooth_patch_feature = sliding_window_smooth(patch_feature[:,1:], window_size=window_size)
            smooth_patch_feature = smooth_patch_feature/smooth_patch_feature.norm(dim = -1, keepdim = True)
            smooth_patch_logit = (smooth_patch_feature @ class_prompts.t()) /0.07
            patch_logits.append(smooth_patch_logit)
        elif window_size == 1:
            norm_patch_logit = (norm_patch_feature[:,1:] @ class_prompts.t()) /0.07
            patch_logits.append(norm_patch_logit)

    patch_logits = torch.stack(patch_logits, dim=0).mean(dim=0) 
    class_logits = (norm_patch_feature[:,0:1] @ class_prompts.t()) /0.07  #CLIP PE
   
    return class_logits, patch_logits

def compute_anomaly_logits(patch_feature, class_prompts, patch_prompts, window_list=[1]):
    class_prompts = class_prompts/class_prompts.norm(dim=-1, keepdim=True)        
    patch_prompts = patch_prompts/patch_prompts.norm(dim=-1, keepdim=True)
    norm_patch_feature = patch_feature/patch_feature.norm(dim = -1, keepdim = True) 

    patch_logits = []
    for window_size in window_list:
        if window_size > 1:
            smooth_patch_feature = sliding_window_smooth(patch_feature[:,1:], window_size=window_size)
            smooth_patch_feature = smooth_patch_feature/smooth_patch_feature.norm(dim = -1, keepdim = True)
            smooth_patch_logit = (smooth_patch_feature @ patch_prompts.t()) /0.07
            patch_logits.append(smooth_patch_logit)
        elif window_size == 1:
            norm_patch_logit = (norm_patch_feature[:,1:] @ patch_prompts.t()) /0.07
            patch_logits.append(norm_patch_logit)

    patch_logits = torch.stack(patch_logits, dim=0).mean(dim=0) 
    class_logits = (norm_patch_feature[:,0:1] @ class_prompts.t()) /0.07  #CLIP PE
   
    return class_logits, patch_logits


def single_scale_nearest_neighbor(query_feat, prompt_feat, img_size, patch_size):
    query_feat = F.normalize(query_feat, dim = -1)  #BLD
    prompt_feat = F.normalize(prompt_feat, dim = -1) #B(LS)D

    b, n, c = query_feat.shape  
    patch_align_score = torch.min(1.0 - torch.bmm(query_feat, prompt_feat.permute(0,2,1)), dim = -1)[0]
    patch_align_score = patch_align_score.reshape(b, 1, img_size//patch_size, img_size//patch_size)
    patch_align_score = F.interpolate(patch_align_score, size=(img_size, img_size), mode='bilinear') #b*1*h*w
        
    return patch_align_score

def multi_scale_nearest_neighbor(query_features, prompt_features, img_size, patch_size, window_list=[1]):
    fs_patch_score = []
    for query_feat, prompt_feat in zip(query_features, prompt_features):
        b, s, n, c = prompt_feat.shape
        
        for window_size in window_list:
            if window_size == 1:
                patch_nn_score = single_scale_nearest_neighbor(query_feat[:, 1:, :], prompt_feat[:,:,1:,:].reshape(b,-1,c), img_size, patch_size)
            else:
                smooth_query_feat = sliding_window_smooth(query_feat[:,1:], window_size=window_size)
                smooth_prompt_feat = sliding_window_smooth(prompt_feat[:,:,1:,:].reshape(b*s, -1, c))
                smooth_prompt_feat = smooth_prompt_feat.reshape(b, -1, c)

                patch_nn_score = single_scale_nearest_neighbor(smooth_query_feat, smooth_prompt_feat, img_size, patch_size)
            fs_patch_score.append(patch_nn_score)
    
    fs_patch_score = torch.cat(fs_patch_score, dim=1).mean(dim=1) 
    return fs_patch_score


class MSDCS(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim=1024, layers_num=4):
        super().__init__()

        self.img_size = int(img_size)
        self.patch_size =  patch_size
        self.embed_dim = embed_dim
        self.layers_num = layers_num
     
       
        self.class_prompts =  nn.ModuleList([nn.Embedding(2, embed_dim) for i in range(layers_num)])
        self.patch_prompts =  nn.ModuleList([nn.Embedding(2, embed_dim) for i in range(layers_num)])

        
        for emb in self.class_prompts:
            nn.init.normal_(emb.weight, std=0.02)
        for emb in self.patch_prompts:
            nn.init.normal_(emb.weight, std=0.02)


    def forward(self, patch_features, window_list=[1]):       
        image_logits, patch_scores = [], []
        for idx, patch_feature in enumerate(patch_features): 
            
            class_prompts = self.class_prompts[idx].weight 
            patch_prompts = self.patch_prompts[idx].weight 
           
            class_logit, patch_logit = compute_anomaly_logits(patch_feature.float(), class_prompts, patch_prompts, window_list=window_list)

            patch_score = patch_logit.softmax(-1)
            side = int(patch_score.shape[1] ** 0.5)
            patch_score = patch_score.reshape(patch_score.shape[0], side, side, -1).permute(0, 3, 1, 2)
            patch_score = F.interpolate(patch_score, self.img_size, mode='bilinear') # 0-1 scores

            patch_scores.append(patch_score)
        
            image_logits.append(class_logit[:,0]) # logits
    
        return image_logits, patch_scores


class BS1MSDCS(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim=1024, layers_num=4):
        super().__init__()

        self.img_size = img_size
        self.patch_size =  patch_size
        self.embed_dim = embed_dim
        self.layers_num = layers_num
     
       
        self.class_prompts =  nn.Embedding(2, embed_dim) 
        nn.init.normal_(self.class_prompts.weight, std=0.02)
    


    def forward(self, patch_features, window_list=[1]):       
        image_logits, patch_scores = [], []
        for idx, patch_feature in enumerate(patch_features): 

            class_prompts = self.class_prompts.weight 
            
            class_logit, patch_logit = bs1_compute_anomaly_logits(patch_feature.float(), class_prompts, window_list=window_list)

            patch_score = patch_logit.softmax(-1)
            side = int(patch_score.shape[1] ** 0.5)
            patch_score = patch_score.reshape(patch_score.shape[0], side, side, -1).permute(0, 3, 1, 2)
            patch_score = F.interpolate(patch_score, self.img_size, mode='bilinear') # 0-1 scores

            patch_scores.append(patch_score)
        
            image_logits.append(class_logit[:,0]) # logits
    
        return image_logits, patch_scores


class BS2MSDCS(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim=1024, layers_num=4):
        super().__init__()

        self.img_size = img_size
        self.patch_size =  patch_size
        self.embed_dim = embed_dim
        self.layers_num = layers_num
     
       
        self.class_prompts =  nn.Embedding(2, embed_dim) 
        self.patch_prompts =  nn.Embedding(2, embed_dim) 


        nn.init.normal_(self.class_prompts.weight, std=0.02)
        nn.init.normal_(self.patch_prompts.weight, std=0.02)
    


    def forward(self, patch_features, window_list=[1]):       
        image_logits, patch_scores = [], []
        for idx, patch_feature in enumerate(patch_features): 

            class_prompts = self.class_prompts.weight
            patch_prompts = self.patch_prompts.weight
            
            class_logit, patch_logit = compute_anomaly_logits(patch_feature.float(), class_prompts, patch_prompts, window_list=window_list)

            patch_score = patch_logit.softmax(-1)
            side = int(patch_score.shape[1] ** 0.5)
            patch_score = patch_score.reshape(patch_score.shape[0], side, side, -1).permute(0, 3, 1, 2)
            patch_score = F.interpolate(patch_score, self.img_size, mode='bilinear') # 0-1 scores

            patch_scores.append(patch_score)
        
            image_logits.append(class_logit[:,0]) # logits
    
        return image_logits, patch_scores


