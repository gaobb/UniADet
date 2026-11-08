
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# 比较 WinCLIP 文本提示 prompt 向量和 UniADet 学习获得向量相似度
# 加载 WinCLIP 的文本特征
checkpoint = torch.load(f'./figures/consine/winclip_textual_learned_embeddings.pth', map_location='cpu')
static_text_features = checkpoint['static_text_features'].cpu()  # [2, D]
learned_text_features = checkpoint['learned_text_features'].cpu()  # [2, D]

# 加载 UniADet 模型
net2 = torch.load(f'./figures/consine/uniadet_mvtec_ViT-L-14@336px.pth', map_location='cpu')


adapter_state = net2['adapter'] if 'adapter' in net2 else net2

# 计算 static_text_features 两个向量的差向量并归一化
# static_text_features[1] - static_text_features[0] 表示 异常 - 正常 的方向
static_diff = static_text_features[1] - static_text_features[0]  # [D]
static_diff_norm = F.normalize(static_diff, dim=-1)

# 网络层数映射
layer_names = [12, 15, 18, 21, 24]
class_similarities = []
patch_similarities = []

# 遍历 adapter 中所有 class_prompts 并与差向量计算相似度
print("=" * 60)
print("Comparing difference vectors (abnormal - normal)")
print("=" * 60)

for key, value in adapter_state.items():

    #import ipdb; ipdb.set_trace()   
    if 'class_prompts' in key:
        # value 是 embedding 的权重 [2, embed_dim]
        class_prompt = value
        
        # 计算 class_prompt 的差向量 [1] - [0]
        class_diff = class_prompt[1] - class_prompt[0]  # [D]
        class_diff_norm = F.normalize(class_diff, dim=-1)
        
        # 计算差向量之间的相似度（余弦相似度）
        sim_with_static = (class_diff_norm @ static_diff_norm).item()
        class_similarities.append(sim_with_static)
        
        # 获取对应的 patch_prompts (替换 class_prompts 为 patch_prompts)
        patch_key = key.replace('class_prompts', 'patch_prompts')
        if patch_key in adapter_state:
            patch_prompt = adapter_state[patch_key]
            patch_diff = patch_prompt[1] - patch_prompt[0]  # [D]
            patch_diff_norm = F.normalize(patch_diff, dim=-1)
            
            sim_patch_with_static = (patch_diff_norm @ static_diff_norm).item()
            patch_similarities.append(sim_patch_with_static)
        else:
            patch_similarities.append(0)
        
        print(f"\n{key}:")
        print(f"  Shape: {class_prompt.shape}")
        print(f"  class_diff Similarity with static_diff: {sim_with_static:.4f}")
        print(f"  patch_diff Similarity with static_diff: {sim_patch_with_static:.4f}")

# 可视化 - 柱状图 (高质量论文风格)
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['axes.linewidth'] = 1.2

fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

x = range(len(layer_names))
width = 0.35

# 使用 Nature/Science 风格配色
color_cls = '#e41a1c'
color_seg = '#377eb8'

bars1 = ax.bar([i - width/2 for i in x], class_similarities, width, 
               label=r'$\cos(\Delta \boldsymbol{w}^l_{cls}, \Delta \boldsymbol{w}_{text})$', 
               color=color_cls, edgecolor='black', linewidth=0.8)
bars2 = ax.bar([i + width/2 for i in x], patch_similarities, width, 
               label=r'$\cos(\Delta \boldsymbol{w}^l_{seg}, \Delta \boldsymbol{w}_{text})$', 
               color=color_seg, edgecolor='black', linewidth=0.8)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Blocks', fontsize=18, fontweight='medium')
ax.set_ylabel(r'Cosine Similarity', fontsize=14, fontweight='medium')
ax.set_xticks(x)
ax.set_xticklabels(layer_names, fontsize=16)
ax.tick_params(axis='y', labelsize=14)

# 设置 y 轴范围
ax.set_ylim(-0.25, 0.45)

# 网格线
ax.yaxis.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
ax.set_axisbelow(True)

# 在柱状图上标注数值
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', 
                fontsize=12, fontweight='medium')
for bar in bars2:
    height = bar.get_height()
    offset = -14 if height < 0 else 4
    va = 'top' if height < 0 else 'bottom'
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, offset), textcoords="offset points", ha='center', va=va, 
                fontsize=12, fontweight='medium')

# 图例
ax.legend(fontsize=12, loc='upper left', frameon=False)

# 移除顶部和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# 保存图片
save_path_png = './figures/consine/prompt_similarity_visualization_1102.png'
save_path_eps = './figures/consine/prompt_similarity_visualization_1102.eps'
plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
plt.savefig(save_path_eps, format='eps', bbox_inches='tight')
print(f"\nFigure saved to: {save_path_png}")
print(f"Figure saved to: {save_path_eps}")
plt.show()