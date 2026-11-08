source "$(dirname "$0")/env.sh"

for train_dataset in  mvtec visa
do
for pretrained_model in  "ViT-L-16-DINOv3" 'ViT-L/14@336px' 'ViT-L-14-DINOv2Reg' #"ViT-L-16-DINOv3""ViT-L-16-DINOv3" #'ViT-L-14-DINOv2Reg' #"ViT-L-16-DINOv3" #"ViT-L-14-DINOv2Reg" #"ViT-L-16-DINOv3" #"ViT-L-16-DINOv3" #"ViT-L/14@336px" #"ViT-L-14-DINOv2Reg" #"PE-Core-G14-448" #"ViT-L/14@336px" #"ViT-L-14-DINOv2Reg"
do

# 根据模型名设置 image_size
case "$pretrained_model" in
"ViT-L/14@336px"|"ViT-L-14-DINOv2Reg")
    image_size=518
    
    ;;
"ViT-L-16-DINOv3")
    image_size=512
    ;;
*)
    # 未明确的模型默认值（可按需要调整）
    image_size=518
    ;;
esac
stage=1

checkpoint_path=${UNIADET_CKPT}/train_on_${train_dataset}_zs_${pretrained_model}_imagesize${image_size}_1015

CUDA_VISIBLE_DEVICES=${UNIADET_DEVICE} python train.py --dataset ${train_dataset} --train_data_path ${UNIADET_DATA}/${train_dataset} \
--save_path ${checkpoint_path} \
--pretrained_model ${pretrained_model} \
--features_list 12 15 18 21 24 --image_size $image_size --batch_size 16 --print_freq 1 \
--epoch 15 --save_freq 1 

done
wait
done
