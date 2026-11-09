# Cross-dataset normal/anomaly patch t-SNE. dataset_pair: visa-mvtec, visa-medical, mvtec-medical.
source "$(dirname "$0")/env.sh"

for dataset_pair in  "visa-mvtec"  "visa-medical"
do
for pretrained_model in 'ViT-L-14@336px' 'ViT-L-14-DINOv2Reg' 'ViT-L-16-DINOv3'
do
    case "$pretrained_model" in
    "ViT-L-14@336px"|"ViT-L-14-DINOv2Reg")
        image_size=518
        batch_size=14
        ;;
    "ViT-L-16-DINOv3")
        image_size=512
        batch_size=16
        ;;
    *)
        image_size=518
        batch_size=14
        ;;
    esac

    python tools/tsne_dataset_pair.py \
        --pretrained_model "$pretrained_model" \
        --batch_size "$batch_size" \
        --image_size "$image_size" \
        --features_list 12 15 18 21 24 \
        --dataset_pair "$dataset_pair" \
        --test_data_path "$UNIADET_DATA"
done
done
