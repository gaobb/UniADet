# Within-class token t-SNE: early vs late patch tokens, and last-layer CLS vs patches.
source "$(dirname "$0")/env.sh"

dataset=mvtec

for pretrained_model in 'ViT-L-14@336px' 'ViT-L-14-DINOv2Reg' 'ViT-L-16-DINOv3'
do
    case "$pretrained_model" in
    "ViT-L-14@336px"|"ViT-L-14-DINOv2Reg")
        image_size=518
        ;;
    "ViT-L-16-DINOv3")
        image_size=512
        ;;
    *)
        image_size=518
        ;;
    esac

    python tools/tsne_token.py \
        --dataset $dataset \
        --test_data_path ${UNIADET_DATA}/${dataset} \
        --pretrained_model "$pretrained_model" \
        --batch_size 16 \
        --image_size "$image_size" \
        --features_list 12 15 18 21 24 \
        --class_name "hazelnut"
done
