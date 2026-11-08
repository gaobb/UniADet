source "$(dirname "$0")/env.sh"

train_dataset=visa

for test_dataset in medical-cls medical-seg
do 
    for pretrained_model in 'ViT-L/14@336px' 'ViT-L-16-DINOv3' 'ViT-L-14-DINOv2Reg'
    do
        case "$pretrained_model" in
        "ViT-L/14@336px"|"ViT-L-14-DINOv2Reg")
            image_size=518
            ;;
        "ViT-L-16-DINOv3")
            image_size=512 #592
            ;;
        *)
            image_size=518
            ;;
        esac
        echo $image_size
        
        save_path=${UNIADET_RESULTS}/test_${test_dataset}_${train_dataset}_${pretrained_model}
        checkpoint_path=${UNIADET_CKPT}/train_on_${train_dataset}_zs_${pretrained_model}_imagesize${image_size}_1015
        epoch=15

        
        # for zero-shot
        k_shot=0
        lamda_pixel=0.5
        lamda_fs=0.1 
        CUDA_VISIBLE_DEVICES=${UNIADET_DEVICE} python test.py --dataset ${test_dataset} --test_data_path ${UNIADET_DATA}/${test_dataset} \
        --save_path ${save_path} \
        --pretrained_model ${pretrained_model} \
        --checkpoint_path ${checkpoint_path}/epoch_${epoch}.pth \
        --k_shot ${k_shot} \
        --window_list 1 3 5 \
        --lamda_fs $lamda_fs \
        --lamda_pixel $lamda_pixel \
        --features_list 12 15 18 21 24 --image_size $image_size --batch_size 32

        # for few-shot
        for k_shot in 1 2 4 
        do
            lamda_pixel=0.5
            lamda_fs=0.1
            for seed in 10 20 30
            do
                CUDA_VISIBLE_DEVICES=${UNIADET_DEVICE} python test.py --dataset ${test_dataset} --test_data_path ${UNIADET_DATA}/${test_dataset} \
                --save_path ${save_path} \
                --pretrained_model ${pretrained_model} \
                --checkpoint_path ${checkpoint_path}/epoch_${epoch}.pth \
                --seed ${seed} \
                --k_shot ${k_shot} \
                --window_list 1 3 5 \
                --lamda_fs $lamda_fs \
                --lamda_pixel $lamda_pixel \
                --features_list 12 15 18 21 24 --image_size $image_size --batch_size 32
            done
        done
    done
done

