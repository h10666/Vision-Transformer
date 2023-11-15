python3 train.py --modality RGB Spec --name ucf101_result --dataset ucf101 --model_type ViT-B_16 \
        --pretrained_dir checkpoint/ViT-B_16.npz --gradient_accumulation_steps=1 \
        --train_list /opt/data/private/datasets/UCF101/ucfTrainTestlist/train.txt \
        --val_list /opt/data/private/datasets/UCF101/ucfTrainTestlist/val.txt \
        --visual_path /opt/data/private/datasets/UCF101/ucf101_frames_blur/ \
        --audio_path /opt/data/private/datasets/UCF101/audio_dict.pkl  --num_segments 8 --local_rank -1