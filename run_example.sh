mkdir logs_date_mixup_5994speaker_10utters_GPU67_lr001_alpha04_repeat_exp4
export CUDA_VISIBLE_DEVICES=6,7
TAG=exp_resnetse34l_angleproto_batch400_nperSpeaker2_distributed4_lr0001_date_mixup_5994speaker_10utters_GPU67_lr001_alpha04_repeat_exp4
rm -rf exps/${TAG}

# Training
python3 ./trainSpeakerNet.py --model ResNetSE34L --trainfunc ap_loss --alpha 0.4 \
        --log_input True --port 2856 --distributed \
        --mixup_mode overlap \
        --train_list data/train_list_5994speakers_10utterances.txt \
        --test_interval 10 --scheduler steplr \
        --lr 0.001 --max_epoch 500 \
        --batch_size 400 --nPerSpeaker 2 \
        --save_path exps/${TAG} 2>&1 | tee -a logs_date_mixup_5994speaker_10utters_GPU67_lr001_alpha04_repeat_exp4/${TAG}.log
# Evaluation
python3 ./trainSpeakerNet.py --model ResNetSE34L --trainfunc ap_loss --alpha 0.0 \
        --log_input True --port 2856 --distributed \
        --batch_size 400 --nPerSpeaker 2 \
	      --eval --eval_frames 400 \
        --save_path exps/${TAG} 2>&1 | tee -a logs_date_mixup_5994speaker_10utters_GPU67_lr001_alpha04_repeat_exp4/${TAG}.log
:
