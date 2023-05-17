eval_folder=$1 # specify the folder to be evaluated
python eval.py --eval_folder ${eval_folder} \
--gpu_id=$2 \
--eval_save_dir save \
--eval_batch_size=4 \
--eval_gt_file_for_caption=data/tacos/loss_ratio/split_test.json \
--eval_caption_file=data/tacos/loss_ratio/split_test.json \
--eval_gt_file_for_grounding data/tacos/loss_ratio/split_test.json \
--eval_enable_maximum_matching_for_grounding \
--eval_disable_captioning \
# --eval_model_path save3/10_09s1_v4_anno_v5_s3_tacos_etg_teg_lr1e5_TE1e5_head1e4_bsz4_v_2022-10-09-11-20-50_/model-best-dvc.pth \