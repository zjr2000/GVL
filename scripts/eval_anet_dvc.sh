eval_folder=$1 # specify the folder to be evaluated
model_path=save/${eval_folder}/model-best-dvc.pth
python eval.py --eval_folder ${eval_folder} \
--gpu_id=$2 \
--eval_model_path=${model_path} \
--eval_batch_size=16 \
--eval_caption_file=data/anet/captiondata/val_1.json \
--eval_save_dir save \