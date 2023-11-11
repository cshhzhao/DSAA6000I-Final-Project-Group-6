export CUDA_VISIBLE_DEVICES=6
python eval_fake_news_detection_zhh.py \
    --model_name_or_path_baseline /data1/haihongzhao/DSAA6000I-Final-Project-Group-7/training_output \
    --model_name_or_path_finetune /data1/haihongzhao/DSAA6000I-Final-Project-Group-7/training_output \
    --batch_size 8 \
&> /data1/haihongzhao/DSAA6000I-Final-Project-Group-7/training_output/test_bs2.log
    # xlingual/output_step1_llama2_7b_V1_10epoch
