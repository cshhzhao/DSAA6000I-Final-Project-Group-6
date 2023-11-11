export CUDA_VISIBLE_DEVICES=6
python prompt_lang_eval_math_zhh.py \
    --model_name_or_path_baseline /home/haihongzhao/project/homework/LORA/step1_supervised_finetuning/training_output \
    --model_name_or_path_finetune /home/haihongzhao/project/homework/LORA/step1_supervised_finetuning/training_output \
    --batch_size 8 \
&> /home/haihongzhao/project/homework/LORA/step1_supervised_finetuning/training_output/test_bs2.log
    # xlingual/output_step1_llama2_7b_V1_10epoch
