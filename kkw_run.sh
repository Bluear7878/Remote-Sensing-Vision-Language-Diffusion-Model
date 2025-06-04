source /home/delta1/miniconda3/etc/profile.d/conda.sh
trap 'kill 0' SIGINT SIGTERM EXIT # 한번에 종료

conda activate llama3_ft
cd /home/delta1/Texture/GYLPH-SR

# # CUTE ##########
# CUDA_VISIBLE_DEVICES=0 python kkw_Gsr.py \
#     --json_path /home/delta1/Texture/Prompt/prompt_v0/CUTE80_image_x4_prompts.jsonl \
#     --save_dir /home/delta1/KKW/SR_sample/output/image_ratio_results/CUTE80/imagex4 \
#     --img_path /home/delta1/KKW/datasets/CUTE80 &
# CUDA_VISIBLE_DEVICES=1 python kkw_Gsr.py \
#     --json_path /home/delta1/Texture/Prompt/prompt_v0/CUTE80_image_x6_prompts.jsonl \
#     --save_dir /home/delta1/KKW/SR_sample/output/image_ratio_results/CUTE80/imagex6 \
#     --img_path /home/delta1/KKW/datasets/CUTE80 &
# CUDA_VISIBLE_DEVICES=2 python kkw_Gsr.py \
#     --json_path /home/delta1/Texture/Prompt/prompt_v0/CUTE80_image_x8_prompts.jsonl \
#     --save_dir /home/delta1/KKW/SR_sample/output/image_ratio_results/CUTE80/imagex8 \
#     --img_path /home/delta1/KKW/datasets/CUTE80 
# wait



# SVT ##########
CUDA_VISIBLE_DEVICES=0 python kkw_Gsr.py \
    --json_path /home/delta1/Texture/Prompt/prompt_v0/SVT_x4_prompts.jsonl \
    --save_dir /home/delta1/KKW/SR_sample/output/image_ratio_results/SVT/imagex4 \
    --img_path /home/delta1/KKW/datasets/SVT &
CUDA_VISIBLE_DEVICES=1 python kkw_Gsr.py \
    --json_path /home/delta1/Texture/Prompt/prompt_v0/SVT_x6_prompts.jsonl \
    --save_dir /home/delta1/KKW/SR_sample/output/image_ratio_results/SVT/imagex6 \
    --img_path /home/delta1/KKW/datasets/SVT &
CUDA_VISIBLE_DEVICES=2 python kkw_Gsr.py \
    --json_path /home/delta1/Texture/Prompt/prompt_v0/SVT_x8_prompts.jsonl \
    --save_dir /home/delta1/KKW/SR_sample/output/image_ratio_results/SVT/imagex8 \
    --img_path /home/delta1/KKW/datasets/SVT 
wait



# # SCUT-CTW1500 ##########
# CUDA_VISIBLE_DEVICES=0 python kkw_Gsr.py \
#     --json_path /home/delta1/Texture/Prompt/prompt_v0/SCUT_CTW1500_x4_prompts.jsonl \
#     --save_dir /home/delta1/KKW/SR_sample/output/image_ratio_results/SCUT_CTW1500/imagex4 \
#     --img_path /home/delta1/KKW/datasets/SCUT_CTW1500 &
# CUDA_VISIBLE_DEVICES=1 python kkw_Gsr.py \
#     --json_path /home/delta1/Texture/Prompt/prompt_v0/SCUT_CTW1500_x6_prompts.jsonl \
#     --save_dir /home/delta1/KKW/SR_sample/output/image_ratio_results/SCUT_CTW1500/imagex6 \
#     --img_path /home/delta1/KKW/datasets/SCUT_CTW1500 &
# CUDA_VISIBLE_DEVICES=2 python kkw_Gsr.py \
#     --json_path /home/delta1/Texture/Prompt/prompt_v0/SCUT_CTW1500_x8_prompts.jsonl \
#     --save_dir /home/delta1/KKW/SR_sample/output/image_ratio_results/SCUT_CTW1500/imagex8 \
#     --img_path /home/delta1/KKW/datasets/SCUT_CTW1500 
# wait


