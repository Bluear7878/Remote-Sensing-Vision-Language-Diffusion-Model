source /home/delta1/miniconda3/etc/profile.d/conda.sh



# original image d 
export HQ_dir="/home/delta1/KKW/datasets/CUTE80/CUTE80_original"

# SR image folder path
export SR_HQ_dir="/home/delta1/Texture/GYLPH-SR/metric/text_focus_batchify_sample_custom_v2"

# txt name 

conda activate metrics
CUDA_VISIBLE_DEVICES=1 python delta_metric.py --ori_dir $HQ_dir --sr_dir $SR_HQ_dir --res_path "./all_score"

