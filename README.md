# dm-utils
Utility scripts for working with diffusion models


## Textual Inversion
accelerate launch textual_inversion_train.py \
  --pretrained_model_name_or_path=$MODEL_ID \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<class01>" \
  --initializer_token="fish" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="ti_class01" 