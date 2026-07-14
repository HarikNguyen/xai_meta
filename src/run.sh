#!/bin/bash

# Training
# python main.py --config ./configs/mini2cub.yaml --mode train --vmap_chunk_size 4
# python main.py --config ./configs/tiered2cub.yaml --mode train --vmap_chunk_size 4
# python main.py --config ./configs/tiered.yaml --mode train --vmap_chunk_size 4

# Check n Explain

echo "Check n Explain - Mini 2 CUB"
python main.py --config ./configs/mini2cub.yaml --checkpoint_dir MAML_mini2cub --log_dir mini2cub --mode test --use_last
python main.py --config ./configs/mini2cub.yaml --checkpoint_dir MAML_mini2cub --log_dir mini2cub --mode explain --use_last

echo "Check n Explain - Tiered 2 CUB"
python main.py --config ./configs/tiered2cub.yaml --checkpoint_dir MAML_tiered2cub --log_dir tiered2cub --mode test --use_last
python main.py --config ./configs/tiered2cub.yaml --checkpoint_dir MAML_tiered2cub --log_dir tiered2cub --mode explain --use_last

echo "Check n Explain - Tiered"
python main.py --config ./configs/tiered.yaml --checkpoint_dir MAML_tiered_tiered --log_dir tiered --mode test --use_last
python main.py --config ./configs/tiered.yaml --checkpoint_dir MAML_tiered_tiered --log_dir tiered --mode explain --use_last

# Check Explain Method (biADT + sanity check params + sanity support - hard/noise/ood)
echo "Check Explain Method"
python main.py --config ./configs/tiered.yaml --mode check_explain --use_last --check_method biADT
python main.py --config ./configs/tiered.yaml --mode check_explain --use_last --check_method sanity_params
python main.py --config ./configs/tiered.yaml --mode check_explain --use_last --check_method sanity_support_set

echo "DONE!"
