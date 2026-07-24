#!/bin/bash
set -e

# Training (kept commented -- run these manually one at a time, they take
# hours; not meant to run as part of this batch script)
# for scenario in only_mini only_cub only_tiered mini2cub tiered2cub tiered2strokes_omnig; do
#   for backbone in conv4 resnet10; do
#     python main.py --config ./configs/${scenario}/${backbone}.yaml --mode train --vmap_chunk_size 4
#   done
# done

# Scenarios to sweep. Each entry is "config_folder:log_label" -- log_label is
# used as the base name for checkpoint_dir/log_dir so results from different
# scenarios/backbones don't collide.
SCENARIOS=(
  "only_mini:only_mini"
  "only_cub:only_cub"
  "only_tiered:only_tiered"
  "mini2cub:mini2cub"
  "tiered2cub:tiered2cub"
  "tiered2strokes_omnig:tiered2strokes_omnig"
)
BACKBONES=("conv4" "resnet10")

for entry in "${SCENARIOS[@]}"; do
  folder="${entry%%:*}"
  label="${entry##*:}"

  for backbone in "${BACKBONES[@]}"; do
    # conv4 keeps the plain label (matches checkpoints already trained this
    # way, e.g. mini2cub_ckpt / mini2cub_logs); resnet10 gets an explicit
    # suffix (mini2cub_resnet10_ckpt / mini2cub_resnet10_logs).
    if [ "$backbone" = "conv4" ]; then
      name="$label"
    else
      name="${label}_${backbone}"
    fi

    config="./configs/${folder}/${backbone}.yaml"
    explain_config="./configs/${folder}/explain_test_${backbone}.yaml"
    ckpt_dir="${name}_ckpt"

    echo "=== ${folder} / ${backbone} ==="

    echo "-- test --"
    python main.py --config "$config" --checkpoint_dir "$ckpt_dir" --log_dir "${name}_logs" --mode test --use_last

    echo "-- explain --"
    python main.py --config "$config" --checkpoint_dir "$ckpt_dir" --log_dir "${name}_logs" --mode explain --use_last
    python main.py --config "$config" --checkpoint_dir "$ckpt_dir" --log_dir "${name}_logs_exwf" --mode explain --use_last --flip_ratio 0.75
    python main.py --config "$config" --checkpoint_dir "$ckpt_dir" --log_dir "${name}_logs_exb" --mode explain --use_last --blur

    echo "-- check_explain (biADT + sanity_params + sanity_support_set) --"
    for method in biADT sanity_params sanity_support_set; do
      python main.py --config "$explain_config" --checkpoint_dir "$ckpt_dir" --log_dir "${name}_logs_${method}" --mode check_explain --use_last --check_method "$method"
    done
  done
done

echo "DONE!"
