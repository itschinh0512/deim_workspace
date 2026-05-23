#!/bin/bash
# ============================================================
# OHEM Experiment Runner
# Usage: ./run_ohem_experiments.sh [strategy] [experiment]
#
# Examples:
#   ./run_ohem_experiments.sh s2 B1    # Run Strategy 2, Experiment B1
#   ./run_ohem_experiments.sh s1 O1    # Run Strategy 1, Experiment O1
#   ./run_ohem_experiments.sh s4 Z1    # Run Strategy 4, Experiment Z1
#   ./run_ohem_experiments.sh all      # Run all experiments sequentially
# ============================================================

PYTHON="/home/locth/miniconda3/envs/deimv2/bin/python"
TRAIN_SCRIPT="train.py"
CONFIG_DIR="configs/deimv2/ohem_experiments"
DEVICE="0"

# Map experiment names to config files
declare -A CONFIGS
CONFIGS[s2_B1]="s2_B1_gamma2.0.yml"
CONFIGS[s2_B2]="s2_B2_gamma2.5.yml"
CONFIGS[s2_B3]="s2_B3_gamma2.0_alpha0.5.yml"
CONFIGS[s2_B4]="s2_B4_gamma2.0_progloss.yml"
CONFIGS[s1_O1]="s1_O1_ohem_neg3.0.yml"
CONFIGS[s1_O2]="s1_O2_ohem_neg5.0.yml"
CONFIGS[s1_O3]="s1_O3_ohem_warmup20.yml"
CONFIGS[s4_Z1]="s4_Z1_zone_a0.5.yml"
CONFIGS[s4_Z2]="s4_Z2_zone_a1.0.yml"
CONFIGS[s4_Z3]="s4_Z3_zone_a2.0_b3.0.yml"

# Recommended execution order
ORDERED_KEYS=(s2_B1 s2_B2 s2_B3 s2_B4 s1_O1 s1_O2 s1_O3 s4_Z1 s4_Z2 s4_Z3)

run_experiment() {
    local key=$1
    local config_file="${CONFIG_DIR}/${CONFIGS[$key]}"
    
    if [ ! -f "$config_file" ]; then
        echo "ERROR: Config file not found: $config_file"
        return 1
    fi
    
    echo "============================================================"
    echo "Running experiment: $key"
    echo "Config: $config_file"
    echo "Started at: $(date)"
    echo "============================================================"
    
    CUDA_VISIBLE_DEVICES=$DEVICE $PYTHON $TRAIN_SCRIPT \
        -c "$config_file" \
        --use-amp \
        -d cuda:0
    
    echo "Finished experiment: $key at $(date)"
    echo ""
}

if [ "$1" == "all" ]; then
    echo "Running ALL experiments sequentially..."
    echo "Estimated total time: ~${#ORDERED_KEYS[@]} × 12h = $((${#ORDERED_KEYS[@]} * 12))h"
    echo ""
    for key in "${ORDERED_KEYS[@]}"; do
        run_experiment "$key"
    done
elif [ -n "$1" ] && [ -n "$2" ]; then
    key="${1}_${2}"
    if [ -z "${CONFIGS[$key]}" ]; then
        echo "Unknown experiment: $key"
        echo "Available experiments:"
        for k in "${ORDERED_KEYS[@]}"; do echo "  $k"; done
        exit 1
    fi
    run_experiment "$key"
elif [ -n "$1" ]; then
    # Run all experiments for a strategy
    echo "Running all $1 experiments..."
    for key in "${ORDERED_KEYS[@]}"; do
        if [[ "$key" == ${1}_* ]]; then
            run_experiment "$key"
        fi
    done
else
    echo "Usage: $0 [strategy] [experiment]"
    echo "       $0 all"
    echo ""
    echo "Available experiments:"
    for key in "${ORDERED_KEYS[@]}"; do
        echo "  $key -> ${CONFIGS[$key]}"
    done
fi
