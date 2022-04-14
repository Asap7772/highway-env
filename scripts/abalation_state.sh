env='highway-fast-v0'
learning_rates=(1e-3 3e-4)
discount_factors=(0.9 0.975)
epsilons=(0.8 0.9 0.95)
layers=(2 4)

total_hyperparam_combinations=$((${#learning_rates[@]} * ${#discount_factors[@]} * ${#epsilons[@]} * ${#layers[@]}))
echo "Total number of hyperparameter combinations: $total_hyperparam_combinations"

GPUS=(0 1 2 3) # for azure
NUM_GPUS=${#GPUS[@]}
NUM_REPEATS=6

# GPUS=(0) # for azure
# NUM_GPUS=${#GPUS[@]}
# NUM_REPEATS=1
# echo "DEBUGGING"

INDEX_START=0
INDEX_END=$((INDEX_START + NUM_GPUS * NUM_REPEATS))
CURRENT_INDEX=0

echo "=========================================================="
echo "Running with $((${NUM_GPUS} * ${NUM_REPEATS})) parallel jobs"
echo "INDEX_START: $INDEX_START"
echo "INDEX_END: $INDEX_END"
echo "=========================================================="

for learning_rate in "${learning_rates[@]}"
do
    for discount_factor in "${discount_factors[@]}"
    do
        for epsilon in "${epsilons[@]}"
        do
            for layer in "${layers[@]}"
            do
                if [ $CURRENT_INDEX -ge $INDEX_START ] && [ $CURRENT_INDEX -lt $INDEX_END ]
                then
                    WHICH_GPU=${GPUS[$((CURRENT_INDEX - INDEX_START)) % NUM_GPUS]}
                    
                    echo "=========================================================="
                    echo "Running on GPU $WHICH_GPU"
                    echo "Learning rate: $learning_rate"
                    echo "Discount factor: $discount_factor"
                    echo "Epsilon: $epsilon"
                    echo "Num Layers: $layer"
                    echo "=========================================================="

                    export CUDA_VISIBLE_DEVICES=$WHICH_GPU

                    python3 /home/asap7772/highway-env/scripts/sb3_highway_dqn.py --env $env --learning_rate $learning_rate \
                    --discount $discount_factor --exploration_fraction $epsilon --num_layers $layer &

                    sleep 1.0
                fi
                CURRENT_INDEX=$(($CURRENT_INDEX + 1))   
            done
        done
    done
done
