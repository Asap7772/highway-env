min_q_weights=(0.1 1 2 5)
method_temps=(1.0 2.0 3.0 20.0)

gpus=(0 1 2 3 4 5 6 7)
num_repeat=2

current_index=0
start_index=0
end_index=$(($start_index + ${#gpus[@]}*$num_repeat - 1))

echo "start_index: $start_index"
echo "end_index: $end_index"

for i in "${min_q_weights[@]}"
do
    for j in "${method_temps[@]}"
    do
        if [ $current_index -ge $start_index ] && [ $current_index -le $end_index ]; then
            which_gpu=$(($current_index % ${#gpus[@]}))
            gpu_id=${gpus[$which_gpu]}
            
            echo "=========================================================="
            echo "GPU ID: $gpu_id"
            echo "min_q_weight: $i"
            echo "method_temp: $j"
            echo "=========================================================="

            export CUDA_VISIBLE_DEVICES=$gpu_id
            python /home/asap7772/asap7772/highway-env/scripts/sb3_highway_cql.py \
            --env highway-fast-v0 --method --method_temp $j --min_q_weight $i &
        fi
        current_index=$(($current_index + 1))
    done
done
