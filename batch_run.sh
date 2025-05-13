#!/bin/bash

# Directory containing .yml files
config_dir="/root/autodl-tmp/workdir/llmc/config_llama_tmp/backup"

# Iterate over all .yml files in the directory
for config in "$config_dir"/*.yml; do
    # Skip if no .yml files are found
    [[ -f "$config" ]] || continue

    # Extract task_name from the basename of the .yml file (without extension)
    task_name=$(basename "$config" .yml)

    # Set environment variables and paths
    llmc=/root/autodl-tmp/workdir/llmc
    export PYTHONPATH=$llmc:$PYTHONPATH

    nnodes=1
    nproc_per_node=1

    # Function to find an unused port
    find_unused_port() {
        while true; do
            port=$(shuf -i 10000-60000 -n 1)
            if ! ss -tuln | grep -q ":$port "; then
                echo "$port"
                return 0
            fi
        done
    }
    UNUSED_PORT=$(find_unused_port)

    # Set distributed training parameters
    MASTER_ADDR=127.0.0.1
    MASTER_PORT=$UNUSED_PORT
    task_id=$UNUSED_PORT

    # Run the torchrun command and wait for it to complete
    torchrun \
    --nnodes $nnodes \
    --nproc_per_node $nproc_per_node \
    --rdzv_id $task_id \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    ${llmc}/llmc/__main__.py --config "$config" --task_id $task_id \
    > "${task_name}.log" 2>&1

    # Save the process ID (optional, since the process has completed)
    ps aux | grep '__main__.py' | grep $task_id | awk '{print $2}' > "${task_name}.pid"

    echo "Completed task: $task_name with config $config"
done