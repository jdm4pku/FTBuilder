#!/bin/bash

input_dir="library/merge_result/unified_library.json"
embeddings=("TF-IDF" "all-MiniLM-L6-v2" "all-mpnet-base-v2" "text-embedding-ada-002")  
clusters=("kmeans" "gmm")  
choose_cluster_nums=("elbow" "sihouette" "BIC")  
min_stop_cluster_num=4
max_cluster_num=80
summarization_models=("gpt-4o")
max_tokens=512
max_layer_num=5
output_dir="result"

for embedding in "${embeddings[@]}"; do
    for cluster in "${clusters[@]}"; do
        for choose_cluster_num in "${choose_cluster_nums[@]}"; do
            for summarization_model in "${summarization_models[@]}"; do
                CUDA_VISIBLE_DEVICES=0 python src/tree_construction.py \
                    --input_dir $input_dir \
                    --embedding $embedding \
                    --cluster $cluster \
                    --choose_cluster_num $choose_cluster_num \
                    --min_stop_cluster_num $min_stop_cluster_num \
                    --max_cluster_num $max_cluster_num \
                    --summarization_model $summarization_model \
                    --max_tokens $max_tokens \
                    --max_layer_num $max_layer_num \
                    --output_dir $output_dir
            done
        done
    done
done

for embedding in "${embeddings[@]}"; do
    for summarization_model in "${summarization_models[@]}"; do
        # 运行 tree_construct.py 脚本
        CUDA_VISIBLE_DEVICES=0 python src/tree_construction.py \
            --input_dir $input_dir \
            --embedding $embedding \
            --cluster "hierarchical" \
            --min_stop_cluster_num $min_stop_cluster_num \
            --max_cluster_num $max_cluster_num \
            --summarization_model $summarization_model \
            --max_tokens $max_tokens \
            --max_layer_num $max_layer_num \
            --output_dir $output_dir
    done
done
