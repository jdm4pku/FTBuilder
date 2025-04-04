

embeddings=("TF-IDF" "all-MiniLM-L6-v2" "all-mpnet-base-v2" "text-embedding-ada-002")  
clusters=("kmeans" "gmm")  
choose_cluster_nums=("elbow" "sihouette" "BIC") 
max_cluster_num=80
summarization_models="gpt-4o"

for embedding in "${embeddings[@]}"; do
    for cluster in "${clusters[@]}"; do
        for choose_cluster_num in "${choose_cluster_nums[@]}"; do
            for summarization_model in "${summarization_models[@]}"; do
                echo "going on ${embedding} ${cluster} ${choose_cluster_num} ${summarization_model}"
                if [[ "$cluster" == "kmeans" && "$choose_cluster_num" == "BIC" ]]; then
                    continue
                fi
                CUDA_VISIBLE_DEVICES=0 python src/tree_eval.py \
                    --embedding $embedding \
                    --cluster $cluster \
                    --choose_cluster_num $choose_cluster_num \
                    --max_cluster_num $max_cluster_num \
                    --summarization_model $summarization_model
            done
        done
    done
done

for embedding in "${embeddings[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python src/tree_eval.py \
        --embedding $embedding \
        --cluster "hierarchical" \
        --choose_cluster_num "none" \
        --max_cluster_num $max_cluster_num \
        --summarization_model $summarization_model
done