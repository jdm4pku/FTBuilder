import os
import json

embeddings = ["TF-IDF","all-MiniLM-L6-v2","all-mpnet-base-v2","text-embedding-ada-002"]
cluster = ["kmeans","gmm"]
choose = ["elbow","sihouette","BIC"]
summarize_model = "gpt-4o"
max_cluster = 80

def compute_node(all_json_path):
    node_num = 0
    for i in range(len(all_json_path)):
        with open(all_json_path[i],'r',encoding='utf-8') as f:
            cluster_data = json.load(f)
            node_num +=len(cluster_data)
    return node_num

tree_info = {}
for embed in embeddings:
    for clu in cluster:
        for ch in choose:
            if clu == "kmeans" and ch == "BIC":
                continue
            all_json_path = ["result/0_layer.json"]
            input_dir = f"result/{embed}_{clu}_{ch}_4_{max_cluster}_{summarize_model}"
            # 将input_dir中的所有json文件路径加入all_json_path
            for file in os.listdir(input_dir):
                if file.endswith(".json"):
                    all_json_path.append(os.path.join(input_dir, file))
            layer = len(all_json_path)
            node = compute_node(all_json_path)
            tree_info[f"{embed}_{clu}_{ch}_{max_cluster}_{summarize_model}"] = {"layer":layer,"node":node}

output_dir = f"tree_info"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(f"{output_dir}/tree_info.json",'w',encoding='utf-8') as f:
    json.dump(tree_info,f,indent=4)