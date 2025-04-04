import os
import json
import ast
import openai
import time
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer,util
from sklearn.metrics import silhouette_score
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("eval.log"),
        logging.StreamHandler()
    ]
)

def load_file(file_name):
    with open(file_name, "r") as file:
        return json.load(file)
    
def get_parser_args():
    parser = ArgumentParser()
    parser.add_argument("--embedding", type=str)
    parser.add_argument("--cluster", type=str)
    parser.add_argument("--choose_cluster_num",type=str)
    parser.add_argument("--summarization_model",type=str)
    parser.add_argument("--max_cluster_num",type=int)
    return parser.parse_args()

logger = logging.getLogger(__name__)

def get_embedding(text):
    client = openai.Client(
        api_key="sk-sR8RiK6YYrtk8Rss1b29047069804d108211285c7a25356c", # replace with your API key
        base_url="https://api.yesapikey.com/v1" # use the default API base URL
    )
    text = text.replace("\n", " ")
    flag = False
    while not flag:
        try:
            embedding = client.embeddings.create(
                input = [text],
                model = 'text-embedding-ada-002',
            ).data[0].embedding
            flag = True
        except:
            print("Error in creating embedding")
            print("Retrying...")
            time.sleep(0.5)
    return embedding

def compute_sihouette_coefficient(all_json_path):
    store_sil_score = {}
    cnt1=0
    for i in range(len(all_json_path)-1):
        
        with open(all_json_path[i],'r',encoding='utf-8') as f:
            cluster_data = json.load(f)
        with open(all_json_path[i+1],'r',encoding='utf-8') as f:
            result_data = json.load(f)
        texts = []
        for item in cluster_data:
            texts.append(item["name"])
        labels = []
        for item in cluster_data:
            name = item["name"]
            id = 0
            for result_item in result_data:
                # result_item["children"] = result_item["children"].replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
                # print(result_item["children"])
                # child = ast.literal_eval()
                child=result_item["children"]
                if name in child:
                    labels.append(id)
                    break
                id +=1
        embeddings = []
        print(f"compute_sihouette_coefficient----------------------------{i}----------")
        print(all_json_path)
        for text in tqdm(texts, desc="Processing texts", unit="text"):
            embed = get_embedding(text)
            embeddings.append(embed)
            # break
        embeddings = np.array(embeddings)
        sil_score = silhouette_score(embeddings, labels)
        print("YES")
        store_sil_score[i] = sil_score
    store_sil_score["average"] = sum(store_sil_score.values())/len(store_sil_score)
    return store_sil_score["average"]

def compute_compact_value(result_data,cluster_data,st_embed_model):
    compact_value_list = []
    for item in result_data:
        description_in_a_cluster = []
        for child in item["children"]:
            for cluster_item in cluster_data:
                if cluster_item["name"] == child:
                    description_in_a_cluster.append(cluster_item["description"])
                    break
        embeddings = st_embed_model.encode(description_in_a_cluster,convert_to_tensor=True)
        cosine_score = util.cos_sim(embeddings,embeddings)
        cosine_score = cosine_score.cpu().detach().numpy()
        # 对cosine_score进行求和取平均
        compact_value = np.mean(cosine_score)
        compact_value_list.append(compact_value)
    return compact_value_list

def compute_relevance_value(result_data,cluster_data,st_embed_model):
    relevance_value_list = []
    for item in result_data:
        parent_feature = item["description"]
        children_feature = []
        for child in item["children"]:
            for cluster_item in cluster_data:
                if cluster_item["name"] == child:
                    children_feature.append(cluster_item["description"])
                    break
        parent_embedding = st_embed_model.encode(parent_feature,convert_to_tensor=True)
        children_embedding = st_embed_model.encode(children_feature,convert_to_tensor=True)
        cosine_score = util.cos_sim(parent_embedding,children_embedding)
        cosine_score = cosine_score.cpu().detach().numpy()
        sum = 0
        count = 0
        for i in range(cosine_score.shape[1]):
            sum += cosine_score[0][i]
            count +=1
        relevance_value = sum/count
        relevance_value_list.append(relevance_value)
    return relevance_value_list

def name_simi_score(str1,str2):
    def edit_distance(s1, s2):
        m, n = len(s1), len(s2)
        dp = np.zeros((m+1, n+1))
        for i in range(m+1):
            for j in range(n+1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j],      # 删除
                                       dp[i][j-1],      # 插入
                                       dp[i-1][j-1])   # 替换
        return dp[m][n]
    len_str1 = len(str1)
    len_str2 = len(str2)
    ed = edit_distance(str1, str2)
    max_len = max(len_str1, len_str2)
    similarity = 1 - (ed / max_len)
    return similarity

def compute_difference_value(result_data,st_embed_model):
    all_name = []
    all_description = []
    for item in result_data:
        all_name.append(item["name"])
        all_description.append(item["description"])
    desc_embed = st_embed_model.encode(all_description,convert_to_tensor=True)
    desc_simi = util.cos_sim(desc_embed,desc_embed)
    desc_simi = desc_simi.cpu().detach().numpy()
    name_edit_dis = []
    for i,name1 in enumerate(all_name):
        row=[]
        for j,name2 in enumerate(all_name):
                row.append(name_simi_score(name1,name2))
        name_edit_dis.append(row)
    total_difference_list = []
    total_difference = 0
    for i in range(desc_simi.shape[0]):
        for j in range(desc_simi.shape[1]):
            if i==j:
                continue
            desc_diff = 1 - (desc_simi[i][j] + 1) / 2
            # print(name_edit_dis)
            name_diff = name_edit_dis[i][j]
            total_difference = total_difference + desc_diff + name_diff
    total_difference = total_difference / (desc_simi.shape[0] * (desc_simi.shape[1]-1))
    total_difference_list.append(total_difference)
    return total_difference_list
    

def compute_gvalue(all_json_file,st_embed_model):
    store_gvalue = {}
    compact_value_list = []
    relevance_value_list = []
    difference_value_list = []
    for i in range(len(all_json_path)-1):
        with open(all_json_path[i],'r',encoding='utf-8') as f:
            cluster_data = json.load(f)
        with open(all_json_path[i+1],'r',encoding='utf-8') as f:
            result_data = json.load(f)
        compact_value = compute_compact_value(result_data,cluster_data,st_embed_model)
        relevance_value = compute_relevance_value(result_data,cluster_data,st_embed_model)
        difference_value = compute_difference_value(result_data,st_embed_model)
        compact_value_list.extend(compact_value)
        relevance_value_list.extend(relevance_value)
        difference_value_list.extend(difference_value)
    compact_average = sum(compact_value_list)/len(compact_value_list)
    relevance_average = sum(relevance_value_list)/len(relevance_value_list)
    difference_average = sum(difference_value_list)/len(difference_value_list)
    gvalue = (compact_average + relevance_average + difference_average) / 3
    return gvalue


def main():
    metric = {}
    st_embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    args = get_parser_args()
    # if args.cluster == "hierarchical":
    #     args.choose_cluster = "None"
    all_json_path = ["result/0_layer.json"]
    input_dir = f"result/{args.embedding}_{args.cluster}_{args.choose_cluster}_4_{args.max_cluster}_{args.summarize_model}"
    logger.info(f"Computing metric for {args.embedding}_{args.cluster}_{args.choose_cluster}_4_{args.max_cluster}_{args.summarize_model}")
    for file in os.listdir(input_dir):
        if file.endswith(".json"):
            all_json_path.append(os.path.join(input_dir, file))
    sihouette = compute_sihouette_coefficient(all_json_path)
    gvalue = compute_gvalue(all_json_path,st_embed_model)
    metric["silhouette"] = sihouette
    metric["gvalue"] = gvalue
    emb,clu,ch,max_cluster,summarize_model = args.embedding,args.cluster,args.choose_cluster,args.max_cluster,args.summarize_model
    logger.info(f"{emb}_{clu}_{ch}_{max_cluster}_{summarize_model}-------silhouette:{sihouette}------gvalue:{gvalue}")
    with open(f"{input_dir}/metric.json", "w") as f:
        json.dump(metric, f, indent=4)

if __name__=="__main__":
    main()

            