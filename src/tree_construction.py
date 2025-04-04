import os
import re
import time
import json
import umap
import openai
import logging
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from kneed import KneeLocator
from sklearn.cluster import KMeans
from argparse import ArgumentParser
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tree_construct.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_file(file_name):
    with open(file_name,'r',encoding='utf-8') as f:
        return json.load(f)
    
def get_parser_args():
    parser = ArgumentParser(description="Tree Construction")
    parser.add_argument("--input_dir", type=str, default="library/merge_result")
    parser.add_argument("--embedding",type=str, choices=["TF-IDF", "all-MiniLM-L6-v2", "all-mpnet-base-v2", "text-embedding-ada-002"])
    parser.add_argument("--cluster", type=str, choices=["kmeans","gmm","hierarchical"])
    parser.add_argument("--choose_cluster_num",type=str,choices=["elbow","sihouette","BIC"])
    parser.add_argument("--min_stop_cluster_num",type=int,default=4)
    parser.add_argument("--max_cluster_num", type=int,default=100) 
    parser.add_argument("--summarization_model",type=str,choices=["gpt-4o"])
    parser.add_argument("--max_tokens",type=int,default=512)
    parser.add_argument("--max_layer_num", type=int,default=4)
    parser.add_argument("--output_dir", type=str, default="feature_tree/construct_tree")
    return parser.parse_args()

def get_embedding(args,node):
    if args.embedding == "TF-IDF":
        vectorizer = TfidfVectorizer()
        descriptions = [n['description'] for n in node]
        embeddings = vectorizer.fit_transform(descriptions)
        numpy_embeddings = embeddings.toarray() 
        return numpy_embeddings
    if args.embedding == "all-MiniLM-L6-v2" or args.embedding == "all-mpnet-base-v2":
        model = SentenceTransformer(args.embedding)
        descriptions = [n['description'] for n in node] 
        embeddings = model.encode(descriptions)
        return np.array(embeddings)
    if args.embedding == "text-embedding-ada-002":
        descriptions = [n['description'] for n in node]
        client = openai.Client(
            api_key="sk-xxx", # replace with your API key
            base_url="xxx" # use the default API base URL
        )
        def create_openai_embedding(client, text,model):
            text = text.replace("\n", " ")
            flag = False
            while not flag:
                try:
                    embedding = client.embeddings.create(
                        input = [text],
                        model = model,
                    ).data[0].embedding
                    flag = True
                except:
                    print("Error in creating embedding")
                    print("Retrying...")
                    time.sleep(0.5)
            return embedding
        embeddings = np.array([create_openai_embedding(client, text, args.embedding) for text in tqdm(descriptions)])
        return embeddings
    
def get_summarization(args,context):
    format_example = {
        "name": "Parent Feature Name",
        "description": "Detailed description of the parent feature"
    }
    prompt= f"Please generate a parent feature that can cover the following sub-features.\nThe sub-features are: \n{context}The output should be in JSON format, including the feature name and description.Ensure the structure is clear and the content follows this format:{format_example}. \n\n Output: "
    if args.summarization_model=="gpt-4o":
        flag = False
        while not flag:
            try:
                client = OpenAI(
                    api_key="sk-xxx", # replace with your API key
                    base_url="xxx") # use the default API base URL) # 替换成你的api-key
                response = client.chat.completions.create(
                    model = "gpt-4o-2024-11-20",
                    messages=[
                        {"role": "system", "content": "You are an expert in Linux functional analysis and feature modeling."},
                        {
                            "role": "user",
                            # "content": f"Based on the following sub-features, please generate a parent feature that can cover these sub-features. The sub-features are: {context}: \n Please only output parent feature in the format of 'feature name:\n feature description:'.",
                            "content": prompt,
                        },
                    ],
                    max_tokens=args.max_tokens,
                )
                flag = True
            except Exception as e:
                print("Error in generating summarization")
                print("Retrying...")
                time.sleep(0.5)
        print(f"prompt:{prompt}")
        return response.choices[0].message.content

def construct_tree(args):
    current_level_nodes = load_file(args.input_dir) # List[Dict]
    layer_to_nodes = {0:current_level_nodes} # Dict[int,List[Dict]]
    for layer in range(1,args.max_layer_num):
        next_level_nodes = []
        print(f"Constructing Layer {layer}--------------------------------------------------------------------------------------------------")
        if len(current_level_nodes) <= args.min_stop_cluster_num:
            print(f"Stop clustering at layer {layer} because the number of nodes is less than {args.min_stop_cluster_num}")
            break
        # 1. get embedding
        embeddings = get_embedding(args,current_level_nodes)
        # 2. reduce dimension
        n_neighbors = int((len(embeddings) - 1) ** 0.5) 
        n_components = 10 
        embeddings = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=n_components).fit_transform(embeddings)
        # 3. clustering
        if args.cluster == "hierarchical":
            clustering = AgglomerativeClustering(n_clusters=args.max_cluster_num)
            clusters = clustering.fit_predict(embeddings)
            node_clusters = []
            for label in np.unique(clusters):
                indices = [i for i, cluster in enumerate(clusters) if cluster == label]
                cluster_nodes = [current_level_nodes[i] for i in indices]
                node_clusters.append(cluster_nodes)

        max_cluster_num = min(args.max_cluster_num, len(embeddings)-1)
        if args.cluster == "kmeans":
            if args.choose_cluster_num == "elbow":
                sse = []
                for i in tqdm(range(1, max_cluster_num),desc="choose the best cluster num"):
                    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
                    kmeans.fit(embeddings)
                    sse.append(kmeans.inertia_)
                kn = KneeLocator(range(1, max_cluster_num), sse, curve='convex', direction='decreasing')
                best_cluster_num = kn.knee
            if args.choose_cluster_num == "sihouette":
                silhouette = []
                for i in tqdm(range(2, max_cluster_num),desc="choose the best cluster num"):
                    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
                    kmeans.fit(embeddings)
                    silhouette.append(silhouette_score(embeddings, kmeans.labels_))
                best_cluster_num = silhouette.index(max(silhouette)) + 2
            if args.choose_cluster_num == "BIC":
                print("BIC is not supported for KMeans")
                exit(0)
            kmeans = KMeans(n_clusters=best_cluster_num, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(embeddings)
            clusters = kmeans.labels_ # List[int]
            node_clusters = []
            for label in np.unique(clusters):
                indices = [i for i, cluster in enumerate(clusters) if cluster==label]
                cluster_nodes = [current_level_nodes[i] for i in indices]
                node_clusters.append(cluster_nodes)
        if args.cluster == "gmm":
            if args.choose_cluster_num == "elbow":
                sse = []
                for i in tqdm(range(1, max_cluster_num),desc="choose the best cluster num"):
                    gmm = GaussianMixture(n_components=i, random_state=0)
                    gmm.fit(embeddings)
                    sse.append(gmm.bic(embeddings))
                kn = KneeLocator(range(1, max_cluster_num), sse, curve='convex', direction='decreasing')
                best_cluster_num = kn.knee
            if args.choose_cluster_num == "sihouette":
                silhouette = []
                for i in tqdm(range(2, max_cluster_num),desc="choose the best cluster num"):
                    gmm = GaussianMixture(n_components=i, random_state=0)
                    gmm.fit(embeddings)
                    silhouette.append(silhouette_score(embeddings, gmm.predict(embeddings)))
                best_cluster_num = silhouette.index(max(silhouette)) + 2
            if args.choose_cluster_num == "BIC":
                bic = []
                for i in tqdm(range(1, max_cluster_num),desc="choose the best cluster num"):
                    gmm = GaussianMixture(n_components=i, random_state=0)
                    gmm.fit(embeddings)
                    bic.append(gmm.bic(embeddings))
                best_cluster_num = np.arange(1,max_cluster_num)[np.argmin(bic)]
            gmm = GaussianMixture(n_components=best_cluster_num, random_state=0)
            gmm.fit(embeddings)
            clusters = gmm.predict(embeddings)
            node_clusters = []
            for label in np.unique(clusters):
                indices = [i for i, cluster in enumerate(clusters) if label==cluster]
                cluster_nodes = [current_level_nodes[i] for i in indices]
                node_clusters.append(cluster_nodes)
        # 4. generate parent feature
        for cluster in tqdm(node_clusters,desc="generating parent feature"):
            def get_origin_feature(node_list):
                text = ""
                for node in node_list:
                    text += node["name"]
                    text += ":"
                    text += node["description"]
                    text += "\n"
                return text
            node_texts_a_cluster = get_origin_feature(cluster)
            parent_feature = get_summarization(args, node_texts_a_cluster)
            match = re.search(r'\{.*?\}', parent_feature, re.DOTALL)
            if match:
                parent_feature = match.group(0)
            else:
                print(f"Parent Feature: {parent_feature}")
                raise ValueError("No valid JSON found in the summarization output")

            print(f"Parent Feature: {parent_feature}")
            parent_feature = json.loads(parent_feature)
            parent_feature["children"] = [node["name"] for node in cluster]
            print(f"Parent Feature: {parent_feature['name']}")
            print(f"Parent Feature Description: {parent_feature['description']}")
            next_level_nodes.append(parent_feature)
        layer_to_nodes[layer] = next_level_nodes
        current_level_nodes = next_level_nodes
        output_dir = f"{args.output_dir}/{args.embedding}_{args.cluster}_{args.choose_cluster_num}_{args.min_stop_cluster_num}_{args.max_cluster_num}_{args.summarization_model}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(f"{output_dir}/{layer}.json", 'w', encoding='utf-8') as f:
            json.dump(next_level_nodes, f, ensure_ascii=False, indent=4)
        print(f"Layer {layer} is constructed successfully!")

if __name__=="__main__":
    args = get_parser_args() 
    logger.info(f"going on {args.embedding}_{args.cluster}_{args.choose_cluster_num}_{args.summarization_model}")
    construct_tree(args)