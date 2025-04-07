import os
import json
import time
import openai
from openai import OpenAI
from tqdm import tqdm

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_json_files(file_path_list):
    layers = []
    for file_path in file_path_list:
        with open(file_path,'r',encoding='utf-8') as f:
            layers.append(json.load(f))
    return layers

def get_completion(prompt):
    client = OpenAI(
        api_key="sk-sR8RiK6YYrtk8Rss1b29047069804d108211285c7a25356c", # replace with your API key
        base_url="https://api.yesapikey.com/v1" # use the default API base URL
    )
    flag = False
    while not flag:
        try:
            response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-4o-2024-11-20",
                    temperature=0   
            )
            flag=True
        except Exception as e:
            print("Error in generating summarization")
            print("Retrying...")
            time.sleep(0.5)
    return response.choices[0].message.content

def deepseek_completion(prompt):
    client = OpenAI(api_key="sk-e721a556c4004086b97b1d67fee15d63", base_url="https://api.deepseek.com")
    flag = False
    while not flag:
        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt},
                    ],
                stream=False
            )
            flag=True
        except Exception as e:
            print(e)
            print("Retrying...")
            time.sleep(0.5)
    return response.choices[0].message.content

class TreeNode:
    def __init__(self,name):
        self.name = name
        self.children = []
    
    def add_child(self,child_node):
        self.children.append(child_node)

    def check_requirements(self,requirements):
        prompt = f"""
            You are selecting an reusable artifact for a requirement based on a requirement feature tree.
            Now, I will give you a requirement and a feature node of the requirement feature tree.
            Please judge whether the feature node is suitable for the requirement. If it is suitable, please return "yes", otherwise return "no".
            The requirement is: {requirements}
            The feature node is: {self.name}
            Answer in a single word: "yes" or "no".
        """
        answer = deepseek_completion(prompt)
        if answer=="yes":
            return True
        else:
            return False

    def __repr__(self):
        return self.name

class Tree:
    def __init__(self):
        self.root_nodes = []
        self.nodes = {}
    
    def get_or_create_node(self, name):
        if name not in self.nodes:
            self.nodes[name] = TreeNode(name)
        return self.nodes[name]
    
    def build_tree(self,layers):
        # create node
        for layer in layers:
            for item in layer:
                node_name = item['name']
                self.get_or_create_node(node_name)
        # create link
        for layer in layers[1:]:
            for item in layer:
                child_name = item["name"]
                parent_name = item["parent"]
                child_node = self.get_or_create_node(child_name)
                parent_node = self.get_or_create_node(parent_name)
                parent_node.add_child(child_node)
        # decide root
        all_children = set()
        for node in self.nodes.values():
            for child in node.children:
                all_children.add(child.name)
        self.root_nodes = [node for name, node in self.nodes.items() if name not in all_children]

    def dfs_find_matching_node(self,requirement,node=None):
        if node is None:
            for root in self.root_nodes:
                result = self.dfs_find_matching_node(requirement, root)
                if result:
                    return result
            return None
        if node.check_requirements(requirement):
            if not node.children:
                return node.name
        for child in sorted(node.children,key=lambda x:x.name):
            result = self.dfs_find_matching_node(requirement, child)
            if result:
                return result
        return None

def predict_with_offical_tree(tree):
    artsel = load_json("dataset/artsel/dataset_2.json")
    predicted_artifact_list = []
    for req_item in tqdm(artsel, desc="Selecting Artifacts", unit="requirement"):
        req = req_item["requirement"]
        artifact = req_item["artifact"]
        predict_artifact = tree.dfs_find_matching_node(req)
        predicted_artifact_list.append(
            {
                "requirement":req,
                "artifact":artifact,
                "predicted_artifact":predict_artifact
            }
        )
    output_dir = "prediction/artsel"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f"{output_dir}/with_offical_tree_deepseek.json",'w',encoding='utf-8') as f:
        json.dump(predicted_artifact_list,f,indent=4)

if __name__=="__main__":
    file_dir = "feature_tree/official_tree"
    file_name_list = ["lay_0.json","lay_1.json","lay_2.json","lay_3.json","lay_4.json"]
    file_path_list = []
    for file_name in file_name_list:
        file_path = os.path.join(file_dir,file_name)
        file_path_list.append(file_path)
    layers = load_json_files(file_path_list)
    tree = Tree()
    tree.build_tree(layers)

    start_time = time.time()
    predict_with_offical_tree(tree)
    end_time = time.time()
    execution_time = end_time - start_time  # 计算执行时间
    print(f"Function executed in {execution_time:.4f} seconds")
    
    