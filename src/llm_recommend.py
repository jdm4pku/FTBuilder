import os
import json
from tqdm import tqdm
from openai import OpenAI

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_prompt(req,node):
    prompt = f"""
    You are selecting an reusable artifact for a requirement based on a requirement feature tree.
    Now, I will give you a requirement and a feature node of the requirement feature tree.
    Please judge whether the feature node is suitable for the requirement. If it is suitable, please return "yes", otherwise return "no".
    The requirement is: {req}
    The feature node is: {node}
    Answer in a single word: "yes" or "no".
    """
    return prompt

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


def find_with_construct_tree(req):
    ## DFS便利，大模型判断
    # 遍历最上层的feature
    layer1_feature = load_json("feature_tree/best_tree/2_layer.json")
    for lay1_feature_item in layer1_feature:
        name = lay1_feature_item["name"]
        description = lay1_feature_item["description"]
        feature_node = f"Name:{name} \t Description:{description}"
        prompt = get_prompt(req,feature_node)
        answer = get_completion(prompt)
        if answer == "no":
            continue
        ## 如果是yes，继续遍历它的children
        children = eval(lay1_feature_item["children"])
        layer2_feature = []
        all_layer2_feature = load_json("feature_tree/best_tree/1_layer.json")
        for item in all_layer2_feature:
            if item["name"] in children:
                layer2_feature.append(item)
        assert len(layer2_feature) == len(children)
        for layer2_feature_item in lay2_feature:
            name = layer2_feature_item["name"]
            description = layer2_feature_item["description"]
            feature_node = f"Name:{name} \t Description:{description}"
            prompt = get_prompt(req,feature_node)
            answer = get_completion(prompt)
            if answer == "no":
                continue
            ## 如果是yes，继续遍历它的children
            children = eval(lay2_feature_item["children"])
            layer3_feature = []
            all_layer3_feature = load_json("feature_tree/best_tree/0_layer.json")
            for item in all_layer3_feature:
                if item["name"] in children:
                    layer3_feature.append(item)
            assert len(layer3_feature) == len(children)
            for layer3_feature_item in layer3_feature:
                name = layer3_feature_item["name"]
                description = layer3_feature_item["description"]
                feature_node = f"Name:{name} \t Description:{description}"
                prompt = get_prompt(req,feature_node)
                answer = get_completion(prompt)
                if answer == "no":
                    continue
                return name
    return "not find"

def predict_with_construct_tree():
    artsel = load_json("dataset/artsel/dataset_1.json")
    predicted_artifact_list = []
    for req_item in tqdm(artsel, desc="Processing requirements", unit="requirement"):
        req = req_item["requirement"]
        artifact = req_item["artifact"]
        predict_artifact = find_with_construct_tree(req)
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
    with open(f"{output_dir}/with_construct_tree.json.json",'w',encoding='utf-8') as f:
        json.dump(predicted_artifact_list,f,indent=4)

if __name__=="__main__":  
    predict_with_construct_tree()

                     

            
            
            
            

    pass

if __name__=="__main":
    find_artifact_with_construct_tree()