import json

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

if __name__=="__main__":
    file_path = "feature_tree/official_tree/lay_0.json"
    data = load_json(file_path)
    new_file = []
    for item in data:
        new_file.append(
            {
                "name":item["description"]
            }
        )
    with open(f"lay_0.json",'w',encoding='utf-8') as f:
        json.dump(new_file,f,indent=4)
