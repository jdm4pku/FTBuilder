import json

def create_dataset(req_path,art_path,out_path):
    with open(req_path,'r',encoding='utf-8') as f_req, open(art_path,'r',encoding='utf-8') as f_art:
        requirements_lines = [line.strip() for line in f_req]
        artifacts_lines = [line.strip() for line in f_art]
    if len(requirements_lines) != len(artifacts_lines):
        raise ValueError("两个文件的行数不一致！")
    
    combined_data = []
    for req,art in zip(requirements_lines, artifacts_lines):
        combined_data.append({"requirement": req, "artifact": art})
    with open(out_path, 'w', encoding='utf-8') as f_out:
        json.dump(combined_data, f_out, ensure_ascii=False, indent=4)
    print(f"数据集已保存到 {out_path}。")

if __name__=="__main__":
    req_path = "dataset/artsel/user_requirements.txt"
    # art_path = "dataset/artsel/reused_artifacts_1.txt"
    # out_path = "dataset/artsel/dataset_1.json"
    art_path = "dataset/artsel/reused_artifacts_2.txt"
    out_path = "dataset/artsel/dataset_2.json"
    create_dataset(req_path,art_path,out_path)