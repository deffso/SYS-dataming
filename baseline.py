import json
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# 检查并选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载预训练模型和分词器
model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

# 加载输入数据
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = [json.loads(line.strip()) for line in file]
        #print(data)
    return data

# 事件摘要生成函数
def generate_event_summary(event_content):
    inputs = tokenizer.encode("summarize: " + event_content, return_tensors="pt", max_length=512, truncation=True).to(device)
    summary_ids = model.generate(inputs, max_length=50, min_length=25, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# 处理单个实例
def process_instance(instance):
    doc = instance['doc']
    events = instance['events']
    summarizations = []

    for event in events:
        event_id = event['id']
        event_content = event['content']
        print(event_content)
        summary = generate_event_summary(event_content)
        print(summary)
        summarizations.append({"id": event_id, "event-summarization": summary})

    return {"new-ID": instance['new-ID'], "summarizations": summarizations}

# 处理数据集
def process_data(data):
    results = []
    for instance in data:
        print(data,"\n")
        result = process_instance(instance)
        results.append(result)
    return results

# 保存结果
def save_results(results, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + '\n')

# 主函数
def main():
    input_filename = 'round1_traning_data/testa2.json'
    output_filename = 'summarized_events.json'

    data = load_data(input_filename)
    results = process_data(data)
    #save_results(results, output_filename)

if __name__ == '__main__':
    main()
