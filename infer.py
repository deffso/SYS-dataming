import json
import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertTokenizer, BertModel, pipeline

input_file = 'round1_traning_data/testa.json'
output_file = 'summarizations.json'

#with open(input_file, 'r', encoding='utf-8') as f:
 #   data = json.load(f)
def preprocess_text(text):
    # 简化和清理输入文本
    return text.replace(" ", "").replace("\n", "")

data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line.strip()))

'''model_name = 'google/pegasus-large'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)'''
model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
summarizations = []

for item in data:
    new_id = item['new-ID']
    doc = item['doc']
    events = item['events']
    
    for event in events:
        event_id = event['id']
        event_content = event['content']
        # 预处理输入内容
        #preprocessed_content = preprocess_text(event_content)
        preprocessed_content = event_content.replace(" ", "").replace("\n", "")
        # 调试信息：打印输入内容
        print(f"Processing event id: {event_id}")
        print(f"Event content: {event_content}")
        summary = summarizer(preprocessed_content, max_length=60, num_beams=5, early_stopping=True)
        
        #inputs = tokenizer(event_content, truncation=True, padding='longest', return_tensors='pt')
        #summary_ids = model.generate(inputs['input_ids'])
        #summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summary_text = summary[0]['summary_text']
        # 调试信息：打印生成的摘要
        print(f"Generated summary: {summary_text}")
        
        summarizations.append({
            "id": event_id,
            "event-summarization": summary_text
        })
        

output_data = {"summarizations": summarizations}

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print("摘要生成完成并已保存到:", output_file)
