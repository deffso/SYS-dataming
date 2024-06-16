import json
import json
from datasets import Dataset, DatasetDict
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
# 定义读取和解析JSON文件的函数
def read_and_parse_input_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line.strip()) for line in file]

    all_parsed_data = []

    for entry in data:
        doc_id = entry.get("new-ID", "")
        doc_text = entry.get("doc", "")
        events = entry.get("events", [])
        summarizations = entry.get("summarizations", [])

        for event, summarization in zip(events, summarizations):
            event_id = event.get("id", "")
            event_content = event.get("content", "")
            event_spans = event.get("spans", [])
            
            summary_id = summarization.get("id", "")
            event_summary = summarization.get("event-summarization", "")
            
            # 格式化为模型训练所需的结构
            parsed_data = {
                "doc_id": doc_id,
                "doc_text": doc_text,
                "event_id": event_id,
                "event_content": event_content,
                "event_spans": event_spans,
                "summary_id": summary_id,
                "event_summary": event_summary
            }

            all_parsed_data.append(parsed_data)

    return all_parsed_data




# 打印解析后的数据




def prepare_data(parsed_data):
    input_texts = []
    target_texts = []

    for item in parsed_data:
        input_text = f"document: {item['doc_text']} event: {item['event_content']}"
        target_text = item['event_summary']
        input_texts.append(input_text)
        target_texts.append(target_text)

    return input_texts, target_texts

file_path = 'round1_traning_data/train.json'

parsed_data = read_and_parse_input_data(file_path)
input_texts, target_texts = prepare_data(parsed_data)

# 创建数据集
dataset = Dataset.from_dict({
    "input_text": input_texts,
    "target_text": target_texts
})
#for item in parsed_data:
 #   print(json.dumps(item, ensure_ascii=False, indent=4))
# 划分训练集和验证集
dataset = dataset.train_test_split(test_size=0.1)

# 打印训练集中的数据
'''print("训练集:")
for example in dataset['train']:
    print(f"输入文本: {example['input_text']}")
    print(f"目标文本: {example['target_text']}\n")

# 打印测试集中的数据
print("测试集:")
for example in dataset['test']:
    print(f"输入文本: {example['input_text']}")
    print(f"目标文本: {example['target_text']}\n")'''




# 加载预训练的T5模型和分词器
#model_name = "t5-small"
#tokenizer = T5Tokenizer.from_pretrained(model_name)
#model = T5ForConditionalGeneration.from_pretrained(model_name)

# 加载预训练的Pegasus模型和分词器

model_name = 'google/pegasus-xsum'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)
# 数据预处理函数
def preprocess_function(examples):
    inputs = examples["input_text"]
    targets = examples["target_text"]
    model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)

    # 设置为文本目标
    labels = tokenizer(targets, max_length=128, padding="max_length", truncation=True).input_ids

    # 替换padding token id
    labels_with_padding = []
    for label in labels:
        label_with_padding = [token if token != tokenizer.pad_token_id else -100 for token in label]
        labels_with_padding.append(label_with_padding)

    model_inputs["labels"] = labels_with_padding

    return model_inputs
# 应用数据预处理

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["input_text", "target_text"])

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=100,
    weight_decay=0.01,
    save_total_limit=3,
)

# 创建Trainer对象
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

# 训练模型
trainer.train()
# 评估模型
eval_results = trainer.evaluate()

print(f"Perplexity: {eval_results['eval_loss']}")
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
