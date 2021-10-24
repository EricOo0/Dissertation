import numpy as np
import torch
import transformers
from transformers import pipeline
MODEL_PATH = r"./test_classifify/"
# a.通过词典导入分词器
tokenizer = transformers.BertTokenizer.from_pretrained(r"./test_classifify/vocab.txt") 
# b. 导入配置文件
model_config = transformers.BertConfig.from_pretrained(MODEL_PATH)
# 修改配置
model_config.output_hidden_states = True
model_config.output_attentions = True
# 通过配置和路径导入模型
#model = transformers.BertModel.from_pretrained(MODEL_PATH,config = model_config)
model = transformers.BertForSequenceClassification.from_pretrained(MODEL_PATH,config = model_config)
model.eval()
#sentence="bad date!"
#nlp=pipeline('sentiment-analysis',model=MODEL_PATH,tokenizer=MODEL_PATH)
#print(nlp(sentence))
token_codes = tokenizer("it's good, i enjoy it'",return_tensors="pt")
with torch.no_grad():
    
       # outputs = model(input_ids=torch.tensor([token_codes['input_ids']]),token_type_ids = torch.tensor([token_codes['token_type_ids']])).detach().numpy()
        outputs = model(**token_codes).logits
outputs=torch.softmax(outputs,dim=1).tolist()
print(outputs)
print(np.argmax(outputs[0],axis=0))
