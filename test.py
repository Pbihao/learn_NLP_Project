import pandas as pd
from main import *
test_data = pd.read_csv("./data/test1_data.csv", index_col=0)

test_data.sample(10)

import pandas as pd
from main import *


train_data = pd.read_csv('./data/train_data.csv', index_col=0)
label_denotation = {
    1: 'positive',
    0: 'negative'
}
train_data.sample(10)



from sklearn.model_selection import train_test_split

train, val = train_test_split(train_data, test_size=0.17, random_state=12)

databunch = TextClasDataBunch.from_df(".", train, val,
                  tokenizer=fastai_tokenizer,
                  vocab=fastai_bert_vocab,
                  include_bos=False,
                  include_eos=False,
                  text_cols="text",
                  label_cols='label',
                  bs=config.bs,
                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),
             )



from pytorch_pretrained_bert.modeling import BertConfig, BertForSequenceClassification
bert_model = BertForSequenceClassification.from_pretrained(config.bert_model_name, num_labels=2)

loss_func = nn.CrossEntropyLoss()

learner = Learner(databunch, bert_model, loss_func=loss_func, metrics=accuracy)
learner.load("./stage1")

texts = test_data.text
texts = list(texts)

test_label = []
for text in texts:
    out = learner.predict(text)[2]
    if out[0] > out[1]:
        test_label.append("0")
    else:
        test_label.append("1")

out_f = open("./data/answer.txt", "w")
for label in test_label:
    out_f.write(label)
    out_f.write("\n")
out_f.close()