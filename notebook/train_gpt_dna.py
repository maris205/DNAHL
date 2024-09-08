from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import GPT2Tokenizer,GPT2Model,AutoModel

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset

from tokenizers import Tokenizer

#然后我们可以使用from_file() 方法从该文件里重新加载 Tokenizer 对象：
new_tokenizer = Tokenizer.from_file("dna_eng_bpe_dict.json")
#或者下面方法
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast(tokenizer_object=new_tokenizer)

#model = GPT2LMHeadModel.from_pretrained("gpt2")
config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=128,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = GPT2LMHeadModel(config)

train_file = "human2.fna.line.train3"
eval_file = "human2.fna.line.valid"
max_seq_length = 512
out_model_path = "gpt_dna_v0"
train_epoches = 10
batch_size = 20

tokenizer.pad_token = tokenizer.eos_token

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=max_seq_length,
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

eval_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=eval_file,
    block_size=max_seq_length,
)

training_args = TrainingArguments(
        output_dir=out_model_path,
        overwrite_output_dir=True,
        num_train_epochs=train_epoches,
        per_device_train_batch_size=batch_size,
        save_steps=2000,
        save_total_limit=2,
        prediction_loss_only=True,
        #fp16=True, v100没法用
    )


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(out_model_path)
