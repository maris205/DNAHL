from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False) #use_regex=False,空格当成一般字符串

trainer = trainers.BpeTrainer(vocab_size=50000, special_tokens=["<|endoftext|>"]) #
tokenizer.train(["human2.fna.line.train","human2.fna.line.valid","wiki.train.raw","wiki.valid.raw"], trainer=trainer) #all file list

#test
encoding = tokenizer.encode("TGGCGTGAACCCGGGATCGGG")
print(encoding.tokens)

encoding = tokenizer.encode("helloworld")
print(encoding.tokens)

tokenizer.save("dna_eng_bpe_dict.json")