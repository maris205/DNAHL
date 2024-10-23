# DNAHL
DNAHL- DNA sequence and Human Language mixed large language model

There are already many DNA large language models, but most of them still follow traditional uses, such as extracting sequence features for classification tasks. More innovative applications of large language models, such as prompt engineering, RAG, and zero-shot or few-shot prediction, remain challenging for DNA-based models. The key issue lies in the fact that DNA models and human natural language models are entirely separate; however, techniques like prompt engineering require the use of natural language, thereby significantly limiting the application of DNA large language models. This paper introduces a hybrid model trained on the GPT-2 network, combining DNA sequences and English text to explore the potential of using prompts and fine-tuning in DNA models. The model has demonstrated its effectiveness in DNA related zero-shot prediction and multitask application.


# Model list
gpt2_small_128： gpt2 small, 128 max input tokens, Pre-training from scratch on DNA sequences and English text.

gpt2_small_1024： gpt2 small, 1024 max input tokens, Pre-training from scratch on DNA sequences and English text.

llama_7b: llama 7B, Continue pre-training on DNA sequences.




# References
model/dataset : https://huggingface.co/dnagpt

paper: https://arxiv.org/abs/2410.16917, DNAHL- DNA sequence and Human Language mixed large language model
