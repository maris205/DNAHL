continue pretrain by llama 7b

## step 1, build dna dict
build_sentence_dict.ipynb

## step 2, merge dict
merge_dna_eng_llama_dict.ipynb

## step 3, continue pretrain
run_pt.sh

## step 4, instruction finetune 
run_sft.sh

## step 5, test
llama_test2.ipynb

Due to the limited amount of fine-tuning training data, the model was trained for only one epoch, resulting in poor prediction accuracy. Further training is needed.
