The paper "Trillion Dollar Words: A New Financial Dataset, Task \& Market Analysis" presents an innovative approach to quantifying the impact of monetary policy on market dynamics through a nouvel dataset of Federal Open Market Committee (FOMC) communications. The authors use a unique approach by employing NLP techniques to categorize FOMC communications as hawkish, dovish, or neutral, rather than simply positive or negative. This categorization provides more nuanced insights into the influence of these communications on financial markets. This project is dedicated to replicating the paper's findings, with a specific focus on the task of classifying the stance of monetary policies, not just through conventional fine-tuning approaches but also through cutting-edge Parameter-Efficient Fine-Tuning (PEFT) methods. The results show that the PEFT techniques such as Infused Adapter by Inhibiting and Amplifying Inner Activations (IA3) and Low-Rank Adaptation (LoRA) have not only demonstrated a reduction in training time but have also achieved F1 scores similar with traditional fine-tuning methods. 

Drawing inspiration from the authors' codebase: https://github.com/gtfintechlab/fomc-hawkish-dovish and leveraging the data they provided, our project aims to replicate specific results from the original paper focusing on the task of classifying sentences from FOMC communications into hawkish, dovish, or neutral categories. We concentrated on a dataset consisting of meeting minutes, press conferences, and speeches, alongside a split version where sentences are segmented based on certain keywords. 

Due to constraints in training time and computational resources, the replication concentrated on four small models: FinBERT, FLANGBERT, FLANG-RoBERTa, and RoBERTa, optimizing them through a grid search to find the hyperparameter combination that yields the highest F1 score. Additionally, we expanded our examination to include PEFT methods by employing the PEFT library from Huggingface to fine-tune a larger model, RoBERTa-large. 

The goal of integrating these new fine-tuning strategies is to examine their impact on the accuracy and effectiveness of the classification task for monetary policy communications, thereby potentially offering improvements over the established methods. 

# Folders Description:
## code/model
This directory contains essential Python files for training and evaluating the language model.
#### evaluate.py: 
This script implements methods for evaluating the performance of the specified language model using the provided test data loader.
#### train.py: 
This script contains functions for fine-tuning the specified language model using the provided training data loader.

## code/utils
This directory houses essential utility files that support various functions such as loading models and tokenizers, preparing data loaders, plotting results, and more. These utilities facilitate efficient data handling and visualization throughout the model training and evaluation process.

## code/ChatGPT_API.ipynb 
This notebook uses the "gpt-3.5-turbo" model, configured to generate responses with a maximum of 1000 tokens and a temperature setting of 0.0. It demonstrates how to use the model in a zero-shot setting to perform the classification of FOMC communication stances. 

## code/Finetune_IA3.ipynb
This notebook implements the classification of FOMC communication stances by using the IA3 method to fine-tune the RoBERTa-large model.

## code/Finetune_LoRA.ipynb
This notebook implements the classification of FOMC communication stances by using the LoRA method to fine-tune the RoBERTa-large model.

## code/Finetune_P_Tuning.ipynb
This notebook implements the classification of FOMC communication stances by using the P-tuning method to fine-tune the RoBERTa-large model.

## code/Finetune_PrefixTuning.ipynb
This notebook implements the classification of FOMC communication stances by using the Prefix-tuning method to fine-tune the RoBERTa-large model.

## code/Finetune_PromptTuning.ipynb
This notebook implements the classification of FOMC communication stances by using the Prompt-tunikng method to fine-tune the RoBERTa-large model.

## code/ReproduceAuthorCode_FinetuneSmallModels.ipynb
This notebook focuses on replicating the classification of FOMC communication stances from the original paper. It uses traditional fine-tuning methods to fine-tune four small models: FinBERT, FLANGBERT, FLANG-RoBERTa, and RoBERTa. 

## data_PEFT
This folder contains the training and test datasets in CSV format, for training and evaluating models using both PEFT methods and traditional fine-tuning approaches. 

## grid_search_results_FinetuneSmallModels
This folder contains the results of the grid search hyperparameter optimization process for fine-tuning four small language models.