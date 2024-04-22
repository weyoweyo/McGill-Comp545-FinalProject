from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer


def classify(text, tokenizer, inference_model):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    output = inference_model(**inputs)
    prediction = output.logits.argmax(dim=-1).item()
    return prediction


def run_test_cases(peft_model_name, tokenizer_name):
    label_map = {0: 'Dovish', 1: 'Hawkish', 2: 'Neutral'}
    # LOAD the Saved PEFT model
    inference_model = AutoPeftModelForSequenceClassification.from_pretrained(peft_model_name, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    print("Example 1")
    example1 = "The Committee anticipates that ongoing increases in the target range for the federal funds rate will be appropriate."
    print(example1)
    print("correct answer: Hawkish")
    prediction1 = classify(example1, tokenizer, inference_model)
    print(f'predicted classification: ', label_map[prediction1])
    print("=====================================================")

    print("Example 2")
    example2 = "In light of the softer economic outlook, we will maintain the current federal funds rate."
    print(example2)
    print("correct answer: Dovish")
    prediction2 = classify(example2, tokenizer, inference_model)
    print(f'predicted classification: ', label_map[prediction2])
    print("=====================================================")

    print("Example 3")
    example3 = "Economic activity has been rising at a strong rate."
    print(example3)
    print("correct answer: Neutral")
    prediction3 = classify(example3, tokenizer, inference_model)
    print(f'predicted classification: ', label_map[prediction3])
    print("=====================================================")

    print("Example 4")
    example4 = "Considering the elevated inflation pressures, we may consider further rate hikes."
    print(example4)
    print("correct answer: Hawkish")
    prediction4 = classify(example4, tokenizer, inference_model)
    print(f'predicted classification: ', label_map[prediction4])
    print("=====================================================")

    print("Example 5")
    example5 = "The labor market continues to strengthen, but inflation remains below our 2 percent longer-run objective."
    print(example5)
    print("correct answer: Dovish")
    prediction5 = classify(example5, tokenizer, inference_model)
    print(f'predicted classification: ', label_map[prediction5])
    print("=====================================================")
