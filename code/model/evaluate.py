# Reference: https://github.com/gtfintechlab/fomc-hawkish-dovish/blob/main/code_model/bert_fine_tune_lm_hawkish_dovish_train_test.py

import torch
import evaluate


def evaluate_model(model, test_dataloader, device):

    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    model.eval()

    predictions_list = []
    labels_list = []

    for (input_ids, attention_masks, labels) in test_dataloader['test']:
        input_ids= input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        predictions_batch = predictions.cpu().numpy().tolist()
        labels_batch = labels.cpu().numpy().tolist()

        predictions_list.extend(predictions_batch)
        labels_list.extend(labels_batch)

        metric_acc.add_batch(predictions=predictions, references=labels)
        metric_f1.add_batch(predictions=predictions, references=labels)

    accuracy = metric_acc.compute()
    f1 = metric_f1.compute(average="weighted")
    print(f'Test accuracy: ', accuracy)
    print(f'Test f1: ', f1)

    return accuracy, f1, predictions_list, labels_list