# Reference: https://github.com/gtfintechlab/fomc-hawkish-dovish/blob/main/code_model/bert_fine_tune_lm_hawkish_dovish_train_test.py

import torch
import evaluate
from tqdm import tqdm

def training_loop(model, device, num_epochs, train_val_dataloader, optimizer, lr_scheduler):
    model.to(device)
    model.train()
    accuracy_list = []
    f1_list = []

    for epoch in range(num_epochs):
        for step, (input_ids, attention_masks, labels) in enumerate(tqdm(train_val_dataloader['train'])):

            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)

            outputs = model(input_ids = input_ids, attention_mask = attention_masks, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Training loss: {loss.item():.4f}')

        # evaluate on the validation dataset
        metric_acc = evaluate.load("accuracy")
        metric_f1 = evaluate.load("f1")
        model.eval()
        for step, (input_ids, attention_masks, labels) in enumerate(tqdm(train_val_dataloader['val'])):
            input_ids= input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric_acc.add_batch(predictions=predictions, references=labels)
            metric_f1.add_batch(predictions=predictions, references=labels)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation accuracy:')
        accuracy = metric_acc.compute()
        accuracy_list.append(accuracy)
        print(accuracy)

        f1 = metric_f1.compute(average="weighted")
        f1_list.append(f1)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation f1:')
        print(f1)
        print('---------------------------------------------------')

    return accuracy_list, f1_list