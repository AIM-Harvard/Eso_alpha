from transformers import Trainer
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CELTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").to(device)
        # forward pass
        outputs = model(inputs['input_ids'])
        logits = outputs.get("logits").to(device)
        # compute custom loss (suppose one has 3 labels with different weights)
        # loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0])).to(device)
        loss_fct = nn.CrossEntropyLoss().to(device)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels).to(device), labels.long().view(-1).to(device))
        return (loss, outputs) if return_outputs else loss

class FLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        alpha = 0.25
        gamma = 1
        labels = inputs.get("labels").to(device)
        # forward pass
        outputs = model(inputs['input_ids'])
        logits = outputs.get("logits").to(device)
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss().to(device)
        ce = loss_fct(logits.view(-1, self.model.config.num_labels).to(device), labels.long().view(-1).to(device))
        pt = torch.exp(-ce)
        loss = (alpha * (1-pt)**gamma * ce).mean()
        return (loss, outputs) if return_outputs else loss

class BCETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").to(device) # batch[0, 1, 0, 1, 0, 0]
        # forward pass
        outputs = model(inputs['input_ids'])
        logits = outputs.get("logits").to(device)
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.BCEWithLogitsLoss().to(device)
        loss = loss_fct(logits.to(device), labels.float().to(device))
        return (loss, outputs) if return_outputs else loss