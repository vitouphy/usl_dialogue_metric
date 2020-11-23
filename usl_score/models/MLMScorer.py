import math
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer, BertConfig
from collections import namedtuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLMScorer(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        vocab_size = len(self.tokenizer)

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = torch.nn.Linear(768, vocab_size)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, mask_index=None):
        last_hidden_state, _= self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        # get the state corresponding to the mask
        selector = mask_index.view(-1, 1, 1).expand(last_hidden_state.size(0), 1, last_hidden_state.size(2))
        hidden_state = last_hidden_state.gather(1, selector).squeeze()
        output = self.linear(hidden_state)
        return output

    def predict(self, x):

        tokens = self.tokenizer.tokenize(x)
        sampling_length = min(len(tokens)+2, self.args.res_token_len)

        instance = self.tokenizer.encode_plus(
            x,
            add_special_tokens=True,
            max_length=self.args.res_token_len,
            pad_to_max_length=True,
            return_tensors="pt"
        )
        input_ids = instance['input_ids'].to(device)
        token_type_ids = instance['token_type_ids'].to(device)
        attention_mask = instance['attention_mask'].type(torch.FloatTensor).to(device)

        score = 0 # log-likelihood score
        for i in range(1, sampling_length-1): # ignore [CLS] and [SEP]
            mask_index = torch.LongTensor([i]).to(device)
            label = torch.LongTensor([ input_ids[0][mask_index].item() ]).squeeze()
            input_ids[0][mask_index] = 103

            # Apply to model
            output = self(input_ids, token_type_ids, attention_mask, mask_index)
            probabs = F.softmax(output, dim=-1)
            log_likeli = torch.log(probabs[label])
            score += log_likeli.item()

            # prepare the input_ids for another round
            input_ids[0][mask_index] = label

        # negative log-likelihood
        score = -score

        # TODO: add other metrics like perplexity,  nce
        # TODO: we might also need to do normalization

        return score

    def training_step(self, batch, batch_nb):
        input_ids, token_type_ids, attention_mask, mask_index, label = batch
        input_ids = input_ids.squeeze().to(device)
        token_type_ids = token_type_ids.squeeze().to(device)
        mask_index = mask_index.to(device)
        attention_mask = attention_mask.squeeze().type(torch.FloatTensor).to(device)
        label = label.squeeze()

        output = self(input_ids, token_type_ids, attention_mask, mask_index)
        loss = F.cross_entropy(output, label)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        output = self.training_step(batch, batch_nb)
        loss = output['loss']
        return {'val_loss': loss }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        print ("val_loss: ", avg_loss)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

if __name__ == "__main__":
    args = {
        "lr": 1e-5,
        "res_token_len": 25
    }
    args = namedtuple('args', args.keys())(*args.values())
    model = MLMScorer(args)
    score = model.predict("I want to go to school.ß")
    print (score)
