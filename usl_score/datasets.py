from data_utils import *
from torch.utils.data import Dataset
from transformers import BertTokenizer

class VUPDataset(Dataset):
    def __init__(self, instances):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.instances = instances

    def __len__(self):
        return len(self.instances)

    def apply_negative_syntax(self, x):
        '''
            #0: Reorder
            #1: Word Drop
            #2: Word Repeat
        '''
        probabs = [0.3, 0.6, 1]
        thresh = random.random()
        for idx, val in enumerate(probabs):
            if thresh < val:
                break

        tokens = self.tokenizer.tokenize(x)
        if idx == 0:
            return apply_word_order(tokens)
        elif idx == 1:
            return apply_word_drop(tokens)
        else:
            return apply_word_repeat(tokens)

    def apply_positive_syntax(self, x):
        '''
            #0: Remove Puntuation at the end
            #1: Simplify Response
            #2: Remove Stopword
        '''
        probabs = [0.1, 0.2, 0.3, 1]
        thresh = random.random()
        for idx, val in enumerate(probabs):
            if thresh < val:
                break

        tokens = self.tokenizer.tokenize(x)
        if idx == 0:
            return apply_remove_puntuation(x)
        elif idx == 1:
            return apply_simplify_response(tokens)
        elif idx == 2:
            return apply_remove_stopwords(tokens)
        else:
            return x

    def __getitem__(self, index, maxlen=25):
        instance = self.instances[index]
        label = 1

        # apply negative syntactic
        tokens = self.tokenizer.tokenize(instance)
        if random.random() > 0.5 and len(tokens) >= 3:
            instance = self.apply_negative_syntax(instance)
            label = 0
        else:
            instance = self.apply_positive_syntax(instance)
            label = 1

        instance = self.tokenizer.encode_plus(instance,
                                         add_special_tokens=True,
                                         max_length=maxlen,
                                         pad_to_max_length=True,
                                         return_tensors="pt")
        input_ids = instance['input_ids']
        token_type_ids = instance['token_type_ids']
        attention_mask = instance['attention_mask']
        return input_ids, token_type_ids, attention_mask, label


if __name__ == "__main__":
    sents = [
        "i go to school.",
        "really? you don't like burger?"
    ]
    dataset = VUPDataset(sents)
    for x in dataset:
        print (x)
