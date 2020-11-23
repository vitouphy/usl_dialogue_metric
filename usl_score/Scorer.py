from collections import namedtuple
from usl_score.models import NUPScorer
from usl_score.data_utils import encode_truncate

class Scorer:
    def __init__(self, args, metric='VUP'):
        self.args = args
        self.metric = type

        if metric == 'VUP':
            pass
        elif metric == 'NUP':
            self.scorer = NUPScorer(args)
            score = self.scorer.predict("hi", "how are you?")
            print (score)
        elif metric == 'MLM':
            pass

    def predict(self, res, ctx=None):
        if self.args == "NUP":
            pass


if __name__ == "__main__":

    args = {
        "lr": 1e-5,
        "maxlen": 25,
        "batch_size": 16,
        "max_epochs": 1,
        "num_workers": 1,
        "model": "MLM",
        "dropout": 0.2,
        "weight_decay": 1e-5,
        "ctx_token_len": 25,
        "res_token_len": 25
    }
    args = namedtuple('args', args.keys())(*args.values())
    scorer = Scorer(args, metric='NUP')
