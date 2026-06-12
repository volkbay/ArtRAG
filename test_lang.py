
import sys
import os
sys.path.append(os.path.abspath('.'))

from artrag.huggingface_eval import evaluate_batch
from pprint import PrettyPrinter
pprint = PrettyPrinter().pprint

predicts = ['i am a boy', 'she is a girl']
answers = [['am i a boy ?'], ['is she a girl ?']]

results = evaluate_batch(predicts, answers)
pprint(results)
# Example output (metrics depend on installed metric implementations):
# {'Bleu_1': 0.5, 'Bleu_2': 0.0, 'ROUGE_1': 0.75, 'ROUGE_L': 0.75, 'METEOR': 0.4}