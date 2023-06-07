
from argparse import ArgumentParser

import numpy as np
from scipy.stats import permutation_test

from data import results_data, predefined_alternatives

parser = ArgumentParser(description='permutation test')
parser.add_argument('--permutation_type',type=str,default='independent',choices=['independent','samples','pairings'])
parser.add_argument('--alternative',type=str,default=None,choices=['less','greater','two-sided'])
parser.add_argument('--key_strings',type=str,help='[gender/religion/race]-[bert/gpt2]-[metric]-[method1]-[method2]')
args = parser.parse_args()

np.random.seed(42)
N_SAMPLE = 1000


keys = args.key_strings.split('-')
assert len(keys)==5
assert keys[0] in ['gender','religion','race']
assert keys[1] in ['bert','gpt2']
assert keys[2] in ['cp','ss','ppl','lms']
assert keys[3] in ['orig','finetune','prefix','prompt','adapter','sentdebias','selfdebias']
assert keys[4] in ['orig','finetune','prefix','prompt','adapter','sentdebias','selfdebias']
x = results_data[keys[0]][keys[1]][keys[3]][keys[2]]
y = results_data[keys[0]][keys[1]][keys[4]][keys[2]]


if args.key_strings in predefined_alternatives:
	alternative = predefined_alternatives[args.key_strings]
elif args.alternative is not None:
	print('key_strings not pre-defined, use user-specified alternative')
	alternative = args.alternative
else:
	print('Warning: no alternative is specified, using the default value "less"')
	alternative = 'less'

def statistic(x,y,axis):
	return np.mean(x,axis=axis)-np.mean(y,axis=axis)

res = permutation_test((x,y),statistic,permutation_type=args.permutation_type,
	vectorized=True,n_resamples=np.inf,alternative=alternative,axis=0)
print(args.key_strings)
print(f'{keys[2]}-{keys[3]} mean: {np.mean(x)}, std: {np.std(x)}')
print(f'{keys[2]}-{keys[4]} mean: {np.mean(y)}, std: {np.std(y)}')
print(f"statistic: {res.statistic}")
print(f"p value: {res.pvalue}")