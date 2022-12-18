df0 = pd.read_csv('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/saved_file/ThreeSet/v4/proj/test_result_final/ori/step2.csv')
df1 = pd.read_csv('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/saved_file/ThreeSet/v4/proj/test_result_final/drnn/f1max/combine_step2.csv')

gt_list = df0['gt'].tolist()

assert gt_list == df1['gt'].tolist()

pred0_list = df0['pred'].tolist()
pred1_list = df1['pred'].tolist()

pred0_list = [int(i > 0.5) for i in pred0_list]
pred1_list = [int(i > 0.5) for i in pred1_list]


t0_0 = sum([int(pred0_list[i] == gt_list[i] and pred1_list[i] == gt_list[i]) for i in range(len(gt_list))])

t1_0 = sum([int(pred0_list[i] != gt_list[i] and pred1_list[i] == gt_list[i]) for i in range(len(gt_list))])

t0_1 = sum([int(pred0_list[i] == gt_list[i] and pred1_list[i] != gt_list[i]) for i in range(len(gt_list))])

t1_1 = sum([int(pred0_list[i] != gt_list[i] and pred1_list[i] != gt_list[i]) for i in range(len(gt_list))])

table = [[t0_0, t0_1], [t1_0, t1_1]]

from statsmodels.stats.contingency_tables import mcnemar

print (table)

result = mcnemar(table, exact=True)

print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))


# define contingency table
table = [[100, 8],[1, 3]]
# calculate mcnemar test
result = mcnemar(table, exact=True)
# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
	print('Same proportions of errors (fail to reject H0)')
else:
	print('Different proportions of errors (reject H0)')