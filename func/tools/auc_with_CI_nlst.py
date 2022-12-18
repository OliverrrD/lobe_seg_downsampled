import pandas as pd
import numpy as np
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score, roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
#from func.tools.plotfig import plot_confusion_matrix, plot_roc, plot_testval, plot_roc_multi

#https://stackoverflow.com/questions/19124239/scikit-learn-roc-curve-with-confidence-intervals

def get_CI_bootstrap(y_true, y_pred):
    n_bootstraps = 100
    rng_seed = 100
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        fpr, tpr, thresholds = roc_curve(y_true[indices], y_pred[indices])
        #fprs[i] = fpr
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)

        aucs.append(roc_auc)
        #print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    return mean_auc, std_auc, mean_fpr, mean_tpr, tprs_lower, tprs_upper

def get_CI(y_trues, y_preds):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(len(y_trues)):
        fpr, tpr, thresholds = roc_curve(y_trues[i], y_preds[i])
        #fprs[i] = fpr
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)

        aucs.append(roc_auc)
        #print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    return mean_auc, std_auc, mean_fpr, mean_tpr, tprs_lower, tprs_upper

df = pd.read_csv('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/ThreeSet/label_noempty.csv')
df = df.loc[df['source'] == 'nlst']
df = df.loc[df['lastitem'] == 1]
df0 = df.loc[df['phase'] == 5]
df1 = df.loc[df['phase'] == 6]
df2 = df.loc[df['phase'] == 7]
df3 = df.loc[df['phase'] == 8]
df4 = df.loc[df['phase'] == 9]
cancer = 'kaggle_cancer'

kaggle_mean_auc, kaggle_std_auc, kaggle_mean_fpr, kaggle_mean_tpr, kaggle_tprs_lower, kaggle_tprs_upper = get_CI([df0['gt'], df1['gt'], df2['gt'], df3['gt'],df4['gt']], [df0[cancer], df1[cancer], df2[cancer], df3[cancer], df4[cancer]])

data_root = '/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/saved_file/ThreeSet/v7_rep/nlst/rnn/test_result'
df0 = pd.read_csv( data_root + '/epoch57_val0_CRNNClassifier.csv')
df1 = pd.read_csv(data_root + '/epoch66_val1_CRNNClassifier.csv')
df2 = pd.read_csv(data_root + '/epoch55_val2_CRNNClassifier.csv')
df3 = pd.read_csv(data_root + '/epoch85_val3_CRNNClassifier.csv')
df4 = pd.read_csv(data_root +  '/epoch68_val4_CRNNClassifier.csv')

rnn_mean_auc, rnn_std_auc, rnn_mean_fpr, rnn_mean_tpr, rnn_tprs_lower, rnn_tprs_upper = get_CI([df0['gt'], df1['gt'], df2['gt'], df3['gt'], df4['gt']], [df0['pred'], df1['pred'], df2['pred'], df3['pred'], df4['pred']])

data_root = '/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/saved_file/ThreeSet/v7_rep/nlst/cnn/test_result'
df0 = pd.read_csv( data_root + '/epoch93_val0_CNNNet.csv')
df1 = pd.read_csv(data_root + '/epoch96_val1_CNNNet.csv')
df2 = pd.read_csv(data_root + '/epoch67_val2_CNNNet.csv')
df3 = pd.read_csv(data_root + '/epoch37_val3_CNNNet.csv')
df4 = pd.read_csv(data_root +  '/epoch63_val4_CNNNet.csv')
# df0 = df0.loc[df0['islong'] == 1]
# df1 = df1.loc[df1['islong'] == 1]
# df2 = df2.loc[df2['islong'] == 1]
# df3 = df3.loc[df3['islong'] == 1]
# df4 = df4.loc[df4['islong'] == 1]

cnn_mean_auc, cnn_std_auc, cnn_mean_fpr, cnn_mean_tpr, cnn_tprs_lower, cnn_tprs_upper = get_CI([df0['gt'], df1['gt'], df2['gt'], df3['gt'], df4['gt']], [df0['pred'], df1['pred'], df2['pred'], df3['pred'], df4['pred']])

data_root = '/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/saved_file/ThreeSet/v7_rep/nlst/drnn/test_result'
df0 = pd.read_csv( data_root + '/epoch41_val0_DisCRNNClassifier.csv')
df1 = pd.read_csv(data_root + '/epoch42_val1_DisCRNNClassifier.csv')
df2 = pd.read_csv(data_root + '/epoch96_val2_DisCRNNClassifier.csv')
df3 = pd.read_csv(data_root + '/epoch89_val3_DisCRNNClassifier.csv')
df4 = pd.read_csv(data_root + '/epoch76_val4_DisCRNNClassifier.csv')

drnn_mean_auc, drnn_std_auc, drnn_mean_fpr, drnn_mean_tpr, drnn_tprs_lower, drnn_tprs_upper = get_CI([df0['gt'], df1['gt'], df2['gt'], df3['gt'], df4['gt']], [df0['pred'], df1['pred'], df2['pred'], df3['pred'], df4['pred']])

data_root = '/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/saved_file/ThreeSet/v7_rep/nlst/trnn/test_result'
df0 = pd.read_csv( data_root + '/epoch43_val0_CTLSTM.csv')
df1 = pd.read_csv(data_root + '/epoch77_val1_CTLSTM.csv')
df2 = pd.read_csv(data_root + '/epoch72_val2_CTLSTM.csv')
df3 = pd.read_csv(data_root + '/epoch38_val3_CTLSTM.csv')
df4 = pd.read_csv(data_root + '/epoch58_val4_CTLSTM.csv')


trnn_mean_auc, trnn_std_auc, trnn_mean_fpr, trnn_mean_tpr, trnn_tprs_lower, trnn_tprs_upper = get_CI([df0['gt'], df1['gt'], df2['gt'], df3['gt'], df4['gt']], [df0['pred'], df1['pred'], df2['pred'], df3['pred'], df4['pred']])




plt.plot(kaggle_mean_fpr, kaggle_mean_tpr, color='b',
          label=r'Ori CNN [3]: %0.2f $\pm$ %0.2f' % (74.18, 2.11),
         lw=1, alpha=.8)
plt.fill_between(kaggle_mean_fpr, kaggle_tprs_lower, kaggle_tprs_upper, color='b', alpha=.1)

plt.plot(cnn_mean_fpr, cnn_mean_tpr, color='deeppink',
          label=r'MC-CNN: %0.2f $\pm$ %0.2f' % (77.96, 10.98),
         lw=1, alpha=.8)
plt.fill_between(cnn_mean_fpr, cnn_tprs_lower, cnn_tprs_upper, color='deeppink', alpha=.1)

plt.plot(rnn_mean_fpr, rnn_mean_tpr, color='cornflowerblue',
          label=r'LSTM [5,6]: %0.2f $\pm$ %0.2f' % (80.84, 1.20),
         lw=1, alpha=.8)
plt.fill_between(rnn_mean_fpr, rnn_tprs_lower, rnn_tprs_upper, color='cornflowerblue', alpha=.1)

plt.plot(trnn_mean_fpr, trnn_mean_tpr, color='darkorange',
          label=r'tLSTM [11]: %0.2f $\pm$ %0.2f' % (80.80, 1.45),
         lw=1, alpha=.8)
plt.fill_between(trnn_mean_fpr, trnn_tprs_lower, trnn_tprs_upper, color='darkorange', alpha=.1)


plt.plot(drnn_mean_fpr, drnn_mean_tpr, color='aqua',
          label=r'DLSTM (ours): %0.2f $\pm$ %0.2f' % (82.55, 1.31),
         lw=1, alpha=.8)
plt.fill_between(drnn_mean_fpr, drnn_tprs_lower, drnn_tprs_upper, color='aqua', alpha=.1)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
#
#plt.savefig('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/saved_file/for_spore/for_spore_result_CI.eps')
plt.show()
