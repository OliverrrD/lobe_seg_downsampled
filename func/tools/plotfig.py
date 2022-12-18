import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn import metrics
from sklearn.metrics import confusion_matrix
#from graphviz import Digraph
import torch
from torch.autograd import Variable

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_testval(target, pred, save_path = None):
    class_names = ["negative", "positive"]
    cnf_matrix = confusion_matrix(target, pred)  ###
    print (cnf_matrix)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Confusion Matrix')
    if save_path != None:
        plt.savefig(save_path)
    plt.show()

def plot_roc(target, pred_prob):
    fpr, tpr, threshold = metrics.roc_curve(target, pred_prob)
    roc_auc = metrics.auc(fpr, tpr)
    print ('auc:',roc_auc)
    # ROC
    #plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def plot_roc_multi(targets, pred_probs, methods, title, save_path):
    assert len(targets) == len(pred_probs)
    plt.title(title)
    colors = ['-.g', '--c', 'b', 'deeppink', 'aqua', ]
    for i in range(len(targets)):
        if i == 0:
            fpr, tpr, threshold = metrics.roc_curve(targets[i], pred_probs[i])
            roc_auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, colors[i], label= methods[i] + ' (AUC %0.3f)' % roc_auc, linewidth = 2) #  
        else:
            fpr, tpr, threshold = metrics.roc_curve(targets[i], pred_probs[i])
            roc_auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, colors[i], label= methods[i] + ' (AUC %0.3f)' % roc_auc)
    plt.legend(loc='lower right',prop={'size': 13})
#    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.02, 1])
    plt.ylim([0, 1.05])
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.xlabel('False Positive Rate', fontsize=13)
    if len(save_path) > 0:
        plt.savefig(save_path)
    plt.show()

def visual_train(txt_train, txt_test, attr, view_range):
    '''
    exp: visual_train('/share5/gaor2/LungsProj/saved_file/baseline/visualization/training_stats.txt')
    '''
    f_tr = open(txt_train)
    lines_tr = f_tr.readlines()
    f_tr.close()
    f_tt = open(txt_test)
    lines_tt = f_tt.readlines()
    f_tt.close()
    #assert len(lines_tr) == len(lines_tt)
    epoch_list = []
    acc_tr, acc_tt = [], []
    f1_tr, f1_tt = [], []
    recall_tr, recall_tt = [],[]
    precision_tr, precision_tt = [],[]
    loss_tr, loss_tt = [],[]
    for i in range(len(lines_tr)):
        lne_tr = lines_tr[i]
        lne_tr_vec = lne_tr.split(",")
        for i in range(len(lne_tr_vec)):
            lne_tr_vec[i] = lne_tr_vec[i][lne_tr_vec[i].find("=") + 1:]
        if (lne_tr_vec[-1].endswith("\n")): lne_tr_vec[-1] = lne_tr_vec[-1][:-1]
        epoch_list.append(int(lne_tr_vec[0]))
        acc_tr.append(float(lne_tr_vec[1]))
        f1_tr.append(float(lne_tr_vec[2]))
        recall_tr.append(float(lne_tr_vec[3]))
        precision_tr.append(float(lne_tr_vec[4]))
        loss_tr.append(float(lne_tr_vec[5]))
    for i in range(len(lines_tt)):
        lne_tt = lines_tt[i]
        lne_tt_vec = lne_tt.split(",")
        for i in range(len(lne_tt_vec)):
            lne_tt_vec[i] = lne_tt_vec[i][lne_tt_vec[i].find("=") + 1:]
        if (lne_tt_vec[-1].endswith("\n")): lne_tt_vec[-1] = lne_tt_vec[-1][:-1]
        epoch_list.append(int(lne_tt_vec[0]))
        acc_tt.append(float(lne_tt_vec[1]))
        f1_tt.append(float(lne_tt_vec[2]))
        recall_tt.append(float(lne_tt_vec[3]))
        precision_tt.append(float(lne_tt_vec[4]))
        loss_tt.append(float(lne_tt_vec[5]))
    # attr_tr, attr_tt = [], []
    # a = [1, 2, 3]
    # exec('attr_tr' + '=' + 'a')
    # print(attr_tr)
    # exec('attr_tr' + '=' + attr + '_tr')
    # exec('attr_tt' + '=' + attr + '_tt')
    # print (attr_tr, '===========')
    #print (epoch_list)
    plt.plot(epoch_list[:view_range], loss_tr[:view_range], color='red', label='train')
    plt.plot(epoch_list[:view_range], loss_tt[:view_range], color='blue', label = 'val')
#     plt.xticks(np.arange(0, view_range, 10))
#     plt.yticks(np.arange(0.1, 1.01, 0.1))
#     plt.xlabel('Epoch')
#     plt.ylabel('Score')
#     plt.axis((0, view_range, 0, 1))
#     #plt.title(type_metric[0].upper() + type_metric[1:] + " Evaluation between Train and Validation")
    plt.legend(loc='lower right')
    plt.show()

def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    exp:
    x = Variable(torch.randn(1,22,224,224))#change 12 to the channel number of network input
    model = MyNet()
    y = model(x)
    g = make_dot(y)
    g.view()
    # need to sudo apt-get install graphviz
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d' % v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot
