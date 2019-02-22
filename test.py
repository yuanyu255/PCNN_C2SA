# coding=utf-8
import cPickle
import numpy as np
import random
# import attention_textCNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
import cPickle
import os
import data2cv
import torch.nn.functional
from sklearn.metrics import precision_recall_curve
import re
import datetime
from operator import itemgetter, attrgetter
import process_dict





def f_map(a, b):
    return [a, b]


if __name__ == '__main__':
    cuda = 0
    torch.cuda.set_device(cuda)

    modelpath = './model/PCNN_C2SA.pkl'
    # parameter_path = '/home/yyj/pytorchCNN/C2SA/model/2019_01_07_21_45_42_/2019_01_07_21_45_42.para'

    # parameterlist = cPickle.load(open(parameter_path, 'rb'))

    parameterlist = {}
    parameterlist['if_shuffle'] = True
    parameterlist['batch_size'] = 100
    parameterlist['max_sentence_word'] = 80
    parameterlist['wordvector_dim'] = 50
    parameterlist['PF_dim'] = 5
    parameterlist['filter_size'] = 3
    parameterlist['num_filter'] = 100
    parameterlist['classes'] = 58
    parameterlist['dict_threshold'] = 15


    traindatapath = './data'
    print 'use GPU :', cuda

    print 'loading dataset.. ',

    testData = cPickle.load(open(traindatapath + '/test_57w.p'))
    trainData = cPickle.load(open(traindatapath + '/train_57w.p'))


    test = data2cv.make_idx_data_cv(testData, parameterlist['filter_size'], int(parameterlist['max_sentence_word']))
    train = data2cv.make_idx_data_cv(trainData, parameterlist['filter_size'],
                                     int(parameterlist['max_sentence_word']))


    print 'finished. '
    print 'load Wv ...  '
    Wv = cPickle.load(open('./data/wv.p'))
    print 'finished.'

    data__ = process_dict.proceess_data(train + test, threshold=parameterlist['dict_threshold'])
    train = data__[:len(train)]
    test = data__[len(train):]
    rng = np.random.RandomState(3435)
    PF1 = np.asarray(rng.uniform(low=-1, high=1, size=[101, 5]))
    padPF1 = np.zeros((1, 5))
    PF1 = np.vstack((padPF1, PF1))
    PF2 = np.asarray(rng.uniform(low=-1, high=1, size=[101, 5]))
    padPF2 = np.zeros((1, 5))
    PF2 = np.vstack((padPF2, PF2))

    max_sc = 0
    maxeval_score = 0

    print 'eval :', modelpath
    while not os.path.isfile(modelpath):
        print 'sleep .... zzz .. zz . z .'
        time.sleep(60)
        # print modelpath, ' is not exist'
        # continue
    time.sleep(2)


    print 'load the net ...',

    net = torch.load(modelpath)
    net = net.cuda()


    print '  finished.'
    print 'use GPU :', cuda

    # torch.cuda.set_device(0)

    print 'a epoch = %d batch' % (int(len(test))/int(parameterlist['batch_size']) + 1)





    test_label_true = []
    test_label_prob = []

    np.random.seed(1234)
    epoch_now = 0
    batch_now = 0
    tp = 0
    fp = 0
    fn = 0


    shuffled_data = []
    shuffle_indices = np.random.permutation(np.arange(len(test)))
    for i in range(len(test)):
        shuffled_data.append(test[shuffle_indices[i]])
    bag_now = 0


    no_next = False
    while True:
        # print 0
        next_batch_start = bag_now + parameterlist['batch_size']
        if next_batch_start<len(test):
            batch = shuffled_data[bag_now:next_batch_start]
            bag_now = next_batch_start
        else:
            batch = shuffled_data[bag_now:len(test)]
            no_next = True

        labels = []
        for bag in batch:
            labels.append(bag.rel[0])

        labels_arr = labels
        labels = Variable(torch.LongTensor(labels).cuda())

        out = net(batch, parameterlist['max_sentence_word'], parameterlist['wordvector_dim'],
                  parameterlist['PF_dim'], parameterlist['num_filter'],
                  if_eval=True)

        output_norm = torch.nn.functional.softmax(out).cpu().data
        _, predicted = torch.max(out.data, 1)

        batch_now += 1
        t_1 = datetime.datetime.now()
        sentence_begin = 0
        for i in range(len(batch)):
            sentence_end = sentence_begin + batch[i].num
            prob = []

            t_3 = datetime.datetime.now()

            # 当输出为sentence时使用，求bag中的max
            for ii in range(52):
                # p = output_norm[sentence_begin:sentence_end, ii:ii + 1].max().data[0]
                p = output_norm[sentence_begin:sentence_end, ii:ii + 1].max()
                prob.append(p)

            t_4 = datetime.datetime.now()
            for j in range(52):
                if j != 0:
                    # print j, batch[i].rel[0]

                    if j == batch[i].rel[0]:
                        test_label_true.append(1)
                        test_label_prob.append(prob[j])
                    else:
                        test_label_true.append(0)
                        test_label_prob.append(prob[j])

            sentence_begin += batch[i].num

        t_2 = datetime.datetime.now()


        temst = 'batch = %-7d  cuda = %d' % (batch_now, cuda)
        if batch_now % 200 == 0 or batch_now == 1:
            print temst


        if no_next:
            break


    epoch_now += 1

    pred = map(f_map, test_label_true, test_label_prob)
    s_pred = sorted(pred, key=itemgetter(1), reverse=True)


    precision, recall, thresholds = precision_recall_curve(test_label_true, test_label_prob)

    F1max = 0
    for i in range(len(precision)):
        if (2*recall[i]*precision[i]/(recall[i]+precision[i])>F1max):
            F1max = 2*recall[i]*precision[i]/(recall[i]+precision[i])

    name = '%s_%.3f.PR' % (modelpath, F1max)

    outfile = open(name, 'w')



    for i in range(len(precision)):
        if recall[i] <= 0.6:
            if i < len(precision) - 1:
                tem = '%-15s   %-15s  %-15f\n' % (recall[i], precision[i], thresholds[i])
            else:
                tem = '%-15s   %-15s\n' % (recall[i], precision[i])
            outfile.write(tem)
    outfile.close()


    print "F1 = %.3f" % F1max


    
    
    
    
