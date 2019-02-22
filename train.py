# coding=utf-8
import numpy as np
# import attention_textCNN
import torch.optim as optim
from torch.autograd import Variable
import time
import cPickle
import os
import data2cv
import pcnn3
import torch.nn.functional
import process_dict

if __name__ == '__main__':
    gpu = 0


    torch.cuda.set_device(gpu)
    print 'ues GPU: ', gpu



    parameterlist = {}
    parameterlist['trainepoch'] = 20
    parameterlist['batch_size'] = 100
    parameterlist['max_sentence_word'] = 80
    parameterlist['wordvector_dim'] = 50
    parameterlist['PF_dim'] = 5
    parameterlist['relation_embedding_dim'] = 50
    parameterlist['filter_size'] = 3
    parameterlist['num_filter'] = 100
    parameterlist['classes'] = 58
    parameterlist['dict_threshold'] = 15
    parameterlist['Superbag_size'] = 2


    # log_path = './log/train%d.tlog' % gpu

    # save the model parameter
    if parameterlist['batch_size']%parameterlist['Superbag_size'] != 0:
        print "Super bag size error!"
        exit()

    while True:
        seed = int(1547269966)
        np.random.seed(seed)
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed(seed)  # gpu
        # torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False

        timenow = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        runpath = './model/'
        # mdir = os.mkdir(runpath)

        modelpath = runpath + 'PCNN_C2SA.pkl'

        traindatapath = './data'

        print 'model save in: ', modelpath

        # save_log = modelpath + '1.pkl\n'
        # log_file = open(log_path, 'a')
        # log_file.write(save_log)
        # log_file.close()
    

        print 'loading dataset.. '
        if not os.path.isfile(traindatapath+'/test_57w.p'):
            import dataset
            dataset.data2pickle(traindatapath + '/test_data.txt', traindatapath + '/test_57w.p')

        if not os.path.isfile(traindatapath + '/train_57w.p'):
            import dataset
            dataset.data2pickle(traindatapath + '/train_data.txt', traindatapath + '/train_57w.p')

        if not os.path.isfile(traindatapath + '/wv.p'):
            import dataset
            dataset.wv2pickle('./data/wv.txt', 50,'./data/wv.p')

        testData = cPickle.load(open(traindatapath+'/test_57w.p'))
        trainData = cPickle.load(open(traindatapath + '/train_57w.p'))
        # testData = testData[1:5]
        # trainData = trainData[1:15]
        tmp = traindatapath.split('_')

        test = data2cv.make_idx_data_cv(testData, parameterlist['filter_size'], int(parameterlist['max_sentence_word']))
        train = data2cv.make_idx_data_cv(trainData, parameterlist['filter_size'], int(parameterlist['max_sentence_word']))
        num_s = 0
        for bag in train:
            num_s += bag.num
        print 'training set sentence : %d' % num_s

        print 'finished. '

        print 'load Wv ...  '
        Wv = cPickle.load(open('./data/wv.p'))
        print 'finished.'

        data__ = process_dict.proceess_data(train + test, threshold=parameterlist['dict_threshold'])
        train = data__[:len(train)]
        test = data__[len(train):]
        # print Wv[1]
        # print Wv[160695]
        # print Wv[160696]
        rel_num_bag = [0 for i in range(parameterlist['classes'])]
        for i in range(len(train)):
            rel_num_bag[train[i].rel[0]] += 1


        rng = np.random.RandomState(3435)
        PF1 = np.asarray(rng.uniform(low=-1, high=1, size=[101, 5]))
        padPF1 = np.zeros((1, 5))
        PF1 = np.vstack((padPF1, PF1))
        PF2 = np.asarray(rng.uniform(low=-1, high=1, size=[101, 5]))
        padPF2 = np.zeros((1, 5))
        PF2 = np.vstack((padPF2, PF2))

        net = pcnn3.textPCNN(sequence_length=parameterlist['max_sentence_word'],
                                    num_sentences_classes=parameterlist['classes'],
                                    word_embedding_dim=parameterlist['wordvector_dim'],
                                    PF_embedding_dim=parameterlist['PF_dim'],
                                    filter_size=parameterlist['filter_size'],
                                    num_filters=parameterlist['num_filter'],
                                    word_embedding=Wv,
                                    PF1_embedding=PF1,
                                    PF2_embedding=PF2,
                                    RE_dim=parameterlist['relation_embedding_dim'],
                                    RelationEmbedding=None,
                                    Superbag_size=parameterlist['Superbag_size'],
                                    RelationMaxT=rel_num_bag).cuda()

        optimizer = optim.Adadelta(net.parameters(), lr=1.0, rho=0.95, eps=1e-06, weight_decay=0)

        for name, param in net.named_parameters():
           print(name, param.size())


        RelationMean = torch.zeros(parameterlist['classes'], parameterlist['num_filter']*3).cuda()


        epoch_now = 0
        batch_now = 0


        for epoch in range(parameterlist['trainepoch']):

            print 'epoch = %d , start.. ' % epoch_now
            shuffled_data = []
            shuffle_indices = np.random.permutation(np.arange(len(train)))
            for i in range(len(train)):
                shuffled_data.append(train[shuffle_indices[i]])

            SB_data = []
            tem_SB = {}
            for i in range(parameterlist['classes']):
                tem_SB[i] = []
            for i in range(len(train)):
                T_label = shuffled_data[i].rel[0]
                tem_SB[T_label].append(shuffled_data[i])
                if len(tem_SB[T_label]) == parameterlist['Superbag_size']:
                    for t_bag in tem_SB[T_label]:
                        if T_label == 0:
                            for ii in range(parameterlist['Superbag_size']):
                                SB_data.append(t_bag)
                            break
                        else:
                            SB_data.append(t_bag)
                    tem_SB[T_label] = []

            shuffled_data = SB_data
            print 'a epoch = %d batch' % (int(len(shuffled_data)) / int(parameterlist['batch_size']) + 1)

            bag_now = 0
            no_next = False
            loss_all = 0
            e_b = 0

            while True:
                num_all = 0
                num_att = 0

                next_batch_start = bag_now + parameterlist['batch_size']
                if next_batch_start < len(shuffled_data):
                    batch = shuffled_data[bag_now:next_batch_start]
                    bag_now = next_batch_start
                else:
                    batch = shuffled_data[bag_now:len(shuffled_data)]
                    no_next = True

                if no_next:
                    break

                out, labels, RelationMean = net(input_=batch,
                                  sentence_word=parameterlist['max_sentence_word'],
                                  word_embedding_dim=parameterlist['wordvector_dim'],
                                  PF_embedding_dim=parameterlist['PF_dim'],
                                  num_filters=parameterlist['num_filter'],
                                  RelationMean=RelationMean)

                labels_arr = labels
                labels = Variable(torch.LongTensor(labels).cuda())

                loss = torch.nn.functional.cross_entropy(out, labels, weight=None)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_now += 1

                loss_all += loss.data[0]
                e_b += 1

                if batch_now % 100 == 0:
                    print 'train batch = %d, epoch = %d, last batch loss = %.3f, gpu = %d' % \
                          (batch_now, epoch_now, loss.data[0], gpu)
                    _, predicted = torch.max(out.data, 1)

                if batch_now == 1:
                    print 'train epoch = %d, last batch loss = %.3f' % (epoch_now, loss.data[0])



            epoch_now += 1
            print 'train epoch = %d, loss = %.3f' % (epoch_now, loss_all/e_b)
            # if epoch_now == 1:
            #     out_f = open(modelpath + str(epoch_now) + '.seed', "w")
            #     out_f.write(str(seed))
            #     out_f.close()
            if epoch_now == 20:
                torch.save(net, modelpath)
        break
