# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
from torch.autograd import Variable

########### bag和SB都用distance ,pcnn3#####################
class textPCNN(torch.nn.Module):
    def __init__(self, sequence_length, num_sentences_classes,
                 word_embedding_dim, PF_embedding_dim,
                 filter_size, num_filters,
                 word_embedding, PF1_embedding, PF2_embedding,
                 RE_dim=0, RelationEmbedding=None, Superbag_size=2,
                 RelationMaxT=None):
        super(textPCNN, self).__init__()

        self.classes = num_sentences_classes

        self.conv = nn.Conv2d(1, num_filters, (filter_size, word_embedding_dim + 2 * PF_embedding_dim))
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)
        self.output = nn.Linear(num_filters * 3, num_sentences_classes)

        self.embedding_wv = nn.Embedding(sequence_length, word_embedding_dim)
        self.embedding_PF1 = nn.Embedding(PF_embedding_dim, word_embedding_dim)
        self.embedding_PF2 = nn.Embedding(PF_embedding_dim, word_embedding_dim)

        self.wordvec = Parameter(torch.FloatTensor(word_embedding))  
        self.PF1_embedding = Parameter(torch.FloatTensor(PF1_embedding))
        self.PF2_embedding = Parameter(torch.FloatTensor(PF2_embedding))


        self.Superbag_size = Superbag_size
        self.ATT = nn.Linear(num_filters * 3, num_filters * 3, bias=False)
        self.cosDistance = nn.CosineSimilarity(dim=1, eps=1e-08)
        self.RelationMean = Parameter(torch.randn(num_sentences_classes, num_filters*3))






    def forward(self, input_, sentence_word, word_embedding_dim, PF_embedding_dim, num_filters,
                bag_all=0, bag_att=0, if_eval=False, if_drop=False, drop_threshold=0, RelationMean=None):
        # if RelationMean is not None:
        #     RelationMean = Variable(RelationMean)



        labels = []
        sentence_label = []

        sentence_num = 0
        for bag in input_:
            sentence_num += bag.num

        self.embedding_wv.weight = self.wordvec
        self.embedding_PF1.weight = self.PF1_embedding
        self.embedding_PF2.weight = self.PF2_embedding

        # if if_share_weight:
        #     self.ATTweight = self.output.weight

        sentence_all = []
        sentence_PF1_all = []
        sentence_PF2_all = []

        entitypos1 = []
        entitypos2 = []
        num_sentence = 0
        for bag in input_:
            sentence_all += [sentence for sentence in bag.sentences]
            sentence_PF1_all += [sentence_PF[0] for sentence_PF in bag.positions]
            sentence_PF2_all += [sentence_PF[1] for sentence_PF in bag.positions]
            for i in range(bag.num):
                entitypos1_ = bag.entitiesPos[i][0]
                entitypos2_ = bag.entitiesPos[i][1]
                entitypos1.append(entitypos1_)
                entitypos2.append(entitypos2_)
                num_sentence += 1
                sentence_label.append(bag.rel[0])

        sentence_embedding = self.embedding_wv(Variable(torch.LongTensor(sentence_all).cuda()))
        sentence_PF1_enbedding = self.embedding_PF1(Variable(torch.LongTensor(sentence_PF1_all).cuda()))
        sentence_PF2_embedding = self.embedding_PF2(Variable(torch.LongTensor(sentence_PF2_all).cuda()))

        batch_input = torch.cat((sentence_embedding, sentence_PF1_enbedding, sentence_PF2_embedding), 2)
        batch_input = torch.unsqueeze(batch_input, 1)

        conv = self.conv(batch_input)
        # print conv.size()

        conv = self.tanh(conv)

        for i in range(num_sentence):
            pool1 = torch.nn.functional.max_pool2d(torch.unsqueeze(conv[i], 0)[:, :, :entitypos1[i] + 1],
                                                   (entitypos1[i] + 1, 1))
            
            pool2 = torch.nn.functional.max_pool2d(torch.unsqueeze(conv[i], 0)[:, :, entitypos1[i]:entitypos2[i] + 1],
                                                   (entitypos2[i] - entitypos1[i] + 1, 1))
            pool3 = torch.nn.functional.max_pool2d(torch.unsqueeze(conv[i], 0)[:, :, entitypos2[i]:],
                                                   (sentence_word - entitypos2[i], 1))

            pool1 = torch.squeeze(pool1, 2)
            pool1 = torch.squeeze(pool1, 2)
            pool2 = torch.squeeze(pool2, 2)
            pool2 = torch.squeeze(pool2, 2)
            pool3 = torch.squeeze(pool3, 2)
            pool3 = torch.squeeze(pool3, 2)

            pool_all = torch.cat((pool1, pool2, pool3), 0)
            # print pool_all.size()

            sentence_feature = torch.t(pool_all).clone().resize(1, 3 * num_filters)
            if i == 0:
                bag_sentence_feature = sentence_feature
            else:
                bag_sentence_feature = torch.cat((bag_sentence_feature, sentence_feature), 0)

        # conv = torch.squeeze(conv)

        # print bag_sentence_feature.size()

        scores = self.output(bag_sentence_feature)

        # print 'scores_out'
        # print scores_out

        if if_eval:
            return scores


        #### 使用 cos distance
        sentence_select_index = [i / self.classes for i in range(num_sentence*self.classes)]
        relation_select_index = [i % self.classes for i in range(num_sentence*self.classes)]

        sentence_vec = torch.index_select(bag_sentence_feature, 0, Variable(torch.LongTensor(sentence_select_index).cuda()))
        relation_vec = torch.index_select(self.RelationMean, 0, Variable(torch.LongTensor(relation_select_index).cuda()))

        sentence_distance = self.cosDistance(sentence_vec, relation_vec)
        sentence_distance = sentence_distance.resize(num_sentence, self.classes)
        att_norm1 = torch.nn.functional.softmax(sentence_distance, dim=1)
        ########################

        #####使用点乘#############

        # sentence_distance = torch.matmul(bag_sentence_feature, self.RelationMean.t())
        # att_norm1 = torch.nn.functional.softmax(sentence_distance, dim=1)
        # #########################




        sentence_begin = 0
        bag_now = 0

        # att_all_w = Variable(torch.zeros(1).cuda())


        for bag in input_:
            sentence_end = sentence_begin + bag.num
            rel = bag.rel[0]
            feature_weight = att_norm1[sentence_begin:sentence_end, rel:rel + 1]

            weight_all = 0.
            for sentence_idx in range(bag.num):
                weight_all += feature_weight[sentence_idx].data[0]

            if weight_all < 1e-3:
                feature_weight = Variable(torch.ones(bag.num, 1).cuda()) * (1. / bag.num)
            else:
                feature_weight = feature_weight / weight_all

            tem_feature = torch.matmul(bag_sentence_feature[sentence_begin:sentence_end].t(),
                                       feature_weight).t()

            if bag_now == 0:
                bag_feature = tem_feature
            else:
                bag_feature = torch.cat((bag_feature, tem_feature), 0)
            bag_now += 1
            sentence_begin = sentence_end
            labels.append(bag.rel[0])


        super_bag_labels = []

        for i_bag in range(len(bag_feature)):
            label = labels[i_bag]
            if i_bag % self.Superbag_size == 0:
                super_bag_labels.append(label)

        Re_m = torch.index_select(self.RelationMean, 0, Variable(torch.LongTensor(labels).cuda()))

        # xAr = self.Distance(bag_feature, Re_m)
        ## cos distance #############
        xAr = self.cosDistance(bag_feature, Re_m)
        xAr = xAr.resize(len(labels) / self.Superbag_size, self.Superbag_size)
        norm_att = torch.nn.functional.softmax(xAr)
        norm_att = norm_att.resize(len(labels), 1)
        ################################

        ######dot product #############
        # tem_sc = torch.matmul(bag_feature, Re_m.t())
        # t_eye = Variable(torch.eye(len(bag_feature)).cuda())
        # tem_sc = tem_sc*t_eye
        # xAr = torch.sum(tem_sc, 1)
        # xAr = xAr.resize(len(labels) / self.Superbag_size, self.Superbag_size)
        # norm_att = torch.nn.functional.softmax(xAr)
        # norm_att = norm_att.resize(len(labels), 1)
        ##############################



        # norm_att = Variable(torch.ones(len(labels), 1).cuda()) - norm_att
        Tem_SB_norm = bag_feature * norm_att
        Tem_SB_norm = Tem_SB_norm.resize(len(labels) / self.Superbag_size, self.Superbag_size, num_filters *3)
        SB_features = torch.sum(Tem_SB_norm, 1)

        output = self.output(self.dropout(SB_features))
        return output, super_bag_labels, RelationMean
