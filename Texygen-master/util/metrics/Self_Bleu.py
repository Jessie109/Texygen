import os, multiprocessing
from multiprocessing import Pool

import nltk
from nltk.translate.bleu_score import SmoothingFunction

from abc import abstractmethod


class Metrics(object):
    def __init__(self):
        self.name = 'Metric'

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    @abstractmethod
    def get_score(self):
        pass

class Self_Bleu(Metrics):
    def __init__(self, test_text='', gram=3):
        super(Self_Bleu, self).__init__()
        self.name = 'Self_Bleu'
        self.test_data = test_text
        self.real_data = ''
        self.gram = gram
        self.sample_size = 200
        self.reference = None
        self.is_first = True

    def get_name(self):
        return self.name

    def get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            return self.get_bleu_fast()
        return self.get_bleu_parallel()

    def get_reference(self):
        if self.reference is None:
            reference = list()
            with open(self.real_data) as real_data:
                for text in real_data:
                    text = nltk.word_tokenize(text)
                    reference.append(text)
            self.reference = reference
            return reference
        else:
            return self.reference

    def get_bleu(self):
        ngram = self.gram
        bleu = list()
        reference = list()
        tok_list = list()
        with open(self.test_data) as test_data:
            for text in test_data:
#                 print('text',text)
                tok_list.append(text)
#         print('len(tok_list):', len(tok_list))  
        lng = len(tok_list)
        weight = tuple((1. / ngram for _ in range(ngram)))
        for i in range(lng):
#             print('i',i)
            reference = tok_list[:-1]
            test_data = tok_list[-1]
            tok_list = tok_list[:-1]
           
            self.reference = reference

            
            for hypothesis in test_data:
                bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                                    smoothing_function=SmoothingFunction().method1))
        return sum(bleu) / len(bleu)

    def calc_bleu(self, reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def get_bleu_fast(self):
        reference = self.get_reference()
        # random.shuffle(reference)
        reference = reference[0:self.sample_size]
        return self.get_bleu_parallel(reference=reference)

    def get_bleu_parallel_self(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(multiprocessing.cpu_count())
        result = list()
        with open(self.test_data) as test_data:
            for hypothesis in test_data:
                hypothesis = nltk.word_tokenize(hypothesis)
                rf = [p for p in reference]
#                 print('len:',len(rf),'#',len(reference))
                rf.remove(hypothesis)
#                 print('lenA:',len(rf),'#',len(reference))
                result.append(pool.apply_async(self.calc_bleu, args=(rf, hypothesis, weight)))
        score = 0.0
        cnt = 0
        for i in result:
#             print('i:',i)
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt    
      
    def get_bleu_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(multiprocessing.cpu_count())
        result = list()
        with open(self.test_data) as test_data:
            for hypothesis in test_data:
                hypothesis = nltk.word_tokenize(hypothesis)
                print('lenrf',len(reference))
                result.append(pool.apply_async(self.calc_bleu, args=(reference, hypothesis, weight)))
        score = 0.0
        cnt = 0
        for i in result:
            print('i:',i)
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt