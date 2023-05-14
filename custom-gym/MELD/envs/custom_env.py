import gym
import pickle
import os
import pandas as pd
import numpy as np
import scipy.io.wavfile as wavfile
from gym import spaces
from gensim.models import KeyedVectors
from gensim.downloader import load
from sklearn.metrics.pairwise import pairwise_distances

class MELD(gym.Env):
    def __init__(self, 
                 dataset_path:str, 
                 partition:str,
                 label_emb:str,
                 w2v_path='',
                 predefined_feat=True):
        '''Construct
        dataset_path: path to dataset 
        (for raw data, it should be path/to/MELD.Raw, and for predefined feature, it should be path/to/MELD.Features.Models/features) \t
        
        partition: 'train', 'test' or 'dev' \t
        
        label_emb: indicator of applied embedding space \t
        
        w2v_path (default:''): only activate when label_emb is 'w2v', path to w2v model. Default value will lead to downloading a new w2v model \t
        
        predefined_feat: bool value, used to indicate use predefined feature or raw data
        '''
        # set up paths
        self.dataset_path = dataset_path
        self.partition = partition
        if self.partition == 'dev': self.partition = 'val'
        assert self.partition in ['train', 'test', 'val']
        self.label_emb = label_emb
        self.labels = {}
        self.utt = {}
        self.audio = {}
        self.dia_id = -1
        self.utt_id = None
        self.utt_lst = []
        self.utt_ptr = -1
        self.dia_flag = False
        
                 
        # loading predefined features
        if predefined_feat:
            if self.partition == 'dev': self.partition = 'val'
            print("Loading states from predefined features ...")
            print('loading labels from %s/data_emotion.p,\n feature dim = 300' % self.dataset_path)            
            # loading emotion labels
            data_package = pickle.load(open('%s/data_emotion.p' % self.dataset_path, 'rb'))[0]
            # print(data_package)
            data_package = list(filter(lambda x: x['split']==self.partition, data_package))
            # print(data_package)
            for data in data_package:
                dia_idx = data['dialog']
                utt_idx = data['utterance']
                emo = data['y']
                if dia_idx in self.labels:
                    self.labels[dia_idx][utt_idx] = emo
                else:
                    self.labels[dia_idx] = {utt_idx:emo}
                    
            print('loading utterances from %s/text_glove_average_emotion.pkl,\n feature dim = 300' % self.dataset_path)
            # loading utt feat
            if self.partition == 'train':
                data_package, _, _ = pickle.load(open('%s/text_glove_average_emotion.pkl' % self.dataset_path, 'rb'))
            if self.partition == 'val':
                _, data_package, _ = pickle.load(open('%s/text_glove_average_emotion.pkl' % self.dataset_path, 'rb'))
            if self.partition == 'test':
                _, _, data_package = pickle.load(open('%s/text_glove_average_emotion.pkl' % self.dataset_path, 'rb'))
            for k, v in data_package.items():
                dia_idx, utt_idx = k.split('_')
                if dia_idx in self.utt:
                    self.utt[dia_idx][utt_idx] = v
                else:
                    self.utt[dia_idx] = {utt_idx:v}
                    
                    
            print('loading utterances from %s/audio_embeddings_feature_selection_emotion.pkl,\n feature dim = 1611' % self.dataset_path)
            # loading audio feat
            if self.partition == 'train':
                data_package, _, _ = pickle.load(open('%s/audio_embeddings_feature_selection_emotion.pkl' % self.dataset_path, 'rb'))
            if self.partition == 'val':
                _, data_package, _ = pickle.load(open('%s/audio_embeddings_feature_selection_emotion.pkl' % self.dataset_path, 'rb'))
            if self.partition == 'test':
                _, _, data_package = pickle.load(open('%s/audio_embeddings_feature_selection_emotion.pkl' % self.dataset_path, 'rb'))
            for k, v in data_package.items():
                dia_idx, utt_idx = k.split('_')
                if dia_idx in self.audio:
                    self.audio[dia_idx][utt_idx] = v
                else:
                    self.audio[dia_idx] = {utt_idx:v}
                    
        # loading raw data   
        else:
            print("Loading states from raw data ...")
            lst_path = '%s/%s_sent_emo.csv' % (self.dataset_path, self.partition)
            df = pd.read_csv(lst_path)[['Dialogue_ID', 'Utterance_ID', 'Emotion', 'Utterance']]
            for _, row in df.iterrows(): 
                dia_idx, utt_idx, emo, utt = list(row)
                wav = wavfile.read('%s/%s/dia%s_utt%s.wav' % 
                                   (self.dataset_path,
                                    self.partition,
                                    dia_idx,
                                    utt_idx))
                # loading labels, utterance & audio
                if dia_idx in self.labels:
                    self.labels[dia_idx][utt_idx] = emo
                    self.utt[dia_idx][utt_idx] = utt
                    self.audio[dia_idx][utt_idx] = wav
                else:
                    self.labels[dia_idx] = {utt_idx:emo}
                    self.utt[dia_idx] = utt           
                    self.audio[dia_idx] = wav     
        
        # loading dia info
        self.dia_total = len(self.labels)
        self.utt_total = np.sum([len(dia) for dia in self.labels])
        self.dia_utt = {}
        print('loading finished, totally loaded:\n dialogue: %i\nutterance: %i'%(self.dia_total, self.utt_total))
        for dia_idx in self.labels.keys():
            tmp = list(self.labels[str(dia_idx)].keys())
            tmp = [int(i) for i in tmp]
            tmp.sort()
            self.dia_utt[str(dia_idx)] = tmp
            
        
        self.observation_space = spaces.Dict({
            'utt_idx': spaces.Discrete(len(self.labels)),
            'audio_idx': spaces.Dict([(k, spaces.Discrete(len(v)))
                                      for k, v in self.dia_utt.items()]),
        })
        self.action_space = spaces.Discrete(7)
        
        # print('Action Space:', self.action_space)
        # print('State Space:', self.observation_space)
                
        # set up label embedding
        self.label_dict = {}
        label_k_lst = ['anger', 'joy', 'sadness', 'surprise', 'fear', 'disgust', 'neutral']
        print("Mapping original labels into embedding space:")
        print("Original Labels: ", label_k_lst)
        print("Emotion Embedding: %s"%self.label_emb, end='\t')
        
        
        if self.label_emb == 'w2v':
            # load w2v model
            if w2v_path == '':
                w2v_path = load('word2vec-google-news-300', True)
            w2v = KeyedVectors.load_word2vec_format(fname = w2v_path,
                                                    binary = True)
            for k in label_k_lst:
                self.label_dict[k] = np.tanh(w2v[k])
            del w2v

            print("with embedding dimension: 300")
        elif self.label_emb == 'glove':
            # load glove model
            if w2v_path == '':
                w2v_path = load('glove-twitter-50', True)
            glove = KeyedVectors.load_word2vec_format(fname = w2v_path)
            for k in label_k_lst:
                self.label_dict[k] = np.tanh(glove[k])
            del glove
            print("with embedding dimension: 50")
        elif self.label_emb == 'NRC_VAD':
            # (Valence, Arousal, Domain)
            self.label_dict={
                        'anger':np.tanh(np.array([.167, .865, .657])),
                        'joy':np.tanh(np.array([.98, .824, .794])),
                        'sadness':np.tanh(np.array([.052, .288, .164])),
                        'surprise':np.tanh(np.array([.857, .857, .562])),
                        'fear':np.tanh(np.array([.073, .84, .293])),
                        'disgust':np.tanh(np.array([.052, .775, .317])),
                        'neutral':np.tanh(np.array([.5, .5, .5]))
                                                        }
            print("with embedding dimension: 5")
        else:
            raise ValueError("given label embedding is not supported, now we only support labels within ['w2v', '2d_mapping', 'dic']")
        print("All label dictionary can be accessed by call 'label_emb' member\nFinish initialization.")
        
    def _reward(self, pred_emo, true_emo):
        '''return sum of pairwised cosine distance between
        prediction emotion embedding and real label'''
        return np.sum(pairwise_distances(pred_emo.reshape((1,-1)),
                                         true_emo.reshape((1,-1)),
                                         metric='cosine'))
        
    def step(self, action):
        '''return pair of
        
            (   (utterance / predefined utterance feature of next timestep,
            
                audio / predefined audio feature of next timestep),
                
                flage to indicate whether current dialogue has been finished,
                
                dialogue_utterance index pair,
                
                text emotion label,
                
                corresponding emotion embedding,
                
                reward of current action
            )
            '''
        
        # obtain reward of taking action on t-1
        self.utt_id = self.utt_lst[self.utt_ptr]
        true_y = self.labels[str(self.dia_id)][str(self.utt_id)]
        true_emb = self.label_dict[true_y]
        
        # obtain state of t
        self.utt_ptr += 1
        # print(self.utt_id, int(self.dia_utt[str(self.dia_id)]))
        if self.utt_ptr >= len(self.dia_utt[str(self.dia_id)]):
            self.dia_flag = True
            utt, audio = None, None
        else:
            utt = self.utt[str(self.dia_id)][str(self.utt_id)]
            audio = self.audio[str(self.dia_id)][str(self.utt_id)]

        return ((utt, audio), 
                self.dia_flag,
                '%i_%i'%(self.dia_id, self.utt_id),
                true_y,
                true_emb,
                self._reward(action, true_emb))
        
        
    def reset(self, utt_ptr=-1, dia_id=-1):
        '''
        utt_id, dia_id: used to indicate reseted state ("utt_id" th utterabce of "dia_id" th dialogue)
        '''
        if utt_ptr >= 0:
            self.utt_ptr = utt_ptr
        else:
            self.utt_ptr = 0
            
        if dia_id >= 0:
            self.dia_id = dia_id
        else:
            self.dia_id = np.random.randint(self.dia_total)
            while self.dia_id == 60:
                self.dia_id = np.random.randint(self.dia_total)
            
        # update dialogue depended states
        self.dia_flag = False
        self.utt_lst = self.dia_utt[str(self.dia_id)]
        self.utt_id = self.utt_lst[self.utt_ptr]


        utt = self.utt[str(self.dia_id)][str(self.utt_id)]
        audio = self.audio[str(self.dia_id)][str(self.utt_id)]

        
        return ((utt, audio),
                self.dia_flag,
                '%i_%s'%(self.dia_id, self.utt_id),
                None, None, None)

if __name__ == '__main__':
    
    env_test = MELD(
                dataset_path='E:/Code/datasets/MELD.Features.Models/features',
                partition='train',
                label_emb='w2v',
                predefined_feat=True)
    # env_test.seed(0)
    print(env_test.label_dict)
    # print(env_test.reset())
    # # print(env_test.labels['66'])
    # for _ in range(100):
    #     state, flag, id_pair, emo, emo_emb, reward = env_test.step(np.random.random((1,3)))
    #     # if flag == True:
    #     #     print(state)
    #     print(flag, id_pair)
    #     if flag == True:
    #         _, flag, id_pair, _, _, _ = env_test.reset()
    # env_test.close()