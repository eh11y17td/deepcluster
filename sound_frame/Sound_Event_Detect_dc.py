import dcase_util
from dcase_util.data import Sequencer
import random
from time import time
import numpy as np
import keras.backend as K
from keras.models import Model
from keras import callbacks
from sklearn.cluster import KMeans, MiniBatchKMeans
import pickle
import os
import subprocess
import tensorflow as tf
import pandas as pd
import re
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import argparse
from DatasetDcase2019Task4 import DatasetDcase2019Task4
from DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler
import config as cfg
from utils.Scaler import Scaler
from utils.utils import ManyHotEncoder, create_folder, SaveBest, to_cuda_if_available, weights_init, \
     AverageMeterSet, read_audio
from SED_transform import get_transforms
import torch
from tensorflow.contrib.tensorboard.plugins import projector
from tensorboardX import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# GPU使用量制限
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(graph=tf.get_default_graph(), config=config)
K.set_session(sess)

class Clustering_dc(object):
    def __init__(self, fs, ws, hs, n_mels, resume=None):
        self.fs = fs
        self.ws = ws
        self.hs = hs
        self.frame_length = hs
        self.n_mels = n_mels

        # self.basepath = '../../DCASE2019_task4/dataset/audio'
        # self.open_dir = "../../DCASE2019_task4/dataset/audio/train/weak"
        # self.open_dir = "/Work18/endohayato/models/research/audioset/dcase2018_baseline/task4/dataset/audio/train/weak"
        self.open_dir = "/Work20/endohayato/DCASE2019_task4/dataset/audio/train/weak"
        
        # if resume != None:
        #     self.save_dir = os.path.join("./weak2strong", resume, key_name).replace(".pth.tar", "").replace("exp/", "")
        #     self.save_mel_path = os.path.join("./save_mel", resume, key_name).replace(".pth.tar", "").replace("exp/", "")
        #     if not os.path.isdir('{}'.format(self.save_dir)):
        #         os.makedirs('{}'.format(self.save_dir))
        #     if not os.path.isdir('{}'.format(self.save_mel_path)):
        #         os.makedirs('{}'.format(self.save_mel_path))

        # else:
        #     subprocess.call(["rm {}/*".format(self.save_dir)], shell=True)

        self.audio_path = []
        self.dir_list = []
        self.df_list = []

        self.spe_frame = 20
        self.noise_frame = 4

        self.tag = {}
        self.strong_tag = {}

        self.list_dataset = []
        self.classes = cfg.classes

        self.new_df = pd.DataFrame(columns=["filename", "event_labels"]).set_index('filename')


    def set_audio_path(self, train=True):
        self.weak_path = os.path.join(cfg.workspace, cfg.weak.replace("metadata", "audio").split(".")[0])
        self.synthetic_path = os.path.join(cfg.workspace, cfg.synthetic.replace("metadata", "audio").split(".")[0])
        self.validation_path = os.path.join(cfg.workspace, cfg.validation.replace("metadata", "audio").rsplit("/", 1)[0])
        self.eval_desed_path = os.path.join(cfg.workspace, cfg.eval_desed.replace("metadata", "audio").split(".")[0])

        self.audio_path = [self.weak_path, self.synthetic_path, self.validation_path, self.eval_desed_path]
        self.dir_list = [os.listdir(i) for i in self.audio_path]
        self.set_df_list(train=train)
        # print(len(self.dir_list))
        # print(len(self.df_list))
        # exit()
    
    def set_df_list(self, train):
        dataset = DatasetDcase2019Task4(cfg.workspace,
                                        base_feature_dir=os.path.join(cfg.workspace, "dataset", "features"),
                                        save_log_feature=False)

        transforms = get_transforms(cfg.max_frames)

        weak_df = dataset.initialize_and_get_df(cfg.weak)
        load_weak = DataLoadDf(weak_df, dataset.get_feature_file, None, transform=transforms)
        if train ==True:
            self.list_dataset = [load_weak]

        else:
            synthetic_df = dataset.initialize_and_get_df(cfg.synthetic, download=False)
            synthetic_df.onset = synthetic_df.onset * cfg.sample_rate // cfg.hop_length
            synthetic_df.offset = synthetic_df.offset * cfg.sample_rate // cfg.hop_length

            validation_df = dataset.initialize_and_get_df(cfg.validation)
            validation_df.onset = validation_df.onset * cfg.sample_rate // cfg.hop_length
            validation_df.offset = validation_df.offset * cfg.sample_rate // cfg.hop_length

            eval_desed_df = dataset.initialize_and_get_df(cfg.eval_desed)
            eval_desed_df.onset = eval_desed_df.onset * cfg.sample_rate // cfg.hop_length
            eval_desed_df.offset = eval_desed_df.offset * cfg.sample_rate // cfg.hop_length

            # many_hot_encoder = ManyHotEncoder(classes, n_frames=cfg.max_frames // pooling_time_ratio)

            load_synthetic = DataLoadDf(synthetic_df, dataset.get_feature_file, None, transform=transforms)
            load_validation = DataLoadDf(validation_df, dataset.get_feature_file, None, transform=transforms)
            load_eval_desed = DataLoadDf(eval_desed_df, dataset.get_feature_file, None, transform=transforms)


            self.list_dataset = [load_weak, load_synthetic, load_validation, load_eval_desed]

        scaler = Scaler()
        scaler.calculate_scaler(ConcatDataset(self.list_dataset))

        transforms = get_transforms(cfg.max_frames, scaler)
        for i in range(len(self.list_dataset)):
            self.list_dataset[i].set_transform(transforms)
        print(self.list_dataset)

    def load_feature_name(self):
        # データディレクトリとそれに対応するデータフレームを処理
        for load_path, dr, load in zip(self.audio_path, self.dir_list, self.list_dataset):
            cnt = 0
            # ディレクトリ内のデータとデータフレーム内の情報を照合
            print("\n", load_path, "\n")
            print("dir:{}".format(len(dr)))
            print("df:{}".format(len(load)))
            seq_filenames = load.filenames.reset_index(drop=True)
            print(load.filenames.reset_index())
            # exit()

            for audio_name in dr:
                # print(load.df.query('filename == "{}"'.format(audio_name)))

                # print("ID:{}".format(seq_filenames[seq_filenames == "{}".format(audio_name)].index[0]))
                # print(load.filenames.query('filename == "{}"'.format(audio_name)))
                ID = seq_filenames[seq_filenames == "{}".format(audio_name)].index[0]
                # ID = seq_filenames[seq_filenames == "YyfSco9z1fpw_190.000_200.000.wav"].index[0]
                # (audio, fs) = sf.read(os.path.join(self.open_dir, audio_name))
                # save_audio = audio[round(1.2 * fs):round(3.4 * fs)]
                # print(len(save_audio) / np.array(fs), fs)
                # exit()
                # print(ID)
                # print(audio_name)  


                if os.path.basename(load_path) == "weak":
                    label = load.df.iloc[ID, -1].split(",")
                    # print(label)
                    if len(label) == 1:
                        # sample = load.df[load.df.filename.str.contains(audio_name)].iloc[ID]
                        sample = load[ID]
                        # print(len(sample[0]))
                        # exit()
                        # for i in sample:
                        #     print(len(i))
                        input_data = sample[0]
                        if label[0] not in self.tag:
                            self.tag.setdefault(label[0], {})
                            self.tag[label[0]].setdefault("data", input_data)
                            self.tag[label[0]].setdefault("filename", [audio_name])

                        else:
                            self.tag[label[0]]["data"] = np.concatenate([self.tag[label[0]]["data"], input_data])
                            self.tag[label[0]]["filename"].append(audio_name)
                        print('load {}:{}'.format(audio_name, cnt+1))
                        cnt += 1

                else:
                    sample = load[ID]
                    input_data = sample[0]

                    sub_df = load.df[load.df["filename"] == audio_name].reset_index(drop=True)
                    # print(sub_df)
                    for i in range(len(sub_df)):
                        for j in range(len(sub_df)):
                            if i == j:
                                continue

                            if int(sub_df.loc[i, "onset"]) > int(sub_df.loc[j, "onset"]):
                                if int(sub_df.loc[i, "onset"]) < int(sub_df.loc[j, "offset"]) and int(sub_df.loc[i, "offset"]) > int(sub_df.loc[j, "offset"]):
                                    swap = sub_df.loc[j, "offset"]
                                    sub_df.loc[j, "offset"] = sub_df.loc[i, "onset"]
                                    sub_df.loc[i, "onset"] = swap
                                elif int(sub_df.loc[i, "onset"]) < int(sub_df.loc[j, "offset"]) and int(sub_df.loc[i, "offset"]) < int(sub_df.loc[j, "offset"]):
                                    swap = sub_df.loc[j, "offset"]
                                    sub_df.loc[j, "offset"] = sub_df.loc[i, "onset"]
                                    sub_df.loc[i, "onset"] = sub_df.loc[i, "offset"]
                                    sub_df.loc[i, "offset"] = swap
                                    sub_df.loc[i, "event_label"] = sub_df.loc[j, "event_label"]
                                    
                            elif int(sub_df.loc[i, "onset"]) < int(sub_df.loc[j, "onset"]):
                                if int(sub_df.loc[i, "offset"]) < int(sub_df.loc[j, "offset"]) and int(sub_df.loc[i, "offset"]) > int(sub_df.loc[j, "onset"]):
                                    swap = sub_df.loc[j, "onset"]
                                    sub_df.loc[j, "onset"] = sub_df.loc[i, "offset"]
                                    sub_df.loc[i, "offset"] = swap
                                elif int(sub_df.loc[i, "onset"]) < int(sub_df.loc[j, "onset"]) and int(sub_df.loc[i, "offset"]) > int(sub_df.loc[j, "offset"]):
                                    swap = sub_df.loc[i, "offset"]
                                    sub_df.loc[i, "offset"] = sub_df.loc[j, "onset"]
                                    sub_df.loc[j, "onset"] = sub_df.loc[j, "offset"]
                                    sub_df.loc[j, "offset"] = swap
                                    sub_df.loc[j, "event_label"] = sub_df.loc[i, "event_label"]

                    # print(sub_df)
                    # print('load {}:{}'.format(audio_name, cnt+1))
                    for _, row in sub_df.iterrows():
                        if (pd.isnull(row["onset"]) == True) or (pd.isnull(row["offset"]) == True):
                            continue

                        if row["event_label"] not in self.strong_tag:
                            self.strong_tag.setdefault(row["event_label"], {})
                            self.strong_tag[row["event_label"]].setdefault("data", np.empty((0, self.n_mels), dtype=float))
                            self.strong_tag[row["event_label"]].setdefault("filename", [])
                            self.strong_tag[row["event_label"]].setdefault("num_frame", [])


                        if audio_name not in self.strong_tag[row["event_label"]]["filename"]:
                            self.strong_tag[row["event_label"]]["filename"].append(audio_name)

                        # sub_list = df[(df.iloc[:, 0].isin([audio_name])) & (df.iloc[:, -1].isin([label[0]]))]
                        sub_num_frame, sub_data = self.extract_strong_label(row, input_data)
                        self.strong_tag[row["event_label"]]["data"] = np.vstack((self.strong_tag[row["event_label"]]["data"], sub_data))
                        self.strong_tag[row["event_label"]]["num_frame"].extend(sub_num_frame)

                    cnt += 1



            if os.path.basename(load_path) == "weak":
                for i, item in self.tag.items():
                    print("\nweak_label:{}".format(i))
                    print("data:{}".format(len(item["data"])))
                    print("file:{}".format(len(item["filename"])))
                    # print("num_frame:{}".format(len(item["num_frame"])))

            else:
                for i, item in self.strong_tag.items():
                    print("\nstrong_label:{}".format(i))
                    print("data:{}".format(len(item["data"])))
                    print("file:{}".format(len(item["filename"])))
                    print("num_frame:{}".format(len(item["num_frame"])))
                    
        # exit()


    def extract_strong_label(self, row, sound_data):
        sub_num_frame = []
        sub_data = np.empty((0, self.n_mels), dtype=float)
        # st_frame = round(sub_list.iloc[i, 1] / self.frame_length).astype(int)
        # ed_frame = round(sub_list.iloc[i, 2] / self.frame_length).astype(int)
        # print("st_frame:{}, ed_frame:{}".format(row["onset"], row["offset"]))
        sub_num_frame.append(len(sound_data[int(row["onset"]):int(row["offset"])]))
        sub_data = np.vstack((sub_data, sound_data[int(row["onset"]):int(row["offset"])]))
        return sub_num_frame, sub_data

    def kmeans_clustering(self):
        n_clusters= 3
        for key, _ in self.tag.items():
            kmeans = KMeans(n_clusters=n_clusters, verbose=0, n_init = 50).fit(self.tag[key]["data"])
            # kmeans = KMeans(n_clusters=n_clusters, verbose=0, n_init = 30).fit(np.concatenate([self.tag[key]["data"], self.strong_tag[key]["data"]]))
            pred_weak = kmeans.predict(self.tag[key]["data"])
            pred_strong = kmeans.predict(self.strong_tag[key]["data"])

            past_num = 0
            self.tag[key].setdefault("weak_term_list", [])
            for i in self.tag[key]["filename"]:
                corr_num = past_num + cfg.max_frames
                self.tag[key]["weak_term_list"].append(pred_weak[past_num:corr_num])
                print(i)
                print(pred_weak[past_num:corr_num])
                past_num = corr_num

            print("pred_strong:{}".format(len(pred_strong)))
            count=0
            for j in self.strong_tag[key]["num_frame"]:
                count=count+j
            print("total_frame:{}".format(count))
            maximum = 0
            for number in range(n_clusters):
                cnt = 0
                ### term_listを使用しない場合 ###
                for i in range(len(pred_strong)):
                    if number == pred_strong[i]: cnt=cnt+1
                print("{}:{}".format(number, cnt))
                if cnt > maximum:
                    maximum = cnt
                    spe_number = number
            print("\nClustering:{}\n".format(key))
            print("Specified_number:{}\n".format(spe_number))

            self.tag[key].setdefault("clustering_data", pred_weak)
            self.tag[key].setdefault("spe_number", spe_number)
            print("weak_data:{}".format(len(pred_weak)))
            print("clustering_data:{}".format(len(self.tag[key]["clustering_data"])))
            print("spe_number:{}".format(self.tag[key]["spe_number"]))
            print("##### Kmeans Finish!!! #####")

            self.embedding_projecter(key, pred_weak)

        # exit()
    def embedding_projecter(self, key, pred_weak):
        LOG_DIR='logs'
        metadata = 'metadata_{}.tsv'.format(key)
        vector = tf.Variable(self.tag[key]["data"], name=key)

        with open(metadata, "w") as metadata_file:
            for row in pred_weak:
                metadata_file.write("{}\n".format(row))
        # writer = SummaryWriter()
        # labels = pred_weak
        with tf.Session() as sess:
            saver = tf.train.Saver([vector])
            sess.run(vector.initializer)
            saver.save(sess, os.path.join(LOG_DIR, "vector_{}.ckpt".format(key)))
            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = vector.name
            embedding.metadata_path = 'metadata_{}.tsv'.format(key)
            projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
        # writer.add_embedding(torch.FloatTensor(weights), metadata=labels)
        exit()

    def SED_main(self, key, count):
        for weak_term, audio_name in zip(self.tag[key]["weak_term_list"], self.tag[key]["filename"]):
            print("Processing_SED {}:{}".format(audio_name, count))
            start_point = None
            spe_cnt, noise_cnt = 0, 0
            sub_st2ed_list = []
            for ID, frame_data in enumerate(weak_term):
                if frame_data == self.tag[key]["spe_number"]:
                    if start_point == None:
                        start_point = ID
                        spe_cnt = spe_cnt+1
                    else:
                        spe_cnt = spe_cnt+1
                        if noise_cnt > 0:
                            spe_cnt = spe_cnt + noise_cnt
                            noise_cnt = 0
                if start_point != None and frame_data != self.tag[key]["spe_number"]:
                    noise_cnt = noise_cnt+1
                    if noise_cnt > self.noise_frame:
                        if spe_cnt >= self.spe_frame:
                            sub_st2ed_list.append([start_point, start_point+spe_cnt])
                        start_point = None
                        spe_cnt, noise_cnt = 0, 0
            if spe_cnt >= self.spe_frame:
                sub_st2ed_list.append([start_point, start_point+spe_cnt])
            # print(sub_st2ed_list)
            self.strong_tag[key]["st2ed_list"].append(sub_st2ed_list)
            count += 1
        return count

    def SED_multi(self):
        print("##### Sound Event Detect Start!!! #####")
        count = 1
        for key, value in self.tag.items():
            print("\nSED:{}\n".format(key))
            self.strong_tag[key].setdefault("st2ed_list", [])
            count = self.SED_main(key, count)
            # def SED_mainがあった
            print("filename:{}".format(len(self.tag[key]["filename"])))
            print("st2ed_list:{}".format(len(self.strong_tag[key]["st2ed_list"])))
        print("###### Sound Event Detect Finish!!! #####")
    
    def SED_Single(self, key, pred_weak, spe_number):
        past_num = 0
        # self.tag[key].setdefault("weak_term_list", [])
        # self.tag[key].setdefault("spe_number", spe_number)
        self.tag[key]["weak_term_list"] = []
        self.tag[key]["spe_number"] =  spe_number        

        for i in self.tag[key]["filename"]:
            corr_num = past_num + cfg.max_frames
            self.tag[key]["weak_term_list"].append(pred_weak[past_num:corr_num])
            # print(i)
            # print(pred_weak[past_num:corr_num])
            past_num = corr_num
        print("##### Sound Event Detect Start!!!(Single) #####")
        print("\nSED:{}\n".format(key))

        print("辞書：", len(self.tag[key]["weak_term_list"]))

        count=1
        # self.strong_tag[key].setdefault("st2ed_list", [])
        self.strong_tag[key]["st2ed_list"] = []

        count = self.SED_main(key, count) 
    
    def create_csv_wav_file_multi(self):
        print("\n##### Create File Start!!! #####")
        miss_cnt = 0
        self.new_df = pd.DataFrame(columns=["filename", "event_labels"]).set_index('filename')
        for key, value in self.tag.items():
            self.create_csv_wav_file_main(key, miss_cnt)
            # def create_csv_wav_file_mainがあった
        sub_df = pd.read_csv(os.path.join(cfg.workspace, cfg.weak), sep='\t')
        speech_df = sub_df[(sub_df.iloc[:, -1].str.contains("Speech")) & (~(sub_df.iloc[:, -1]=="Speech"))].sort_values("filename")
        speech_df = speech_df.set_index('filename')
        self.new_df = pd.concat([self.new_df, speech_df])
        for name in list(speech_df.index):
            subprocess.call(["cp", "{}/{}".format(self.open_dir, name), "{}/{}".format(self.save_dir, name)])
            print("##### Copy Weak Data #####")

        self.new_df = self.new_df.groupby(level=0).first()
        self.new_df.to_csv(self.save_dir.replace("audio", "metadata") + ".tsv", sep='\t')
        # new_df.to_csv('../dataset/metadata/train/weak2strong.tsv', sep='\t')
        print("\n##### All Complite #####\n")
        # exit()
    
    def create_csv_wav_file_main(self, key, miss_cnt):
        count = 1
        print("\nCreate_csv_wav_file:{}\n".format(key))
        for st2ed, audio_name in zip(self.strong_tag[key]["st2ed_list"], self.tag[key]["filename"]):
            print("Processing_Create {}:{}".format(audio_name, count))
            sox_cnt = 0
            basename = audio_name.rsplit("_", 2)[0]
            sb = subprocess.Popen(["sox", "--i", "-D", "{}/{}".format(self.open_dir, audio_name)], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
            time = float(sb[0].decode().strip("\n"))
            for section in st2ed:
                st_time = section[0]*self.frame_length
                ed_time = round(section[1]*self.frame_length, 1)
                savename = "_".join([basename, "{:.3f}".format(st_time), "{:.3f}".format(ed_time)]) + '.wav'
                if time > st_time and time >= ed_time:
                    subprocess.call(["sox", "{}/{}".format(self.open_dir, audio_name), "{}/{}".format(self.save_dir, savename), 
                                    "trim", "{:.3f}".format(st_time), "={:.3f}".format(ed_time)])
                    print("##### Create Patern1 #####")
                    print(savename)
                    sox_cnt = sox_cnt+1
                elif time > st_time and ed_time > time:
                    if (time-st_time) >= (self.frame_length*self.spe_frame):
                        subprocess.call(["sox", "{}/{}".format(self.open_dir, audio_name), "{}/{}".format(self.save_dir, savename), 
                                        "trim", "{:.3f}".format(st_time)])
                        print("##### Create Patern2 #####")
                        print(time, st_time, ed_time)
                        sox_cnt = sox_cnt+1
                    else:
                        print("##### miss #####")
                        miss_cnt = miss_cnt+1
                        print(time, st_time, ed_time)
                        continue                      
                else:
                    print("##### miss #####")
                    miss_cnt = miss_cnt+1
                    print(time, st_time, ed_time)
                    continue
                self.new_df.loc['{}'.format(savename)] = [key]
            count += 1

            # 元データのスペクトログラムを保存
            self.save_mel("{}/{}".format(self.open_dir, audio_name), audio_name, key, "./save_mel")
        
        print("\n")
        sub_df = pd.read_csv(os.path.join(cfg.workspace, cfg.weak), sep='\t')
        multi_df = sub_df[(sub_df.iloc[:, -1].str.contains(key)) & (~(sub_df.iloc[:, -1]==key))].sort_values("filename")
        multi_df = multi_df.set_index('filename')
        self.new_df = pd.concat([self.new_df, multi_df])
        for name in list(multi_df.index):
            subprocess.call(["cp", "{}/{}".format(self.open_dir, name), "{}/{}".format(self.save_dir, name)])
            print("##### Copy Weak Data #####")

    def create_csv_wav_Single(self, key, resume=None, select=False):
        print("\n##### Create File Start!!!(Single) #####")
        miss_cnt = 0
        # self.create_csv_wav_file_main(key, miss_cnt)
        self.sound_file(key, miss_cnt, resume=resume, select=select)

    def sound_file(self, key, miss_cnt, resume=None, select=False):
        count = 1
        print("\nCreate_csv_wav_file:{}\n".format(key))
        if resume != None:
            self.save_dir = os.path.join("./weak2strong", resume, key).replace(".pth.tar", "").replace("exp/", "")
            self.save_mel_path = os.path.join("./save_mel", resume, key).replace(".pth.tar", "").replace("exp/", "")
            if not os.path.isdir('{}'.format(self.save_dir)):
                os.makedirs('{}'.format(self.save_dir))
            if not os.path.isdir('{}'.format(self.save_mel_path)):
                os.makedirs('{}'.format(self.save_mel_path))
            subprocess.call(["rm {}/*".format(self.save_dir)], shell=True)
            subprocess.call(["rm {}/*".format(self.save_mel_path)], shell=True)
        
        if select == True:
            self.cp_csv_wav(key)

        else:
            for st2ed, audio_name in zip(self.strong_tag[key]["st2ed_list"], self.tag[key]["filename"]):
                print("Processing_Create {}:{}".format(audio_name, count))
                sox_cnt = 0
                basename = audio_name.rsplit("_", 2)[0]
                # (audio, fs) = sf.read(os.path.join(self.open_dir, audio_name))
                (audio, fs) = self.load_np(audio_name)
                time = len(audio) / fs
                save_audio = []
                
                for section in st2ed:
                    st_time = section[0]*self.frame_length
                    ed_time = round(section[1]*self.frame_length, 2)
                    # savename = "_".join([basename, "{:.3f}".format(st_time), "{:.3f}".format(ed_time)]) + '.wav'
                    if time > st_time and time >= ed_time:
                        term_audio = audio[round(st_time * fs):round(ed_time * fs)]
                        # sf.write("{}.wav".format(os.path.join(self.save_dir, savename)), term_audio, fs)
                        save_audio.extend(term_audio)
                        print("##### Create Patern1 #####")
                        print("{:.3f} {:.3f}".format(st_time, ed_time))
                        sox_cnt = sox_cnt+1
                    elif time > st_time and ed_time > time:
                        if (time-st_time) >= (self.frame_length*self.spe_frame):
                            term_audio = audio[round(st_time * fs):]
                            # sf.write("{}.wav".format(os.path.join(self.save_dir, savename)), term_audio, fs)
                            save_audio.extend(term_audio)
                            print("##### Create Patern2 #####")
                            print(time, st_time, ed_time)
                            sox_cnt = sox_cnt+1
                        else:
                            print("##### miss #####")
                            miss_cnt = miss_cnt+1
                            print(time, st_time, ed_time)
                            continue                      
                    else:
                        print("##### miss #####")
                        miss_cnt = miss_cnt+1
                        print(time, st_time, ed_time)
                        continue
                    # self.new_df.loc['{}'.format(savename)] = [key]

                if len(save_audio) != 0:
                    savename = "_".join([basename, "0.000", "{:.3f}".format(len(save_audio)/fs)]) + '.wav'
                    print(savename)
                    save_path = os.path.join(self.save_dir, savename)
                    self.new_df.loc['{}'.format(savename)] = [key]
                    count += 1
                    if not os.path.isfile("{}".format(save_path)):
                        sf.write("{}".format(save_path), save_audio, fs)
                        # 元データのスペクトログラムを保存
                        self.save_mel("{}/{}".format(self.open_dir, audio_name), audio_name, key, "./save_mel")
                        # 強ラベル化したデータのスペクトログラムを保存
                        self.save_mel(save_path, savename, key, self.save_mel_path)
                    else:
                        print("File already exists!!")

            print("\n")
            sub_df = pd.read_csv(os.path.join(cfg.workspace, cfg.weak), sep='\t')
            multi_df = sub_df[(sub_df.iloc[:, -1].str.contains(key)) & (~(sub_df.iloc[:, -1]==key))].sort_values("filename")
            multi_df = multi_df.set_index('filename')
            self.new_df = pd.concat([self.new_df, multi_df])
            for name in list(multi_df.index):
                subprocess.call(["cp", "{}/{}".format(self.open_dir, name), "{}/{}".format(self.save_dir, name)])
                print("##### Copy Weak Data #####")
            # exit()

    def cp_csv_wav(self, key):
        df = pd.read_csv(os.path.join(cfg.workspace, cfg.weak), sep='\t')
        key_df = df[df.iloc[:, -1].str.contains(key)].sort_values("filename")
        key_df = key_df.set_index('filename')        
        self.new_df = pd.concat([self.new_df, key_df])
        for name in list(key_df.index):
            subprocess.call(["cp", "{}/{}".format(self.open_dir, name), "{}/{}".format(self.save_dir, name)])
            print("##### Copy Weak Data (select) #####")

    def load_np(self, audio_name):
        npy_sound = "./npy_sound/{}".format(cfg.sample_rate)
        if not os.path.isdir(npy_sound):
            os.makedirs(npy_sound)
        npy_path = os.path.join(npy_sound, audio_name)
        if os.path.isfile(npy_path):
            load = np.load(npy_path)
            audio = load
            fs = cfg.sample_rate
        else:
            file_path = os.path.join(self.open_dir, audio_name)
            (audio, fs) = sf.read(file_path)
            np.save(npy_path, audio)

        return audio, fs


    def merge(self, key, base_dir, count, sub_df):
        print("\nMerge:{}\n".format(key))
        for audio_name in self.tag[key]["filename"]:
            basename = audio_name.rsplit("_", 2)[0]
            basepath = os.path.join(self.save_dir, basename + ".wav")
            comand_list_1 = ["sox"]
            for i in base_dir:
                if basename in i:
                    if basename not in self.merge_name:
                        self.merge_name.setdefault("{}".format(basename), [])
                    self.merge_name[basename].append(os.path.join(self.save_dir, i))

            if basename in self.merge_name:
                print("Merge {}:{}".format(basename, count))
                self.merge_name[basename] = sorted(self.merge_name[basename])
                # print(self.merge_name[basename])
                # print(count)
                count += 1
                comand_list_1.extend(self.merge_name[basename])
                comand_list_1.append("{}".format(basepath))
                subprocess.call(comand_list_1)
                for wav_path in comand_list_1[1:]:
                    for wav_name in base_dir:
                        if os.path.basename(wav_path) in wav_name:
                            subprocess.call(["rm {}".format(wav_path)], shell=True)
                            break
                sb = subprocess.Popen(["sox", "--i", "-D", "{}".format(basepath)], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
                time = float(sb[0].decode().strip("\n"))
                rename = "_".join([basename, "0.000", "{:.3f}".format(time)]) + ".wav"
                rename_path = os.path.join(self.save_dir, rename)
                os.rename(basepath, rename_path)
                self.new_df.loc['{}'.format(rename)] = [key]
            else:
                print("##### No sox #####")
            
            # マージしたデータのスペクトログラムを保存
            self.save_mel(rename_path, rename, key, self.save_mel_path)

        sub_df = pd.read_csv(os.path.join(cfg.workspace, cfg.weak), sep='\t')
        multi_df = sub_df[(sub_df.iloc[:, -1].str.contains(key)) & (~(sub_df.iloc[:, -1]==key))].sort_values("filename")
        multi_df = multi_df.set_index('filename')
        self.new_df = pd.concat([self.new_df, multi_df])
        for name in list(multi_df.index):
            if os.path.isfile("{}/{}".format(self.save_dir, name)) == False:
                subprocess.call(["cp", "{}/{}".format(self.open_dir, name), "{}/{}".format(self.save_dir, name)])
                print("##### Copy Weak Data #####")
            else:
                print("##### No Copy #####")

    
    def merge_main(self, key):
        print("\n##### Merge Start!!! #####")
        self.merge_name = {}
        count = 1
        base_dir = os.listdir(self.save_dir)
        self.new_df = pd.DataFrame(columns=["filename", "event_labels"]).set_index('filename')
        sub_df = pd.read_csv(os.path.join(cfg.workspace, cfg.weak), sep='\t')
        for key in self.tag.keys():
            self.merge(key, base_dir, count, sub_df)

        speech_df = sub_df[(sub_df.iloc[:, -1].str.contains("Speech")) & (~(sub_df.iloc[:, -1]=="Speech"))].sort_values("filename")
        speech_df = speech_df.set_index('filename')
        self.new_df = pd.concat([self.new_df, speech_df])
        for name in list(speech_df.index):
            if os.path.isfile("{}/{}".format(self.save_dir, name)) == False:
                print("##### Copy Weak Data #####")
                subprocess.call(["cp", "{}/{}".format(self.open_dir, name), "{}/{}".format(self.save_dir, name)])
            else:
                print("##### No Copy #####")
        self.new_df = self.new_df.groupby(level=0).first()
        self.new_df.to_csv(self.save_dir.replace("audio", "metadata") + ".tsv", sep='\t')
        # new_df.to_csv('./dataset/metadata/train/weak2strong.csv', sep='\t')

    def merge_single(self, key):    
        print("\n##### Merge Start!!! #####")
        self.merge_name = {}
        count = 1
        base_dir = os.listdir(self.save_dir)
        self.new_df = pd.DataFrame(columns=["filename", "event_labels"]).set_index('filename')
        sub_df = pd.read_csv(os.path.join(cfg.workspace, cfg.weak), sep='\t')
        self.merge(key, base_dir, count, sub_df)

    def save_mel(self, wav_path, basename, key_name, mel_path):
        if not os.path.isfile("{}/{}_{}.png".format(mel_path, os.path.basename(basename), key_name)):
            (audio, _) = read_audio(wav_path, cfg.sample_rate)
            ham_win = np.hamming(cfg.n_window)

            spec = librosa.stft(
                audio,
                n_fft=cfg.n_window,
                hop_length=cfg.hop_length,
                window=ham_win,
                center=True,
                pad_mode='reflect'
            )

            mel_spec = librosa.feature.melspectrogram(
                S=np.abs(spec),  # amplitude, for energy: spec**2 but don't forget to change amplitude_to_db.
                sr=cfg.sample_rate,
                n_mels=cfg.n_mels,
                fmin=cfg.f_min, fmax=cfg.f_max,
                htk=False, norm=None)

            mel_spec = librosa.amplitude_to_db(mel_spec)  # 10 * log10(S**2 / ref), ref default is 1
            # mel_spec = mel_spec.T
            mel_spec = mel_spec.astype(np.float32)

            librosa.display.specshow(mel_spec, x_axis='frames', y_axis='hz')
            plt.title("{}_{}".format(basename, key_name))
            cb = plt.colorbar()
            cb.set_label("db")
            plt.savefig("{}/{}_{}.png".format(mel_path, os.path.basename(basename), key_name))
            plt.close()
            print("save_mel:{}".format(basename))
            # exit()
    
    def save_csv(self):
        self.new_df = self.new_df.groupby(level=0).first()
        csv_dir = self.save_dir.replace(self.save_dir.split("/")[-1], "") + "metadata"
        if not os.path.isdir('{}'.format(csv_dir)):
            os.makedirs('{}'.format(csv_dir))
        self.new_df.to_csv(csv_dir + "/weak2strong_NN_sub.tsv", sep='\t')



def save_short_mel(wav_path, basename, key_name):
    with open('./{}_{}/features/{}.cpickle'.format(fs, n_mels, name.replace('.wav', '')), mode="rb") as fp:
        data = pickle.load(fp)
    db = librosa.amplitude_to_db(data["_data"], ref=np.max)
    librosa.display.specshow(db, x_axis='frames', y_axis='hz')
    plt.title("{}_{}".format(basename, key_name))
    cb = plt.colorbar()
    cb.set_label("db")
    plt.savefig("./save_log/{}_{}.png".format(os.path.basename(basename), key_name))
    plt.close() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-fs', '--fs', default='44100', help='Select fs', required=True, type=int)
    parser.add_argument('-n_mels', '--n_mels', default='64', help='Select n_mels', required=True, type=int)

    args = parser.parse_args()

    fs = args.fs
    n_mels = args.n_mels
    ws = 0.04
    hs = cfg.hop_length / cfg.sample_rate

    print(cfg.sample_rate, cfg.hop_length)

    clustering = Clustering_dc(fs, ws, hs, n_mels, "Dog", resume="./test")
    clustering.set_audio_path(train=False)
    clustering.load_feature_name()
    clustering.kmeans_clustering()
    clustering.SED()
    clustering.create_csv_wav_file()
    clustering.merge()