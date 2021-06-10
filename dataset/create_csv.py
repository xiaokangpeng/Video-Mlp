'''
Creates the train and test csv dataset files
'''

import numpy as np
import pandas as pd
import csv
import os
import time

def get_vid(csvfile):
    df = pd.DataFrame()
    vid_list = []
    label_list = []
    #/home/share/GSAI_HD_lab/kinetics/kinetics400/compress/train_256_frame_v1
    #/home/u2020101278/xyx/data/KineticSound/frame/
    for line in open("/home/share/GSAI-M3PL-Lab/Kinetics-Sounds/split-4fps/val_4fps.txt"):
        vid, _, label = line[:-1].split(',')
        vid_list.append(vid)
        label_list.append(int(label))
    df['vid'] = vid_list
    df['label'] = label_list
    df.to_csv(csvfile)
    print(df)

get_vid('KS_val.csv')
    # for vid in os.listdir('/home/share/GSAI-M3PL-Lab/Kinetics-Sounds/ks-train-frame-1fps/ks-train-set'):
    #     label_list.append(vid)
    # print(label_list)
    # for i in range(len(label_list)):
    #     label2int[label_list[i]] = i
    # for label in label_list:
    #     for i, vid in enumerate(
    #             os.listdir('/home/share/GSAI-M3PL-Lab/Kinetics-Sounds/ks-train-frame-1fps/ks-train-set/' + label + '/')):
    #         for num in range(1, 11):
    #             if num == 10:
    #                 audio = label + '/' + vid + '/' + 'adf_000' + str(num) + '.pkl'
    #                 img = label + '/' + vid + '/' + 'img_000' + str(num) + '.jpg'
    #                 if os.path.exists(
    #                         '/home/share/GSAI-M3PL-Lab/Kinetics400/train-frame/' + audio) and os.path.exists(
    #                         '/home/share/GSAI_HD_lab/kinetics/kinetics400/compress/train_256_frame/' + img):
    #                     pos_vids.append(label + '/' + vid + '/')
    #                     neg_vids.append(label + '/' + vid + '/')
    #                     labels.append(label2int[label])
    #                 else:
    #                     print(label + '/' + vid + ' not exit')
    #                     break
    #             else:
    #                 audio = label + '/' + vid + '/' + 'adf_0000' + str(num) + '.pkl'
    #                 img = label + '/' + vid + '/' + 'img_0000' + str(num) + '.jpg'
    #                 if os.path.exists(
    #                         '/home/share/GSAI-M3PL-Lab/Kinetics400/train-frame/' + audio) and os.path.exists(
    #                         '/home/share/GSAI_HD_lab/kinetics/kinetics400/compress/train_256_frame/' + img):
    #                     continue
    #                 else:
    #                     print(label + '/' + vid + ' not exit')
    #                     break
    #
    #
    # # num_label = len(labels)
    # # for i in range(num_label):
    # #     random_id = np.random.randint(num_label)
    # #     while(i == random_id):
    # #         random_id = np.random.randint(num_label)
    # #     fake_vid = pos_vids[random_id]
    # #     pos_vid = pos_vids[i]
    # #     pos_vids.append(pos_vid)
    # #     neg_vids.append(fake_vid)
    # #     labels.append('0')
    #
    #
    # #     while (fake_vid == (label + '/' + vid + '/')):
    # #         random_id = np.random.randint(num_label)
    # #         fake_vid = pos_vids[random_id]
    # #
    # #
    # #
    # # for label in label_list:
    # #     for i, vid in enumerate(os.listdir('/home/share/GSAI_HD_lab/kinetics/kinetics400/compress/train_256_frame_v1/' + label + '/')):
    # #         random_id = np.random.randint(num_label)
    # #         fake_vid = pos_vids[random_id]
    # #         while (fake_vid == (label + '/' + vid + '/')):
    # #             random_id = np.random.randint(num_label)
    # #             fake_vid = pos_vids[random_id]
    # #         pos_vids.append(label + '/' + vid + '/')
    # #         neg_vids.append(fake_vid)
    # #         labels.append('0')
    #
    #
    # # data = [i, img, spec, str(1)]
    # # print(data)
    # # wr.writerow(data)
    # df['vid1'] = pos_vids
    # df['vid2'] = neg_vids
    # df['label'] = labels
    # df.to_csv(csvfile)

#
#
# def get_train_test(csvfile):
#     train_df = pd.DataFrame()
#     test_df = pd.DataFrame()
#
#     df = pd.read_csv(csvfile)
#     vid1s = df['vid1'].tolist()
#     vid2s = df['vid2'].tolist()
#     lbls = df['label'].tolist()
#
#     train_vid1s, test_vid1s,train_vid2s, test_vid2s, train_labels, test_labels = sklearn.model_selection.train_test_split(
#         vid1s, vid2s, lbls, test_size=0.9, random_state=int(time.time()), shuffle=True)
#
#     train_df['vid1'] = train_vid1s
#     train_df['vid2'] = train_vid2s
#     train_df['label'] = train_labels
#
#     test_df['vid1'] = test_vid1s
#     test_df['vid2'] = test_vid2s
#     test_df['label'] = test_labels
#
#     train_df.to_csv('./dataset/train_dataset_vid.csv')
#     test_df.to_csv('./dataset/test_dataset_vid.csv')
#
# def rebuild_dataset():
#     print()
#     print('Creating Dataset')
#     print()
#
#     exists = os.path.isfile('./dataset/dataset_vid.csv')
#     if exists:
#         print('Removing old csv file')
#         print()
#         os.rename('./dataset/dataset_vid.csv', './dataset/removed/dataset_vid.csv')
#
#     get_vid('./dataset/dataset_vid.csv')
#     get_train_test('./dataset/dataset_vid.csv')
# # print('build')
# rebuild_dataset()

# if __name__ == '__main__':
#
#     print()
#     print('Creating Dataset')
#     print()
#
#     exists = os.path.isfile('dataset.csv')
#     if exists:
#         print('Removing old csv file')
#         print()
#         os.rename('dataset.csv', './removed/dataset.csv')

