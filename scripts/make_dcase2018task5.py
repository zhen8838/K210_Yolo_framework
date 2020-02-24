import tensorflow as tf
from pathlib import Path
import librosa
from tqdm import tqdm
import os
import numpy as np
import multiprocessing as mp


def make_example(mel_str: str, label: list):
    """ make example """
    feature = {
        'mel_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mel_str])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def worker_cgf(ann_list, category: dict, root_path, NMEL, FRAME, HOP):
    results = []
    for path, label_str in tqdm(ann_list, total=len(ann_list)):
        y, sr = librosa.load(os.path.join(root_path, path), sr=None, mono=False)
        melspec = np.stack([librosa.power_to_db(librosa.feature.melspectrogram(
            y=np.asfortranarray(aud), sr=sr, n_mels=NMEL,
            n_fft=int(FRAME / 1000 * sr),
            hop_length=int(HOP / 1000 * sr))) for aud in y])
        results.append((melspec, category[label_str]))
    return results


def write_to_tfrecord(res, writer, is_train):
    for melspec, label_int in tqdm(res, total=len(res)):
        if is_train:
            for mel in melspec:
                writer.write(make_example(tf.io.serialize_tensor(mel).numpy(), label_int))
        else:
            writer.write(make_example(tf.io.serialize_tensor(melspec[0]).numpy(), label_int))


def main():
    n = 1  # NOTE 为了方便起见，这里默认选分割1
    ratio = 0.25  # 默认选取25%的样本做有标签训练，剩余标签作为无标签数据
    seed = 10101
    NMEL = 40
    FRAME = 64  # 都是ms
    HOP = 20  # 都是ms
    root_path = Path('/media/zqh/Datas/DCASE2018-task5-dev')
    record_path = Path('/home/zqh/workspace/DCASE2018-task5-mel-feature')
    save_dict = {}
    eval_txt = f'evaluation_setup/fold{n}_evaluate.txt'
    train_txt = f'evaluation_setup/fold{n}_train.txt'
    name = 'train'
    train_list = np.loadtxt(str(root_path / train_txt), 'str', delimiter='\t')
    category = {'absence': 0, 'cooking': 1, 'dishwashing': 2, 'eating': 3,
                'other': 4, 'social_activity': 5, 'vacuum_cleaner': 6, 'watching_tv': 7, 'working': 8}

    labeled_idx = []
    unlabeled_idx = []
    for label in category.keys():
        idx = np.where(train_list[:, 1] == label)[0]
        choice_idx = np.random.choice(np.arange(len(idx)), int(len(idx) * ratio), replace=False)
        labeled_idx.append(idx[choice_idx])
        idx = np.delete(idx, choice_idx)
        unlabeled_idx.append(idx)
    labeled_idx = np.concatenate(labeled_idx)
    unlabeled_idx = np.concatenate(unlabeled_idx)
    np.random.shuffle(labeled_idx)
    np.random.shuffle(unlabeled_idx)
    labeled_ann_list = train_list[labeled_idx][:, :2]
    unlabeled_ann_list = train_list[unlabeled_idx][:, :2]
    val_ann_list = np.loadtxt(str(root_path / eval_txt), 'str', delimiter='\t')

    for name, ann_list in [('train_labeled', labeled_ann_list),
                           ('train_unlabeled', unlabeled_ann_list),
                           ('val', val_ann_list)]:
        if 'train' in name:
            is_train = True
        else:
            is_train = False
        np.random.shuffle(ann_list)
        workers = 4
        pool = mp.Pool(workers)
        split = 3000
        sup_task_num = len(ann_list) // split + 1
        writer = tf.io.TFRecordWriter(str(record_path / f'{name}_{n}.tfrecords'))

        for i in range(sup_task_num):
            res = pool.apply_async(worker_cgf, args=(ann_list[i * split:(i + 1) * split],
                                                     category, str(root_path), NMEL, FRAME, HOP),
                                   callback=lambda x: write_to_tfrecord(x, writer, is_train))
        pool.close()
        pool.join()
        writer.close()
        save_dict[name + '_data'] = str(record_path / f'{name}_{n}.tfrecords')
        save_dict[name + '_num'] = len(ann_list) * 4 if is_train else len(ann_list)
    np.save('/home/zqh/Documents/K210_Yolo_framework/data/dcasetask5_ann_list.npy', save_dict)
