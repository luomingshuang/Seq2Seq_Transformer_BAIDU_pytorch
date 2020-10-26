import os
import pickle

from tqdm import tqdm

from config import pickle_file, imgs_path, annotation_root, trn_txt, val_txt, tst_txt, imgs_padding
from utils import ensure_folder

import re
import glob

def get_data(split):
    print('getting {} data...'.format(split))

    global VOCAB
    all_samples = []
    if split == 'train':  
        with open(trn_txt, 'r') as f:            
            data = [line.strip() for line in f.readlines()]
    if split == 'val':  
        with open(val_txt, 'r') as f:            
            data = [line.strip() for line in f.readlines()]
    if split == 'tst':  
        with open(tst_txt, 'r') as f:            
            data = [line.strip() for line in f.readlines()]
    #print(data[:10])
    
    lines = []
    for line in data:
        with open(os.path.join(annotation_root, line), 'r') as f:
            annotation_lines = [line.strip() for line in f.readlines()]
            img_files = annotation_lines[2:]     
            if(len(img_files) <= imgs_padding): 
                lines.append(line)
    
    samples = []
    max_length = 0
    for line in lines:
        #print(line)
        part = re.split('[\/_\.]', line)
        #print(part)
        img_folder = os.path.join(imgs_path, '_'.join(part[2:6]), part[6][1:])
        with open(os.path.join(annotation_root, line), 'r') as f:
            annotation_lines = [line.strip() for line in f.readlines()]
            hanzi = annotation_lines[0]
            pinyins = annotation_lines[1].strip(' ').split(' ')
            for pinyin in pinyins:
                build_vocab(pinyin)
            trn = [VOCAB[c] for c in pinyins]
            img_files = annotation_lines[2:]
            #print(pinyins,img_folder,img_files)
            img_files = list(filter(lambda file: os.path.exists(os.path.join(img_folder, file)), img_files))
            #print(len(img_files))
            samples.append({'name':img_folder, 'imgs':img_files, 'trn':trn})
            if len(img_files) > max_length:
                max_length = len(img_files)
                #print(max_length)
    print(max_length)
        #images = [cv2.imread(os.path.join(img_folder, file)) for file in img_files]
        #images = [cv2.resize(img, (96, 96)) for img in images]
        #for i in range(75-len(images)):
        #    images.append(np.zeros((96, 96, 3), dtype=np.uint8))
        #print('every image shape is:', images[0].shape)
        #images = np.stack([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #              for img in images], axis=0) / 255.0             
    #all_samples = glob.glob(os.path.join(audio_path, '*.wav'))
    #print(wav_files)
    #print(wav_files[:10])
    
    print('split: {}, num_files: {}'.format(split, len(samples)))
    
    return samples
    
    #     txt = ' '.join(txt).upper()
    #     #trn = [letters.index('<sos>')]
    #     #print(txt.split(' '))
    #     #trn = list(txt) + ['<eos>']
    #     trn = txt.split(' ') + ['<eos>']
    #     #print(trn)
    #     for c in (trn):
    #         build_vocab(c)
    #     trn = [VOCAB[c] for c in trn]
    #     #print(trn)
    #     samples.append({'trn':trn, 'wave':wav_file})
    #     # print(trn)
    #     # print(text)
    #     # print(items)
    # print('split: {}, num_files: {}'.format(split, len(samples)))
    #print(samples)
    #return samples

def build_vocab(token):
    global VOCAB, IVOCAB
    if not token in VOCAB:
        next_index = len(VOCAB)
        VOCAB[token] = next_index
        IVOCAB[next_index] = token
    # with open(tran_file, 'r', encoding='utf-8') as file:
    #     lines = file.readlines()

if __name__ == "__main__":
    VOCAB = {'<sos>': 0, '<eos>': 1}
    IVOCAB = {0: '<sos>', 1: '<eos>'}
    data = dict()
    data['VOCAB'] = VOCAB
    data['IVOCAB'] = IVOCAB
    data['train'] = get_data('train')
    data['val'] = get_data('val')
    data['tst'] = get_data('tst')


    with open(pickle_file, 'wb') as file:
        pickle.dump(data, file)

    print('num_train: ' + str(len(data['train'])))
    print('num_val: ' + str(len(data['val'])))
    print('num_tst: ' + str(len(data['tst'])))
