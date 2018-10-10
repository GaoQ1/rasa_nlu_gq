from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import jieba
import math
import random
import tensorflow as tf

def char_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(chars)  # 字典，包含每个字符及其出现的频率
    dico["<PAD>"] = 10000001  # 定义填充词
    dico['<UNK>'] = 10000000  # 定义未登录词
    char_to_id, id_to_char = create_mapping(dico)
    return dico, char_to_id, id_to_char


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    字典 字符:出现的频率
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = []
    for s in sentences:
        ts = []
        for char in s:
            tag = char[1]
            ts.append(tag)
        tags.append(ts)

    dico_tags = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico_tags)

    return tag_to_id, id_to_tag


def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False, train=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
          
    Args:
      sentences: 传入的句子（字符与对应的tag标记）
      char_to_id: 字符与位置的映射关系
      tag_to_id: tag标记与位置的映射关系

    Return:
      string: 训练数据的句子
      chars:  句子中每个字符在字典中的位置  
      segs:   jieba分词后句子每个词语的长度, 0 表示单个字 1表示词语的开头 2表示词语的中间词 3表示词语的结尾
      tags:   句子中对应的tag标记在字典中的位置
    """

    none_index = 0

    def f(x):
        return x.lower() if lower else x
    data = []
    for s in sentences:
        string = [w[0] for w in s]
        chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
                 for w in string]
        segs = get_seg_features("".join(string))
        if train:
            tags = [tag_to_id[w[1]] for w in s]
        else:
            tags = [none_index for _ in chars]
       
        data.append([string, chars, segs, tags])

    return data


def get_seg_features(string):
    """
    Segment text with jieba
    features are represented in bies format
    s donates single word
    将输入句子进行jieba分词，然后获取每个词的长度特征
    0 代表为单字，1代表词的开头，2代表词的中间部分，3代表词的结尾
    例如，string=高血糖和血压 高血糖=[1,2,3] 和=[0] 高血压=[1,3] seg_inputs=[1,2,3,0,1,3]
    """
    seg_feature = []

    for word in jieba.cut(string):
        if len(word) == 1:
            seg_feature.append(0)
        else:
            tmp = [2] * len(word)
            tmp[0] = 1
            tmp[-1] = 3
            seg_feature.extend(tmp)
    return seg_feature


class BatchManager(object):
    def __init__(self, data,  batch_size):
        # 排序并填充，使单个批次的每个样本保持长度一致，不同批次的长度不一定相同
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        # 按句子长度进行排序
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(
                sorted_data[i*int(batch_size): (i+1)*int(batch_size)]))
        return batch_data

    @staticmethod
    def pad_data(data):
        strings = []
        chars = []
        segs = []
        targets = []
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            string, char, seg, target = line
            padding = [0] * (max_length - len(string))
            strings.append(string + padding)
            chars.append(char + padding)
            segs.append(seg + padding)
            targets.append(target + padding)

        return [strings, chars, segs, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


def result_to_json(string, tags):
    item = {
        "string": string, 
        "entities": []
    }
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            item["entities"].append(
                {"value": char, "start": idx, "end": idx+1, "entity": tag[2:]})
        elif tag[0] == "B":
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "E":
            entity_name += char
            item["entities"].append(
                {"value": entity_name, "start": entity_start, "end": idx + 1, "entity": tag[2:]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags

def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def iob2(tags):
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def save_model(sess, model, checkpoint_path, logger):
    model.saver.save(sess, checkpoint_path)
    logger.warning("model saved")


def create_model(session, Model_class, config, logger):
    # create model, reuse parameters if exists
    model = Model_class(config)

    logger.warning("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
    return model


def input_from_line(line, char_to_id):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    将输入转化为 string, chars, segs, tags 四个特征
    """
    line = full_to_half(line)
    line = replace_html(line)
    inputs = list()
    inputs.append([line])
    line.replace(" ", "$")
    # 未登录词按<UNK>字符处理
    inputs.append([[char_to_id[char] if char in char_to_id else char_to_id["<UNK>"]
                    for char in line]])
    inputs.append([get_seg_features(line)])
    inputs.append([[]])
    return inputs


def full_to_half(s):
    """
    Convert full-width character to half-width one 
    将全角字符转换为半角字符
    """
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        char = chr(num)
        n.append(char)
    return ''.join(n)


def replace_html(s):
    s = s.replace('&quot;', '"')
    s = s.replace('&amp;', '&')
    s = s.replace('&lt;', '<')
    s = s.replace('&gt;', '>')
    s = s.replace('&nbsp;', ' ')
    s = s.replace("&ldquo;", "")
    s = s.replace("&rdquo;", "")
    s = s.replace("&mdash;", "")
    s = s.replace("\xa0", " ")
    return(s)
