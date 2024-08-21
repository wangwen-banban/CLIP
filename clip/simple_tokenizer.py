import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re


@lru_cache() # 缓存返回值，即将输入和返回值进行哈希，然后缓存的方式是LRU（Least Recently Used）缓存，即当缓存满了之后，会删除最近最少使用的缓存
def default_bpe():
    # abspath是为了获得当前py文件的绝对路径，dirname是为了获得当前py文件的目录路径
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


# 返回一个集合，集合中包含了word中的所有相邻字符对, word是一个元组，元组中的元素是变长字符串
def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text) # 修复文件，将编码错误的文件修复
    text = html.unescape(html.unescape(text)) # 解码HTMl
    return text.strip() # 移除前后空格


# 去掉多个空格，只保留一个空格
def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode() # bytes to unicode
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()} # unicode to bytes
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n') # some bpe 每行两个字符串，用空格分隔
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges] # 转化为一个数组，每个元素是一个元组
        vocab = list(bytes_to_unicode().values()) # unicode字符集
        vocab = vocab + [v+'</w>' for v in vocab] # unicode添加</w>
        for merge in merges:
            vocab.append(''.join(merge)) # 将merges中的所有tuple合并添加到vocab中
        vocab.extend(['<|startoftext|>', '<|endoftext|>']) # 将这两个字符串逐个加入到vocab中
        self.encoder = dict(zip(vocab, range(len(vocab)))) # 将vocab中的字符和对应的索引组成一个字典
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges)))) # 将bpe的tuple和对应的索引组成一个字典
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        # 正则表达式，用于捕捉字符和数字以及一些特殊字符
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    # 输入的都是unicode字符
    def bpe(self, token):
        if token in self.cache: # 检查token是否为其中一个键
            return self.cache[token]
        # word是的元素是每一个字符，tuple的最后一个元素是</w>
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)
        if not pairs: # 如果pairs的tuple只有一个元素，则为空set
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf'))) # 找到pairs中对应的index最小的，返回的为tuple
            # 如果bigram不在bpe_ranks中，则退出循环
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            # word的大小为token的大小
            while i < len(word):
                # 先找到first，然后将i到first的位置直接合并到new_word中
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break
                
                # 如果下一个为second，则将first和second合并到new_word中
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower() # 得到一个干净的小写文本
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8')) # 将每个字母都转化为byte，然后进行映射unicode
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        # 返回一个列表，列表中的元素为bpe对应的下标
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text
