class CorpusProcess(object):
    def __init__(self,root = None, corp_path = None, precessed_path = None):
        """初始化"""
        self.corpus_path = root + corp_path # 语料路径
        self.process_corpus_path = root + precessed_path # 预处理后的语料路径
        self._maps = {u'CONT': u'CONT', 
                      u'EDU': u'EDU', 
                      u'LOC': u'LOC', 
                      u'NAME': u'NAME',
                      u'ORG': u'ORG',
                      u'PRO': u'PRO',
                      u'RACE': u'RACE',
                      u'TITLE': u'TITLE'
                      }
        self.iswrite = True
        self.__pre_process()
        self.__initialize()

    def __read_corpus_from_file(self, file_path):
        """读取语料"""
        f = open(file_path, 'r', encoding='utf-8')
        lines = f.readlines() # 读取所有行
        f.close()
        return lines

    def __write_corpus_to_file(self, data, file_path):
        """写语料"""
        f = open(file_path, 'wb')
        f.write(data)
        f.close()

    def __pre_process(self):
        """语料预处理 """
        lines = self.__read_corpus_from_file(self.corpus_path)
        new_lines = []
        new_words = ''
        for line in lines:
            word = line.strip().split(' ')
            if word == ['']: 
                new_lines.append(new_words)
                new_words = ''
                continue
            elif word[1] == 'O': # 如果标签为O，直接添加
                new_words += word[0] + '/' + word[1] + '  '
            else:
                word.extend(word.pop(1).split('-'))
                if word[1] == 'B':
                    group = []
                    group.append(word)
                elif word[1] == 'M':
                    group.append(word)
                elif word[1] == 'E':
                    group.append(word)
                    new_words += self.__group_up(group)
        if self.iswrite:
            self.__write_corpus_to_file(data='\n'.join(new_lines).encode('utf-8'), file_path=self.process_corpus_path)

    def __group_up(self, group):
        """合并分词组 """
        new_word = ''
        new_words = ''
        for word in group:
            new_word += word[0]
        new_words = new_word + '/' + group[0][2] + '  '
        return new_words

    def __tag_perform(self, tag, index, sentence_length):
        """标签使用BME模式"""
        if tag == u'O':
            return tag
        elif index == 0 and index != sentence_length - 1:
            return u'B-{}'.format(tag)
        elif index == sentence_length - 1:
            return u'E-{}'.format(tag)
        else:
            return u'M-{}'.format(tag)

    def __initialize(self):
        """初始化 """
        lines = self.__read_corpus_from_file(self.process_corpus_path)
        words_list = [line.strip().split('  ') for line in lines if line.strip()]
        del lines
        self.__init_sequence(words_list)

    def __init_sequence(self, words_list):
        """初始化字序列、词性序列、标记序列 """
        words_seq = [[word.split(u'/')[0] for word in words] for words in words_list]
        pos_seq = [[word.split(u'/')[1] for word in words] for words in words_list]
        tag_seq = [[p for p in pos] for pos in pos_seq]

        self.pos_seq = [[[pos_seq[index][i] for _ in range(len(words_seq[index][i]))]
                         for i in range(len(pos_seq[index]))] for index in range(len(pos_seq))]
        self.tag_seq = [[[self.__tag_perform(tag_seq[index][i], w, len(words_seq[index][i])) for w in range(len(words_seq[index][i]))]
                         for i in range(len(tag_seq[index]))] for index in range(len(tag_seq))]
        self.pos_seq = [[u'un'] + [p for pos in pos_seq for p in pos] + [u'un'] for pos_seq in
                        self.pos_seq]
        self.tag_seq = [[t for tag in tag_seq for t in tag] for tag_seq in self.tag_seq]
        # 为每个句子加上首尾标记
        self.word_seq = [[u'<BOS>'] + [w for word in word_seq for w in word] + [u'<EOS>'] for word_seq in words_seq] 

    def __extract_feature(self, word_grams):
        """特征选取"""
        features, feature_list = [], []
        for index in range(len(word_grams)):
            for i in range(len(word_grams[index])):
                word_gram = word_grams[index][i]
                feature = {u'w-1': word_gram[0], 
                           u'w': word_gram[1], 
                           u'w+1': word_gram[2], 
                           u'w-1:w': word_gram[0] + word_gram[1], 
                           u'w:w+1': word_gram[1] + word_gram[2],
                           u'bias': 1.0}
                feature_list.append(feature)
            features.append(feature_list)
            feature_list = []
        return features

    def __segment_by_window(self, words_list=None, window=3):
        """窗口切分"""
        words = []
        begin, end = 0, window
        for _ in range(1, len(words_list)):
            if end > len(words_list): break
            words.append(words_list[begin:end])
            begin = begin + 1
            end = end + 1
        return words

    def generator(self):
        """训练数据"""
        word_grams = [self.__segment_by_window(word_list) for word_list in self.word_seq]
        features = self.__extract_feature(word_grams)
        return features, self.tag_seq
