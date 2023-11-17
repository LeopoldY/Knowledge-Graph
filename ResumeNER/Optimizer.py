import pycrfsuite
import sklearn_crfsuite
import joblib

from sklearn.metrics import classification_report
from CorpusProcess import CorpusProcess

def train(data_path:str = None, model_path:str = None, T = True):
    '''
    训练模型
    :param data_path: 数据路径
    :param model_path: 模型保存路径
    :param T: 是否打印训练过程，默认打印
    :return: None
    '''
    trainer = pycrfsuite.Trainer(verbose=T)
    train_crop = CorpusProcess(root=data_path, 
                               corp_path='train.char.bmes', # 训练集
                               precessed_path='train.char.bmes.processed' # 预处理后的训练集
                               )
    features, tag_seq = train_crop.generator()
    for xseq, yseq in zip(features, tag_seq):
        trainer.append(xseq, yseq)
    trainer.set_params({ # 设置训练参数
        'c1': 1.0,
        'c2': 1e-3,
        'max_iterations': 50,
        'feature.possible_transitions': True
    })
    trainer.train(model_path + 'model.crfsuite')
    print('Model saved to: {}'.format(model_path))

def test(data_path = None, model_path = None):
    '''
    测试模型
    :param data_path: 数据路径
    :param model_path: 模型保存路径
    :return: None
    '''
    tagger = pycrfsuite.Tagger()
    try:
        tagger.open(model_path + 'model.crfsuite')
    except Exception as e:
        print(e)
        print("请先训练模型")
        return
    test_crop = CorpusProcess(root = data_path, 
                              corp_path = 'test.char.bmes', 
                              precessed_path='test.char.bmes.processed')
    features, tag_seq = test_crop.generator()
    pred = [tagger.tag(xseq) for xseq in features]
    y_true = [i for item in tag_seq for i in item]
    y_pred = [i for item in pred for i in item]
    labels = sorted(set(y_pred))
    labels.remove('O')
    print(classification_report(y_true, y_pred, labels=labels)) # 打印每个标签的精确率、召回率、F1值

def sklearn_train(root, corp_path, precessed_path):
    '''
    使用sklearn_crfsuite训练模型
    :param root: 数据路径
    :param corp_path: 训练集
    :param precessed_path: 预处理后的训练集
    :return: None
    '''
    CRF = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
        verbose=True
    )
    train_crop = CorpusProcess(root=root, 
                               corp_path=corp_path, # 训练集
                               precessed_path=precessed_path # 预处理后的训练集
                                )
    features, tag_seq = train_crop.generator()
    CRF.fit(features, tag_seq)
    joblib.dump(CRF, './model/sklearn_crf.model')
    print('Model saved to: {}'.format('./model/sklearn_crf.model'))

def sklearn_test(root, corp_path, precessed_path):
    CRF = joblib.load('./model/sklearn_crf.model')
    test_crop = CorpusProcess(root=root, 
                              corp_path=corp_path, 
                              precessed_path=precessed_path)
    features, tag_seq = test_crop.generator()
    pred = CRF.predict(features)
    y_true = [i for item in tag_seq for i in item]
    y_pred = [i for item in pred for i in item]
    labels = sorted(set(y_pred))
    labels.remove('O')
    print(classification_report(y_true, y_pred, labels=labels)) # 打印每个标签的精确率、召回率、F1值