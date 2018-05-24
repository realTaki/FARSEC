
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer 
from gensim import corpora 
import string
import numpy as np
import math
import pandas as pd
import json

class Filtering:

    def __init__(self):
        self.cross_word = None
        self.SBRmat = None 
        self.NSBRmat = None
        self.SBR = None 
        self.NSBR = None
        self.dictionary =None
        self.BR = None

    # 文本预处理还需调整
    def treatment(self,dir):
        #读取文件
        BR = pd.DataFrame(pd.read_csv(dir,header=0))

        BR = BR[:500]
        print(" + [OK] read file!")

        # 处理summary和description
        text = BR['summary'] + BR['description']

        # 词根
        p_stemmer = PorterStemmer() 
        docs = []
        
        # 停词表
        list_stopWords = list(set(stopwords.words('english')))

        # 需要去掉标点符号/回车/数字等
        remove_map = string.punctuation+string.punctuation +'\n'
        remove_map = dict((ord(char), " ") for char in remove_map)

        for doc in text:
            # 将所有标点/数字替换成空格以方便分词
            doc = doc.translate(remove_map)

            # 待处理文本，先分词
            list_words = word_tokenize(doc)
            # 删除停词,过长和过短的词语,还原词根
            new_list_words = [w for w in list_words if 2 < len(w) < 15 ]
            filtered_words = [p_stemmer.stem(w) for w in new_list_words if not w in list_stopWords]
            docs.append(filtered_words)

        # 将格式化的数据插回表格
        BR.insert(1,'text',docs )
        # 保存
        BR.drop(['summary','description'],axis=1)
        self.BRtext = BR
        BR.to_csv('date.csv',index = False)

        self.SBR  = BR[BR['Security'] == 1]
        self.NSBR = BR[BR['Security'] == 0]
        print(" + [OK] the file has be Standardized!")

        return True

    def readDateFromFile(self,date):
        BR = pd.DataFrame(pd.read_csv(date,header=0))

        # 保存时,text文本被保存为"['a','b','c']"的字符串格式,重新加载需要解析,此处未完成
        to_do_this = False

        self.BR = BR
        self.SBR  = BR[BR['Security'] == 1]
        self.NSBR = BR[BR['Security'] == 0]

    def findSRW(self):
        # 对应论文3.1 Identifying Security Related Keywords
        
        SBR = self.SBR['text']
        # 提取SBR所有的术语,用词袋模型转化为向量
        self.dictionary = corpora.Dictionary(SBR)
        # 论文中提到去掉一些低频无用词语，并给了关于这些词语的网址，此处应补充

        # 词袋将文本转化为向量,第一步产生的是一个向量队列,(1,2)表示词语1出现2次,一句文本又一组词频表示,需要转化为矩阵
        SBR = [self.dictionary.doc2bow(doc) for doc in SBR ]
        SBRmat = self.makematrix(SBR,len(self.dictionary))

        tf_idf = self.tf_idf(SBRmat)

        '''tf_idf选出前Top100,
        此处有个疑问:
        举个例子, 一个词语在多个文档中出现,和一个词语在较少文本中出现,tf-idf矩阵为:
                t1      t2
        br1     0.2     0.8
        br2     0.2     0.05
        ...     ...     ...
        brn     0.2     0.1
        t1在大多数文本中出现, tf-idf分数低, 但是在各个文本都有分布, t2在很少文本中出现, 但能出现比较大的数字
        最终偏向哪一类型的词语值得考究

        程序运行时发现, 论文中筛选出来的SRW在这个程序中tf-idf反而分数低,可能跟此处的求和有关
        '''
        terms = np.sum(tf_idf,axis=0)
        terms = np.argsort(terms)[-100:]

        # 字典保留前100个词语
        self.dictionary.filter_tokens(good_ids=terms.tolist())
        self.cross_word = self.dictionary.token2id
        print("there are the security related word",self.cross_word)

        return True

    def farsec(self,support='farsecsq',train='knn'):

        M = self.ScoreKeywords(support = support)

        BRscore = self.ScoreBugReports(M)

        # 用0.75划分，高于0.75的为噪音NSBR
        BRscore[BRscore<0.75]=0
        BRscore[BRscore>0]=1
        print("remain : ",BRscore.shape[0] - BRscore.sum())

        to_train_in_this = 1

        return True

    def tf(self,D):
        # 每行的最大值,结果为n行一列矩阵
        max_w = np.array([D.max(axis= 1)]).T
        # 可能存在某些文本不出现关键词,向量全为0,避免0/0改为0/1
        max_w[max_w==0]=1
        tf = 0.5+(0.5 * D)/max_w 
        ''' tf 格式:

            t1 ... tn
        br1 
        ... 
        brn 
        
        '''
        return tf

    def idf(self,D):
        # 文件总数N,
        N=D.shape[0]
        D[D>0]=1

        D = D.sum(axis = 0)
        idf = np.log(N/D)

        ''' idf 格式:

        t1   ... tn
        idf1 ... idfn
        
        '''
        return idf

    def tf_idf(self,D):

        tf = self.tf(D)
        idf = self.idf(D)

        tf_idf = tf*idf
        ''' tf-idf 格式:

             t1 ... tn
        br1 
        ...   tf-idf
        brn 
        
        '''
        return tf_idf

    def ScoreKeywords(self,support):
        dictionary = self.dictionary

        # 对应论文W，安全相关词的个数
        W=len(self.dictionary) 
        SBR = self.SBR['text']
        NSBR = self.NSBR['text']
        dictionary.add_documents(SBR)
        dictionary.add_documents(NSBR)
        print('all terms : the security related keywords',len(self.dictionary),':',W)

        # 将SBR和NSBR所有的词语都翻译成矩阵,前一百个为安全相关词语,故n行前100列为特征矩阵,保存特征矩阵
        S = [dictionary.doc2bow(doc) for doc in SBR]
        S = self.makematrix(S,len(dictionary))
        SBR =  S[:,:W]
        self.SBRmat = SBR 
        
        NS = [dictionary.doc2bow(doc) for doc in NSBR]
        NS = self.makematrix(NS,len(dictionary))
        NSBR = NS[:,:W]
        self.NSBRmat = NSBR

        # 统计所有词语出现的频率和|S|和|NS|
        S = S.sum()
        NS = NS.sum()

        # 合计SBR和NSBR的每个词语的词频,1行n列
        # 论文中tf(Sw)与tf-idf的tf(t,br)不太一样,但论文没有指明怎么算,两处用#注释掉的代码用于处理这个问题
        # 根据结果发现, 使用论文中的tf(t,br)导出的tf介于0.5-1.0,对一个词在SBR中的权重和在NSBR中的权重区分度不大
        # 故此处直接统计词频的总和
        # SBR = self.tf(SBR)
        SBR = SBR.sum(axis= 0)
        if 'clni' in support:
            NSBR = self.CLNI() 
        if 'sq' in support:
            SBR *= SBR
        elif  'two' in support:
            SBR *= 2
        SBR /= S
        SBR[SBR>1]=1 # 不能大于１

        # NSBR = self.tf(NSBR)
        NSBR = NSBR.sum(axis = 0)
        NSBR /= NS
        NSBR[NSBR>1]=1

        #M为一行n列的矩阵，代表每个词语的score
        M = SBR/(SBR+NSBR)
        M[M>0.99]=0.99
        M[M<0.01]=0.01
        print("key words score : ")
        print(M)
        print("the average score:", M.sum())
        return M


    def ScoreBugReports(self,M):
        NSBR = self.NSBRmat
        NSBR[NSBR>1]=1

        Mstar = NSBR * M

        Mstar[Mstar==0] = 1
        Mstar = Mstar.prod(1)

        # 对于个别br 不存在安全相关词语(特征全为0)，M*为1，改为0
        Mstar[Mstar==1] = 0

        Mquote = NSBR * (1-M)
        Mquote[Mquote == 0] = 1
        Mquote = Mquote.prod(1)
        # 对于个别br 不存在安全相关词语(特征全为0)，则M'为1，保留1

        return Mstar/(Mstar+Mquote)

    def CLNI(self):
        NSBR = self.NSBRmat
        SBR = self.SBRmat

        del_count = 0
        e = 0
        while e<0.99:
            disNN = self.EuclideanDistances(NSBR,NSBR)
            disNN.sort()
            disNS = self.EuclideanDistances(NSBR,SBR)
            disNS.sort()
            # 排序后，当一份NSBR第五近的NSBR比最近的SBR远的话，那么它一定是噪音（top5的不同标签的百分比>0.8%--9%,一个进前五就能有20%)
            Noise =  disNN[:,4]-disNS[:,0]
            index = []
            for i in range(Noise.shape[0]):
                if Noise[i] > 0:
                    index.append(i)
            # 统计本次删除的，一共删除的,计算e，并从语料库中移除对应的NSBR
            delect=len(index)
            if del_count +delect == 0:
                e = 1
            else :
                e = del_count /(del_count +delect)
            del_count = del_count +delect
            NSBR = np.delete(NSBR, index, axis=0)  
        self.NSBRmat = NSBR
        print('CLNI delect : ', del_count)
        return NSBR

    def EuclideanDistances(self,A, B):
        AB = A.dot(B.T)
        Asq =  np.array([(A**2).sum(axis = 1)]).T # A行1列
        Bsq =  (B**2).sum(axis =1) # 1行B列
        # 结果是欧氏距离的平方(未开方），已经足够比较距离了
        distance = -2 * AB +Asq +Bsq
        return  distance

    def makematrix(self, data,lenth):
        
        matrix = np.zeros((len(data),lenth))
        
        for row in range(len(data)):
            for col in data[row]: 
                matrix[row,col[0]] = col[1]
               
        return matrix


