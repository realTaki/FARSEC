
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer 
from gensim import corpora 
import csv
import string
import numpy
import math

class Filtering:

    def __init__(self):
        self.cross_word = None
        self.SBRs = None 
        self.NSBRs = None
        self.SBR = None 
        self.NSBR = None
        self.dictionary =None

    def readcsv(self,dir,summary_col=-1, description_col=-1,security_col=-1,title=1):
        csvFile = open(dir, "r") 
        #读取文件
        BugReports = csv.reader(csvFile)
        print(" + [OK] read file!")

        # 读取summary和description
        text =[] 
        S_label=[]
        
        # 读取描述信息，和安全报告标记
        for br in BugReports:
            text.append(br[summary_col] + ' ' + br[description_col])
            S_label.append(br[security_col])
        
        # 处理除了第一行标题以外的文档
        text = self.treatment(text[title:])
        S_label = [int(n[0]) for n in S_label[title:]]
        S_label = numpy.array([S_label]).T    # n行1列标签    
        print(" + [OK] the file has be Standardized!")
        return text,S_label

    # 文本预处理还需调整
    def treatment(self,br):
        # 格式化字符串
        p_stemmer = PorterStemmer() 
        docs = []
        
        # 停词表
        list_stopWords = list(set(stopwords.words('english')))
        for doc in br:
            # 将所有标点/数字替换成空格以方便分词
            remove_punctuation_map = dict((ord(char), " ") for char in string.punctuation)
            remove_number_map = dict((ord(char), " ") for char in string.digits)
            doc = doc.translate(remove_punctuation_map)
            doc = doc.translate(remove_number_map)
            # 待处理文本，先分词
            list_words = word_tokenize(doc)
            # 删除停词,过长和过短的词语,还原词根
            new_list_words = [w for w in list_words if 2 < len(w) < 15 ]
            filtered_words = [p_stemmer.stem(w) for w in new_list_words if not w in list_stopWords]
            docs.append(filtered_words)

        return docs

    def partition(self, br,label):
        SBR = []
        NSBR = []

        for i in range(label.shape[0]):
            if label[i,0]==1:
                SBR.append(br[i])
            else :
                NSBR.append(br[i])
        self.SBRs = SBR
        self.NSBRs =NSBR
        return True

    def findSRW(self, br,label):
        # 对应论文3.1 Identifying Security Related Keywords
        self.partition(br,label)
        self.dictionary = corpora.Dictionary(self.SBRs)
        # 论文中提到去掉一些低频无用词语，并给了关于这些词语的网址，此处应补充

        SBR = [self.dictionary.doc2bow(doc) for doc in self.SBRs]
        SBR = self.makematrix(SBR,len(self.dictionary))

        tf_idf = self.tf_idf(SBR)
        # tf_idf选出前Top100
        terms = (tf_idf.T).dot( numpy.ones((SBR.shape[0],1))  )
        terms = numpy.argsort(terms.T)[0,-100:]
        # 字典保留前100个词语

        self.dictionary.filter_tokens(good_ids=terms.tolist())
        self.cross_word = self.dictionary.token2id
        print("there are the security cross word",self.cross_word)

        # 用已经得到的特征集（安全相关词），讲BR转化成向量
        SBR = [self.dictionary.doc2bow(doc) for doc in self.SBRs]
        self.SBR = self.makematrix(SBR,len(self.dictionary))
        NSBR = [self.dictionary.doc2bow(doc) for doc in self.NSBRs]
        self.NSBR = self.makematrix(NSBR,len(self.dictionary))

        return True

    def farsec(self,support='farsecsq',train='knn'):
        
        if 'clni' in support:
            self.CLNI()
        if 'sq' in support:
            M = self.ScoreKeywords(support = 'sq')
        elif 'two' in support:
            M = self.ScoreKeywords(support = 'two')
        else :
            M = self.ScoreKeywords(support = '')
        BRscore = self.ScoreBugReports(M)

        # 用0.75划分，高于0.75的为噪音NSBR
        BRscore[BRscore<0.75]=0
        BRscore[BRscore>0]=1
        print(BRscore.shape[0] - numpy.ones((1,BRscore.shape[0])).dot(BRscore)[0,0])

        to_train_in_this = 1

        return True

    def tf(self,D):
        max_w = numpy.zeros((D.shape[0],1))
        for i in range(D.shape[0]):
            max_w[i]= max(D[i])
        tf = 0.5+(0.5 * D)/max_w 
        ''' tf 格式:

            t1 ... tn
        br1 
        ... 
        brn 
        
        '''
        return tf

    def idf(self,D):
        N=D.shape[0]
        D[D>0]=1
        D[D<1]=N

        t = numpy.zeros((1,D.shape[0])).dot(D)
        idf = numpy.log(N/D)
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
        dictionary.add_documents(self.SBRs)
        dictionary.add_documents(self.NSBRs)
        print('all terms : the security related keywords',len(self.dictionary),':',W)

        # 将SBR和NSBR所有的词语都翻译成矩阵,统计所有词语出现的频率和|S|和|NS|,前一百个为安全相关词语
        S = [dictionary.doc2bow(doc) for doc in self.SBRs]
        S = self.makematrix(S,len(dictionary))
        SBR = S[:,:W]
        
        NS = [dictionary.doc2bow(doc) for doc in self.NSBRs]
        NS = self.makematrix(NS,len(dictionary))
        NSBR = NS[:,:W]

        # 保存SBR/NSBR用于进一步过滤
        self.SBR = SBR
        self.NSBR = NSBR
        
        S = S.dot(numpy.ones((S.shape[1],1)))
        S = numpy.ones((1,S.shape[0])).dot(S)
        NS = NS.dot(numpy.ones((NS.shape[1],1)))
        NS = numpy.ones((1,NS.shape[0])).dot(NS)

        # 合计SBR和NSBR的词频
        SBR = numpy.ones((1,SBR.shape[0])).dot(SBR)
        if support == 'sq':
            SBR *= SBR
        elif support == 'two':
            SBR *= 2
        SBR /= S
        SBR[SBR>1]=1

        NSBR = numpy.ones((1,NSBR.shape[0])).dot(NSBR)
        NSBR /= NS
        NSBR[NSBR>1]=1

        #M为一行n列的矩阵，代表每个词语的score
        M = SBR/(SBR+NSBR)
        M[M>0.99]=0.99
        M[M<0.01]=0.01
        print("key words score : ")
        print(M)
        return M

    def ScoreBugReports(self,M):
        NSBR = self.NSBR
        NSBR[NSBR>1]=1

        Mstar = NSBR.dot(M.T)
        Mquote = NSBR.dot(1-M.T)
        # 对于个别br 不存在安全相关词语，则上面两个式子都为0，应该避免0/0,改为0/1
        Mquote[Mquote==0]=1

        return Mstar/(Mstar+Mquote)

    def CLNI(self):
        NSBR = self.NSBR
        SBR = self.SBR

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
            NSBR = numpy.delete(NSBR, index, axis=0)  
            n = len(self.NSBRs)
            self.NSBRs = [self.NSBRs[x] for x in range(n) if not x in index]
        print('CLNI delect : ', del_count)
        return True

    def EuclideanDistances(self,A, B):
        AB = A.dot(B.T)
        Asq =  (A**2).dot(numpy.ones((A.shape[1],1)))  # A行1列
        Bsq =  (   (B**2).dot(numpy.ones((B.shape[1],1)))  ).T # 1行B列
        # 结果是欧氏距离的平方(未开方），已经足够比较距离了
        distance = -2 * AB +Asq +Bsq
        return  distance

    def makematrix(self, data,lenth):
        
        matrix = numpy.zeros((len(data),lenth))
        
        for row in range(len(data)):
            for col in data[row]: 
                matrix[row,col[0]] = col[1]
                
        return matrix


