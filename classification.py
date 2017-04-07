import jieba
import json
from datetime import datetime
import math

class Classification:

    def __init__(self):
        self.catagory={'互联网','体育','健康','军事','招聘','教育','文化','旅游','经济'}
        self.directory="news_text"
        self.data={}
        self.TF={}
        self.IDF={}
        self.TF_IDF={}
        self.documents={}
        self.sum_doucuments=0
        self.start_index=10
        self.train_index=1000
        self.max_index=1200
        with open("stopwords.txt") as file:
            self.stopwords=file.read().strip().split()
        self.useless=['，','。',' ','”','“','！','、','？','【','】','：',',','.','[',']','?','-','；','(',')','&nbsp;','（','）',':','"','《','》','<','>','%','$','&','nbsp',';','％','－','···']

    def InitializeData(self):
        for item in self.catagory:
            l=[]
            i=0
            for number in range(self.start_index, self.train_index,1):
                filename = self.directory + '/' +item+'/'+str(number)+'.txt'
                try:
                    with open(filename) as file:
                        text = file.read().strip().split()
                        for s in text:
                            for x in jieba.cut(s):
                                if x not in self.stopwords and x not in self.useless and not x.isdigit() and x!='\x00':
                                    l.append(x)
                        i=i+1
                except:
                    print(item+str(number))
                    continue

            self.data[item]=l
            self.documents[item]=i

        d = 0
        for item in self.documents:
            d += self.documents[item]
        self.sum_doucuments = d

    def Train(self):
        self.ComputeTF()
        self.ComputeIDF()
        self.ComputeTF_IDF()


    def ComputeTF(self):
        for item in self.catagory:
            d={}
            for x in self.data[item]:
                if x in d.keys():
                    d[x]=d[x]+1
                else:
                    d[x]=1
            for s in d.keys():
                d[s]=d[s]/len(self.data[item])
            self.TF[item]=d

    def ComputeIDF(self):
        for item in self.catagory:
            for x in set(self.data[item]):
               if x in self.IDF:
                   self.IDF[x]+=1
               else:
                   self.IDF[x] = 1

        for x in self.IDF:
            i=self.IDF[x]
            self.IDF[x]=math.log(len(self.catagory)/i)+0.01


    def ComputeTF_IDF(self):
        for item in self.catagory:
            l={}
            for x in set(self.data[item]):
                l[x]=self.TF[item][x]*self.IDF[x]
            self.TF_IDF[item]=l

            max_TFIDF=max(self.TF_IDF[item].values())
            min_TFIDF=min(self.TF_IDF[item].values())

            for s in self.TF_IDF[item]:
                d = self.TF_IDF[item][s]
                self.TF_IDF[item][s] = (d - min_TFIDF) / (max_TFIDF - min_TFIDF) + 1




    def WhichClass(self,t):
        pass
    def test(self):
        t=0
        i=0
        for item in self.catagory:
            l=[]
            for number in range(self.train_index, self.max_index,1):
                filename = self.directory + '/' +item+'/'+str(number)+'.txt'
                try:
                    with open(filename) as file:
                        text = file.read().strip().split()
                        for s in text:
                            for x in jieba.cut(s):
                                if x not in self.stopwords and x not in self.useless and not x.isdigit() and x != '\x00':
                                    l.append(x)
                        b = {}
                        for x in self.catagory:
                            b[x] = 1
                        for x in self.catagory:
                            for s in set(l):
                                if s in self.TF_IDF[x]:
                                    d=b[x]
                                    b[x] = d * self.TF_IDF[x][s]

                            b[x] = b[x] * self.documents[x] / self.sum_doucuments

                        max = '互联网'
                        for x in b:
                            if b[x] > b[max]:
                                max = x
                        if max == item:
                            t=t+1

                        i=i+1
                except:
                    print(filename)
                    continue
        return [t,i]

    def SaveData(self):
        with open('data.json','w') as file:
            file.write(json.dumps(self.data))
        with open('tf_idf.json','w') as file:
            file.write(json.dumps(self.TF_IDF))
        print('write back completed')







a=Classification()
a.InitializeData()
a.Train()
# for x in a.TF:
#     print(min(a.TF[x].values()))
[x,y]=a.test()
print(x)
print(y)
print(x/y)
print('over')