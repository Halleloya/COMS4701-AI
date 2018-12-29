'''
I got 0.993 on train.txt, 0.982 on text.txt, 0.978 on dev.txt
After tunning, I set m=1 and k=0.1
I got 0.997 on train.txt, 0.982 on text.txt, 0.982 on dev.txt
After Part5 (modifying stopwords.txt, replace numbers with special symbol, leave some puction '$!?')
I got 0.9973 on train.txt, 0.9838 on text.txt, 0.9856 on dev.txt
'''
import sys
import string
import math

class NbClassifier(object):

    """
    A Naive Bayes classifier object has three parameters, all of which are populated during initialization:
    - a set of all possible attribute types
    - a dictionary of the probabilities P(Y), labels as keys and probabilities as values
    - a dictionary of the probabilities P(F|Y), with (feature, label) pairs as keys and probabilities as values
    """
    def __init__(self, training_filename, stopword_file):
        self.attribute_types = set()
        self.label_prior = {}    
        self.word_given_label = {}   

        self.collect_attribute_types(training_filename)
        if stopword_file is not None:
            self.remove_stopwords(stopword_file)
        self.train(training_filename)


    """
    A helper function to transform a string into a list of word strings.
    You should not need to modify this unless you want to improve your classifier in the extra credit portion.
    """
    def extract_words(self, text):
        no_punct_text = "".join([x for x in text.lower() if not x in (string.punctuation)])#.strip('$!?'))]) #reserve $ and !
        return [word for word in no_punct_text.split()]


    """
    Given a stopword_file, read in all stop words and remove them from self.attribute_types
    Implement this for extra credit.
    """
    def remove_stopwords(self, stopword_file):
        self.attribute_types.difference(set())
        input = open(stopword_file, 'r', encoding='UTF-8')
        for i in input.readlines():
            i = i.strip('\n')
            if i in self.attribute_types:
                self.attribute_types.remove(i)
        tmpset = set()
        for i in self.attribute_types:
            tmpset.add(i)
        for i in tmpset:
            if i.isdigit():
                self.attribute_types.remove(i)
        self.attribute_types.add('##thisisnumber##') #replace all numbers with a common symbol -> strange str
        self.attribute_types.add('##thisislongnumber##') #I guess add long number is useful
         

    """
    Given a training datafile, add all features that appear at least m times to self.attribute_types
    """
    def collect_attribute_types(self, training_filename, m=2):
        self.attribute_types = set()
        input = open(training_filename, 'r', encoding='UTF-8')
        dic = {}
        for i in input.readlines():
            x = self.extract_words (i)
            for j in range (1, len(x)):
                if x[j] in dic:
                    dic[x[j]] += 1
                else:
                    dic[x[j]] = 1
        #print (dic)
        for k in dic:
            if dic[k] >= m:
                self.attribute_types.add(k)
        print ("lenattri:", len(self.attribute_types))
            


    """
    Given a training datafile, estimate the model probability parameters P(Y) and P(F|Y).
    Estimates should be smoothed using the smoothing parameter k.
    """
    def train(self, training_filename, k=0.001):
        self.label_prior = {}
        self.word_given_label = {}
        #count_word_given_label = {}
        input = open(training_filename, 'r', encoding='UTF-8')
        total_spam = 0
        total_ham = 0
        total_spam_n = 0
        total_ham_n = 0
        for i in input.readlines():
            x = self.extract_words (i)
            if x[0] == "ham":
                total_ham += 1
                for i in range (1, len(x)):
                    '''
                    if x[i].isdigit():
                        if len(x[i]) > 10:
                            x[i] = '##thisislongnumber##'
                        else:
                            x[i] = '##thisisnumber##'
                    '''
                    if (x[i], "ham") in self.word_given_label:
                        self.word_given_label[(x[i], "ham")] += 1
                    else:
                        self.word_given_label[(x[i], "ham")] = 1
                    total_ham_n += 1
            elif x[0] == "spam":
                total_spam += 1
                for i in range (1, len(x)):
                    '''
                    if x[i].isdigit():
                        if len(x[i]) > 10:
                            x[i] = '##thisislongnumber##'
                        else:
                            x[i] = '##thisisnumber##'
                    '''
                    if (x[i], "spam") in self.word_given_label:
                        self.word_given_label[(x[i], "spam")] += 1
                    else:
                        self.word_given_label[(x[i], "spam")] = 1
                    total_spam_n += 1
        
        total = total_spam + total_ham
        self.label_prior["ham"] = total_ham / total
        self.label_prior["spam"] = total_spam / total
        totalwords = len(self.attribute_types)
        for key in self.attribute_types:
            if (key, "ham") in self.word_given_label:
                self.word_given_label[(key, "ham")] = (self.word_given_label[(key, "ham")] + k) / (total_ham_n + k*totalwords)
            else:
                self.word_given_label[(key, "ham")] = (k) / (total_ham_n + k*totalwords)
            if (key, "spam") in self.word_given_label:
                self.word_given_label[(key, "spam")] = (self.word_given_label[(key, "spam")] + k) / (total_spam_n + k*totalwords)
            else:
                self.word_given_label[(key, "spam")] = (k) / (total_spam_n + k*totalwords)
        print ("self.word_given_label:", len(self.word_given_label))
        #print (self.word_given_label)
        print (self.label_prior)
        #print (self.word_given_label[('##thisislongnumber##', "ham")], self.word_given_label[('##thisislongnumber##', "spam")])

    """
    Given a piece of text, return a relative belief distribution over all possible labels.
    The return value should be a dictionary with labels as keys and relative beliefs as values.
    The probabilities need not be normalized and may be expressed as log probabilities. 
    """
    def predict(self, text):
        dic = {}
        x = self.extract_words (text)
        #print (x)
        logp_spam = math.log(self.label_prior["spam"])
        logp_ham = math.log(self.label_prior["ham"])
        #print (logp_spam, logp_ham)
        for i in range (1, len(x)):
            '''
            if x[i].isdigit():
                if len(x[i]) > 10:
                    x[i] = '##thisislongnumber##'
                else:
                    x[i] = '##thisisnumber##'
            '''
            if (x[i], "spam") in self.word_given_label:
                logp_spam += math.log(self.word_given_label[(x[i], "spam")])
            if (x[i], "ham") in self.word_given_label:
                logp_ham += math.log(self.word_given_label[(x[i], "ham")])
        dic["ham"] = logp_ham
        dic["spam"] = logp_spam
        #print (dic)
        return dic


    """
    Given a datafile, classify all lines using predict() and return the accuracy as the fraction classified correctly.
    """
    def evaluate(self, test_filename):
        input = open(test_filename, 'r', encoding='UTF-8')
        total = 0
        correct = 0
        for i in input.readlines():
            total += 1
            dic = self.predict(i)
            res = "null"
            if dic["spam"] > dic["ham"]:
                res = "spam"
            else:
                res = "ham"
            x = self.extract_words (i)
            if x[0] == res:
                correct += 1
            else:
                print (i)
        pcrct = correct / total
        return pcrct



if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("\nusage: ./hmm.py [training data file] [test or dev data file] [(optional) stopword file]")
        exit(0)
    elif len(sys.argv) == 3:
        classifier = NbClassifier(sys.argv[1], None)
    else:
        classifier = NbClassifier(sys.argv[1], sys.argv[3])
    print(classifier.evaluate(sys.argv[2]))
