def corpus_file_transform(src_file,dst_file):
    import os
    assert os.path.isfile(src_file),'Src File Not Exists.'

    with open(src_file,'r',encoding = 'utf-8') as text_corpus_src:
        with open(dst_file,'w',encoding = 'utf-8') as text_corpus_dst:
            from tqdm.notebook import tqdm
            text_corpus_dst.write(''.join([(text_word + "\tS\n" if len(text_word) == 1 else (text_word[0] + "\tB\n" + ''.join([(w + "\tM\n") for w in text_word[1 : -1]]) + text_word[-1] + "\tE\n")) for text_line in tqdm_notebook(text_corpus_src.readlines()) for text_word in text_line.strip().split()]))

def IOForFeature(file,feature = None,mode = 'rb',featureList = ['A','B','C']):
    
    assert (mode == 'rb') or (mode == 'wb'),'The third parameter must be \'r\' or \'w\''
    assert not((mode == 'wb') and not feature),'The second parameter feature must not be empty.'
    
    try:
        import pickle
        with open(file,mode) as f:
            if mode == 'rb':
                feature = pickle.load(f)
            elif mode == 'wb':
                pickle.dump(feature,f)
    except:
        feature = {label : {} for label in featureList}
        
    return feature

def TrainingFeatureA(corpus,featureA,wordLabel = {'B' : 0, 'M' : 1, 'E' : 2, 'S' : 3}):
    # p(y_i|x_i)
    if not featureA:
        featureA = {} 
    
    for word in tqdm_notebook(corpus):
        if not featureA.get(word[0]):
            featureA[word[0]] = [0,0,0,0]
        featureA[word[0]][wordLabel[word[2]]] += 1
    return featureA

def TrainingFeatureB(corpus,featureB,wordLabel = {'B' : 0, 'M' : 1, 'E' : 2, 'S' : 3}):
    # p(y_(i+1)|x_i,y_i)
    if not featureB:
        featureB = {} 
    
    for word,nextword in tqdm_notebook(zip(corpus[:-1],corpus[1:])):
        if not featureB.get(word[0]):
            featureB[word[0]] = [[0,0,0,0] for i in range(4)]
        featureB[word[0]][wordLabel[word[2]]][wordLabel[nextword[2]]] += 1
    return featureB

def TrainingFeatureC(corpus,featureC,wordLabel = {'B' : 0, 'M' : 1, 'E' : 2, 'S' : 3}):
    # p(x_(i-1)|x_i,y_i),p(x_(i+1)|x_i,y_i)
    if not featureC:
        featureC = {} 
    
    for lastWord,word,nextWord in tqdm_notebook(zip(corpus[:-2],corpus[1:-1],corpus[2:])):
        if not featureC.get(word[0]):
            featureC[word[0]] = {label : {} for label in wordLabel}

        if not featureC[word[0]][word[2]].get(lastWord[0]):
            featureC[word[0]][word[2]][lastWord[0]] = [0,0]
        featureC[word[0]][word[2]][lastWord[0]][0] += 1

        if not featureC[word[0]][word[2]].get(nextWord[0]):
            featureC[word[0]][word[2]][nextWord[0]] = [0,0]
        featureC[word[0]][word[2]][nextWord[0]][1] += 1
        
    return featureC4

def featureTraining(feature,train_corpus,
                    featureList = ['A','B','C'],
                    featureFunction = {'A' : TrainingFeatureA, 'B' : TrainingFeatureB,'C' : TrainingFeatureC},
                    wordLabel = {'B' : 0, 'M' : 1, 'E' : 2, 'S' : 3}):
    for featureLabel in featureList:
        feature[featureLabel] = featureFunction[featureLabel](train_corpus,feature[featureLabel],wordLabel)

def getTestFeatureABC(test_str,feature,wordLabel):
    import numpy as np
    test_featureA = {word : (-np.log(np.array(feature['A'][word]) / sum(feature['A'][word]))).tolist()
                     if feature['A'].get(word) else [0,0,0,0] for word in test_str}
    test_featureB = {word : (-np.log(np.array(feature['B'][word]).T / np.array(feature['B'][word]).sum(axis = 1)).T).tolist() 
                     if feature['B'].get(word) else [[0,0,0,0] for label in wordLabel.keys()] for word in test_str}
    test_featureC = {word :{d1_key : {d2_key : d2_value for d2_key,d2_value in 
                                zip(d1_value.keys(),(np.array(list(d1_value.values())) / np.array(list(d1_value.values())).sum(axis = 0)).tolist())}
                      for d1_key,d1_value in feature['C'][word].items()} if feature['C'].get(word) else {label : {} for label in wordLabel.keys()}  for word in test_str}
    return test_featureA,test_featureB,test_featureC

def getDividedResult(wordLabel,relationDict,test_str):
    wordLabelk = list(wordLabel.keys())
    thisIndex = relationDict[-1][0].index(min(relationDict[-1][0]))
    dividedResult, lastIndex = [[test_str[-1],wordLabelk[thisIndex]]],relationDict[-1][1][thisIndex]

    for w_id in range(len(test_str) - 2,-1,-1):
        dividedResult.append([test_str[w_id],wordLabelk[lastIndex]])
        lastIndex = relationDict[w_id][1][lastIndex]

    dividedResult.reverse()
    resultString = ''.join([(' ' if d_R[1] == 'S' or d_R[1] == 'B' else '') + d_R[0] + (' ' if d_R[1] == 'S' or d_R[1] == 'E' else '') for d_R in dividedResult])
    return dividedResult,resultString

def CRFWordSeperate(test_str,feature,wordLabel = {'B' : 0, 'M' : 1, 'E' : 2, 'S' : 3} ):
    import numpy as np
    test_featureA,test_featureB,test_featureC = getTestFeatureABC(test_str,feature,wordLabel)

    relationDict = [[[test_featureA[test_str[w_id]][wordLabel[l_id]] * 
                                       (1 - (0 if w_id == 0                 else test_featureC[test_str[w_id]][l_id].get(test_str[w_id - 1], [0,0])[0])) *  
                                       (1 - (0 if w_id == len(test_str) - 1 else test_featureC[test_str[w_id]][l_id].get(test_str[w_id + 1], [0,0])[1]))
                                        for l_id in wordLabel],[0 for l_id in wordLabel]] for w_id in range(len(test_str))]
    relationDict[0][0][wordLabel['E']] = relationDict[0][0][wordLabel['M']] = float('inf')

    for w_id in range(1,len(test_str)):
        for l_id in wordLabel:
            candidateList = [test_featureB[test_str[w_id - 1]][wordLabel[l]][wordLabel[l_id]] 
                             * (1 - (0 if w_id == 0                 else test_featureC[test_str[w_id]][l_id].get(test_str[w_id - 1], [0,0])[0]))
                             * (1 - (0 if w_id == len(test_str) - 1 else test_featureC[test_str[w_id]][l_id].get(test_str[w_id + 1], [0,0])[1]))
                             + relationDict[w_id - 1][0][wordLabel[l]] for l in wordLabel]
            candidateList = [float('inf') if np.isnan(c_l) else c_l for c_l in candidateList]
            relationDict[w_id][0][wordLabel[l_id]] += min(candidateList)
            relationDict[w_id][1][wordLabel[l_id]] = candidateList.index(min(candidateList))
    relationDict[-1][0][wordLabel['B']] = relationDict[-1][0][wordLabel['M']] = float('inf')

    return getDividedResult(wordLabel,relationDict,test_str)

if __name__=="__main__":
    train_corpus_src = 'msr_training.utf8'
    train_corpus_dst = 'msr_training.utf8.pr'
    corpus_file_transform(train_corpus_src,train_corpus_dst)

    with open(train_corpus_dst,'r',encoding = 'utf-8') as f:
        train_corpus = f.readlines()
    print(train_corpus[:10])

    featureFile = 'feature.pkl'
    wordLabel = {'B' : 0, 'M' : 1, 'E' : 2, 'S' : 3}
    feature = IOForFeature(featureFile,mode='rb')

    featureTraining(feature,train_corpus)
    feature = IOForFeature(featureFile,feature,mode='wb')
    t_str = '最近内存在涨价，不能用以前等价值的物品交换了'
    dividedResult,resultString = CRFWordSeperate(t_str,feature,wordLabel)
    dividedSequences = ''.join([result[1] for result in dividedResult])
    print(resultString)
    print(dividedSequences)
    print(dividedResult)

    test_corpus_src = 'pku_training.utf8'
    test_corpus_dst = 'pku_training.utf8.pr'
    corpus_file_transform(test_corpus_src,test_corpus_dst)

    #将已分词的训练文件转换为未分词的测试文件
    with open(test_corpus_src,'r',encoding = 'utf-8') as f:
        test_sentences = f.readlines()
    test_sentences = [sentence.replace(' ','') for sentence in test_sentences]
    test_sentences = [sentence.replace('\n','') for sentence in test_sentences]

    #将获得测试文件的正确标注
    with open(test_corpus_dst,'r',encoding = 'utf-8') as f:
        test_corpus = f.readlines()
    test_label = ''.join([result[2] for result in test_corpus])

    print(test_sentences[0])
    print(test_corpus[:len(test_sentences[0])])
    print(test_label[:len(test_sentences[0])])

    dividedSequences = ''
    dividedResults = []
    resultStrings = []
    for sentences in tqdm_notebook(test_sentences[:500]):
        dividedResult,resultString = CRFWordSeperate(sentences,feature,wordLabel)
        dividedResults.append(dividedResult)
        resultStrings.append(resultString)
        dividedSequences += ''.join([result[1] for result in dividedResult])

    for d_R,r_S in zip(dividedResults[:10],resultStrings[:10]):
        print(r_S)
        print(d_R)

    count = [0,0,0,0]
    for d_S in dividedSequences:
        count[wordLabel[d_S]] += 1
    print(list(zip(wordLabel.keys(),count)))

    accurate = [0,0]
    for d_S in range(len(dividedSequences)):
        accurate[test_label[d_S] == dividedSequences[d_S]] += 1
    print('Wrong : %.2f%%, Right : %.2f%%' % (accurate[0] / sum(accurate) * 100,accurate[1] / sum(accurate) * 100))