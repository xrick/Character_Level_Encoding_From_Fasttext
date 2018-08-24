import numpy as np
import os

file_path = "../pretrained_glove_word_vectors/glove.42B.300d.txt"#"/Volumes/My Passport/GloVe/glove.840B.300d.txt"
vectors = {}
#test_vectors = {}
with open(chinese_file_path,'r') as fin:
    count = 1
    for line in fin : 
        line_split = line.strip().split(" ")
        #print("splitted line : ",line_split[0])
        vec = np.array(line_split[1:],dtype=float)
        word = line_split[0]
        #print("current processing word is : ",word)
        for char in word:
            if char in vectors:
                #print("shape of vectors[{0}][0] is : {1}".format(char,vectors[char][0].shape))
                #print("shape of vectors[{0}][1] is : {1}".format(char, vectors[char][1]))
                #print("current vec is : ", vec.shape)
                #vectors[char] = (vectors[char][0] + (vec,1),vectors[char][1])
                if vectors[char][0].shape != vec.shape:
                    z = np.ones(1,dtype=float)
                    _revec = np.concatenate((vec,z),axis=0)
                    #print("current char : {} 's _revec is {}".format(char,_revec) )
                    vectors[char] = (vectors[char][0] +
                                         _revec, vectors[char][1])
                else:
                    vectors[char] = (vectors[char][0] +
                                         vec, vectors[char][1])    
            else:
                vectors[char] = (vec, 1)

base_name = os.path.splitext(os.path.basename(chinese_file_path))[0] + '_char.txt'
with open(base_name, 'w') as f2:
    for word in vectors:
        avg_vector = np.round(
            (vectors[word][0] / vectors[word][1]), 6).tolist()
        print("{0}'s avg_vector is {1}".format(word,avg_vector))
        f2.write(word + " " + " ".join(str(x) for x in avg_vector) + "\n")
print("character level encoding completed.....")
        

"""
vectors = {}
with open(file_path, 'rb') as f:
    for line in f:
        line_split = line.strip().split(" ")
        vec = np.array(line_split[1:], dtype=float)
        word = line_split[0]

        for char in word:
            if ord(char) < 128:
                if char in vectors:
                    vectors[char] = (vectors[char][0] + vec,
                                     vectors[char][1] + 1)
                else:
                    vectors[char] = (vec, 1)

base_name = os.path.splitext(os.path.basename(file_path))[0] + '-char.txt'
with open(base_name, 'wb') as f2:
    for word in vectors:
        avg_vector = np.round(
            (vectors[word][0] / vectors[word][1]), 6).tolist()
        f2.write(word + " " + " ".join(str(x) for x in avg_vector) + "\n")

"""
