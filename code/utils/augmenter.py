import pickle
from pdb import set_trace as bp

def augment(data, knn):
    
    fp_write = open('../../data/quora/train_barhaaya.tsv', 'w')
    data = [each.strip().split('\t') for each in data]
    num = 105736
    tags = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
    count = 0
    for lines in data:
        label = int(lines[-1])
        content = '\t'.join(lines)
        content += '\n'
        fp_write.write(content)
        if label == 1 and count != num:
            words1 = lines[0].split()
            tags1 = lines[1].split()
            words2 = lines[2].split()
            tags2 = lines[3].split()
            flag = 0
            for i, tag in enumerate(tags1):
                if tag in tags:
                    if words1[i] in knn:
                        words1[i] = knn[words1[i]]
                        flag = 1
            for i, tag in enumerate(tags2):
                if tag in tags:
                    if words2[i] in knn:
                        words2[i] = knn[words2[i]]
                        flag = 1
            if flag == 1:
                words1 = ' '.join(words1)
                tags1 = words1 + '\t' + ' '.join(tags1)
                words2 = tags1 + '\t' + ' '.join(words2)
                tags2 = words2 + '\t' + ' '.join(tags2)
                label = tags2 + '\t' + str(label)
                content = label + '\n'
                fp_write.write(content)
                count += 1
            
    

if __name__ == '__main__':
    train_file = open('../../data/quora/train_tokenized.tsv')
    knn_file = open('../../data/quora/knn_words.pkl', 'rb')
    knn = pickle.load(knn_file)
    data = train_file.readlines()
    train_file.close()
    augment(data, knn)
