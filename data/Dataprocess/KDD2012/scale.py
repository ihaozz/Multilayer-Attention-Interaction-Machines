import math
import numpy as np
def scale(x):
    if x > 2:
        x = int(math.log(float(x))**2)
    return x

DATA_PATH = './KDD2012/'

def scale_each_fold():
    for i in range(1,11):
        print('now part %d' % i)
        data = np.load(DATA_PATH + 'part'+str(i)+'/train_x.npy')
        part = data[:,0:13]
        for j in range(part.shape[0]):
            if j % 100000 ==0:
                print(j)
            part[j] = list(map(scale, part[j]))
        np.save(DATA_PATH + 'part' + str(i) + '/train_x2.npy', data)


if __name__ == '__main__':
    scale_each_fold()