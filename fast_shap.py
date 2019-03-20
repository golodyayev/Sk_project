import scipy.special
import numpy as np
import itertools


class fast_shap:
    """
    #Class for calculation  Shapley Value based on feature corelation matrix
    #time - 20.03.2019
    #Skoltech University, Machine Learning course
    #See how to USE IT in {fast_shap_example.ipynb}
    """

    def __init__(self, sub, X, f, print_result=False):
        self.f = f
        self.y_test = self.f(X)
        self.X = X
        self.shap_val_score = [0 for i in range(self.X.shape[-1])]
        self.M = self.X.shape[1]
        self.mus = np.mean(self.X, axis=0)
        self.std = np.std(self.X, axis=0)
        self.sample_num = sub
        self.dlinna = len(self.X)
        self.print_result = print_result
        self.iter = range(20)
        # self.corelation = corelation

    def randomizer(self, vector):
        return (np.random.permutation(vector))

    def normal_sampling(self, mu, std, len_dis=100):
        # create random sampling dis
        return (np.random.normal(mu, std, len_dis))

    def powerset(self, iterable):
        # create all subsets from iterable
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))

    def kernel(self, M, s):
        # create combinations indexes to iteration
        if s == 0:  # or s == M:
            return 1
        return 1 / (scipy.special.binom(M, s))  # * (M/s)
        # return(2/s/(s+1)/(s+2))

    # def f(self,array):
    #   beta = np.array([2]*(self.M))
    #  return np.dot(array,beta) + 10

    # def f_(self):
    #   beta = np.array([2]*(self.M))
    #  return np.dot(self.X,beta) + 10

    def log_dif(self, y, y_test):

        return (y - y_test)

    def mu(self, s, idx):
        vector = np.copy(self.X)

        if len(s) == 0:
            vector = vector.T
            vec = np.copy(vector[idx])
            vector[idx] = self.randomizer(vec)
            # vector[idx] = self.normal_sampling(self.mus[idx],self.std[idx],self.dlinna)
            return (self.log_dif(self.f(vector.T), self.y_test))
        else:
            vector2 = np.copy(self.X)
            vector = vector.T
            vector2 = vector2.T
            vec = np.copy(vector[idx])
            vector[idx] = self.randomizer(vec)
            # vector[idx] = self.normal_sampling(self.mus[idx],self.std[idx],self.dlinna)

            for idy in s:
                vec = np.copy(vector[idy])
                vector[idy] = self.randomizer(vec)
                vec2 = np.copy(vector2[idy])
                vector2[idy] = self.randomizer(vec2)
            return (self.log_dif(self.f(vector.T), self.f(vector2.T)))

    def shap_iteration(self, idx, neigbor):
        scores = 0
        for s in self.powerset(neigbor):
            # print(s)
            differ = []
            for i in self.iter:
                differ.append(self.mu(s, idx))
            scores += self.kernel(len(neigbor) + 1, len(s)) * np.mean(differ, axis=0)

        # print(scores)
        self.shap_val_score[idx] = scores / self.sample_num  # * 1/len(neigbor)

    def create_graphs(self, cor):
        features_n = self.M
        len_neigh = self.sample_num
        glob = []
        for idx in range(features_n):  # len(cor)):
            neigbor = []
            vector = cor[idx]
            dd = (np.argsort(vector)[features_n - 2])
            neigbor.append(dd)
            for i in range(len_neigh):
                if i != 0:
                    vector = cor[neigbor[i - 1]]
                k = 1
                while True:
                    dd = (np.argsort(vector)[features_n - k])
                    # print (dd)
                    if dd in neigbor:
                        # print("here")
                        k = k + 1
                    else:
                        neigbor.append(dd)
                        break
            glob.append(neigbor)
        return (glob)

    def fit(self, corr):
        for idx, neigbor in enumerate(self.create_graphs(corr)):
            if self.print_result:
                print("calculate for feature {} out of {}".format(idx, self.M))
            self.shap_iteration(idx, neigbor)

            # print(self.shap_val_score)
        return (np.array(self.shap_val_score))