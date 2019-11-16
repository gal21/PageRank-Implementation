import pandas as pd
import numpy as np
from scipy._lib.six import xrange
from scipy.sparse import csc_matrix


class PageRank:
    def __init__(self):
        self.page_rank_results = {}
        self.page_rank_matrix = None
        self.n = 0

    def load_graph(self, path):
        data = pd.read_csv(path, names=["i", "j"])
        values = np.unique(data[["i", "j"]])
        df = pd.DataFrame(0, index=values, columns=values)
        f = df.index.get_indexer
        df.values[f(data.i), f(data.j)] = 1
        self.page_rank_matrix = df

    def calculate_page_rank(self, beta=0.85, epsilon=0.001, max_iterations=20):
        self.n = self.page_rank_matrix.shape[0]

        # transform G into markov matrix A
        A = csc_matrix(self.page_rank_matrix, dtype=np.float)
        nodes = np.array(self.page_rank_matrix.columns)
        rsums = np.array(A.sum(1))[:, 0]
        ri, ci = A.nonzero()
        A.data /= rsums[ri]

        # bool array of sink states
        sink = rsums == 0

        # Compute pagerank r until we converge
        ro, r = np.zeros(self.n), np.ones(self.n)
        num_of_iterations = 0
        while np.sum(np.abs(r - ro)) > epsilon and num_of_iterations <= max_iterations:
            ro = r.copy()
            # calculate each pagerank at a time
            for i in xrange(0, self.n):
                # inlinks of state i
                Ai = np.array(A[:, i].todense())[:, 0]
                # account for sink states
                Di = sink / float(self.n)
                # account for teleportation to state i
                Ei = np.ones(self.n) / float(self.n)

                r[i] = ro.dot(Ai * beta + Di * beta + Ei * (1 - beta))
                self.page_rank_results[str(nodes[i])] = r[i]
            num_of_iterations += 1

        return r / float(sum(r))
        # self.page_rank_results = sorted(self.page_rank_results.items(), key=operator.itemgetter(1))

    def get_PageRank(self, node_name):
        return self.page_rank_results[node_name]

    def get_top_nodes(self, n):
        pass

    def get_all_PageRank(self):
        pass


p_r = PageRank()
p_r.load_graph("Wikipedia_votes.csv")
print(p_r.calculate_page_rank())