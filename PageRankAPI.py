import pandas as pd
import numpy as np
import operator


class PageRank:
    def __init__(self):
        self.activate = False
        self.temp_page_rank_results = {}
        self.page_rank_results = {}
        self.in_neighbors = {}
        self.out_degree = {}
        self.n = 0

    # load data from the csv path and initiate all the dictionaries with the necessary values
    def load_graph(self, path):
        data = pd.read_csv(path, names=["i", "j"])
        values = np.unique(data[["i", "j"]])
        self.n = values.size
        # initiate all dictionaries
        self.temp_page_rank_results = dict.fromkeys(values.astype(str), 0.0)
        self.page_rank_results = dict.fromkeys(values.astype(str), 1.0 / self.n)
        self.in_neighbors = dict.fromkeys(values.astype(str), None)
        self.out_degree = dict.fromkeys(values.astype(str), 0)
        # find for each node the in-neighbors and the out degree
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.rsplit(",")
                ind = line[1].rsplit("\n")[0]
                if self.in_neighbors[ind] is None:
                    self.in_neighbors[ind] = set([])
                self.in_neighbors[ind].add(line[0])
                self.out_degree[line[0]] += 1

    # calculate the page rank of all the nodes in the network, stop after convergence or max_iterations
    def calculate_page_rank(self, beta=0.85, epsilon=0.001, max_iterations=20):
        num_of_iterations = 0
        # stops when the sum of difference values between iteration t to iteration t-1 are less or equal to epsilon
        # or when num_of_iterations = max_iteration
        while sum([abs(self.page_rank_results[n] - self.temp_page_rank_results[n]) for n in self.page_rank_results]) > epsilon and num_of_iterations <= max_iterations:
            self.temp_page_rank_results = self.page_rank_results
            self.page_rank_results = dict.fromkeys(self.temp_page_rank_results.keys(), 0)
            for node in self.page_rank_results:
                # if not exist incoming edges update page rank to 0
                if self.in_neighbors[node] is None:
                    self.page_rank_results[node] = 0
                else:
                    # calculate the page rank of iteration t from the neighbors nodes
                    # with incoming edges to current node, using beta for "teleportation"
                    for neighbor in self.in_neighbors[node]:
                        self.page_rank_results[node] += beta * (self.temp_page_rank_results[neighbor] / self.out_degree[neighbor])
            s = sum(self.page_rank_results.values())
            # re-insert the leaked page rank to each node in the network
            for node in self.page_rank_results:
                self.page_rank_results[node] += (1 - s) / self.n
            num_of_iterations += 1

        self.activate = True

    # return the page rank of the node with key node_name
    def get_PageRank(self, node_name):
        if self.activate and self.page_rank_results.__contains__(node_name):
            return self.page_rank_results[node_name]
        else:
            return -1

    # return a list of the top n nodes with the highest page rank
    def get_top_nodes(self, n):
        if self.activate and n > 0:
            return [(k, v) for k, v in (dict(sorted(self.page_rank_results.items(), key=operator.itemgetter(1), reverse=True)[:n])).items()]
        else:
            return []

    # return a list of all nodes page rank sorted from the highest to the lowest page rank value
    def get_all_PageRank(self):
        if self.activate:
            return [(k, v) for k, v in (dict(sorted(self.page_rank_results.items(), key=operator.itemgetter(1), reverse=True))).items()]
        else:
            return []


p_r = PageRank()
p_r.load_graph("Wikipedia_votes.csv") # soc-Epinions1.csv
p_r.calculate_page_rank()
# print(p_r.get_PageRank('0'))
# print(p_r.get_top_nodes(10))
for val in p_r.get_all_PageRank():
    print(val)

