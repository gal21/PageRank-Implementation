import operator
import os

activate = False
temp_page_rank_results = {}
page_rank_results = {}
in_neighbors = {}
out_degree = {}
num_of_nodes = 0


# load data from the csv path and initiate all the dictionaries with the necessary values
def load_graph(path):
    global num_of_nodes
    global temp_page_rank_results
    global page_rank_results
    global in_neighbors
    global out_degree

    # initiate all dictionaries and find for each node the in-neighbors and the out degree
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.rsplit(",")
            ind = line[1].rsplit("\n")[0]
            if not temp_page_rank_results.__contains__(ind):
                temp_page_rank_results[ind] = 0
            if not temp_page_rank_results.__contains__(line[0]):
                temp_page_rank_results[line[0]] = 0
            if not in_neighbors.__contains__(ind):
                in_neighbors[ind] = set([])
            in_neighbors[ind].add(line[0])
            if not out_degree.__contains__(line[0]):
                out_degree[line[0]] = 0
            out_degree[line[0]] += 1
    num_of_nodes = len(temp_page_rank_results)
    page_rank_results = dict.fromkeys(temp_page_rank_results.keys(), 1.0 / num_of_nodes)


# calculate the page rank of all the nodes in the network, stop after convergence or max_iterations
def calculate_page_rank(beta=0.85, epsilon=0.001, max_iterations=20):
    global num_of_nodes
    global activate
    global temp_page_rank_results
    global page_rank_results
    global in_neighbors
    global out_degree
    num_of_iterations = 0
    # stops when the sum of difference values between iteration t to iteration t-1 are less or equal to epsilon
    # or when num_of_iterations = max_iteration
    while sum([abs(page_rank_results[n] - temp_page_rank_results[n]) for n in page_rank_results]) > epsilon and num_of_iterations <= max_iterations:
        temp_page_rank_results = page_rank_results
        page_rank_results = dict.fromkeys(temp_page_rank_results.keys(), 0)
        for node in page_rank_results:
            # if not exist incoming edges update page rank to 0
            if not in_neighbors.__contains__(node):
                page_rank_results[node] = 0
            else:
                # calculate the page rank of iteration t from the neighbors nodes
                # with incoming edges to current node, using beta for "teleportation"
                for neighbor in in_neighbors[node]:
                    page_rank_results[node] += beta * (temp_page_rank_results[neighbor] / out_degree[neighbor])
        s = sum(page_rank_results.values())
        # re-insert the leaked page rank to each node in the network
        for node in page_rank_results:
            page_rank_results[node] += (1 - s) / num_of_nodes
        num_of_iterations += 1

    activate = True


# return the page rank of the node with key node_name
def get_PageRank(node_name):
    global page_rank_results
    global activate

    if activate and page_rank_results.__contains__(node_name):
        return page_rank_results[node_name]
    else:
        return -1


# return a list of the top n nodes with the highest page rank
def get_top_nodes(n):
    global page_rank_results
    global activate
    if activate and n > 0:
        return [(k, v) for k, v in (dict(sorted(page_rank_results.items(), key=operator.itemgetter(1), reverse=True)[:n])).items()]
    else:
        return []


# return a list of all nodes page rank sorted from the highest to the lowest page rank value
def get_all_PageRank():
    global page_rank_results
    global activate
    if activate:
        return [(k, v) for k, v in (dict(sorted(page_rank_results.items(), key=operator.itemgetter(1), reverse=True))).items()]
    else:
        return []


if __name__ == '__main__':
    path = input("Type path to valid csv file of directed network to load:\n")
    while not os.path.isfile(path):
        print("Error - No valid csv file, please try again\n")
        path = input("Type path to valid csv file of directed network to load:\n")

    print("load graph from path " + path + "...\n")
    load_graph(path)

    while(True):
        command = input("For running calculate_page_rank() function please type 1,\n"
                        "For running get_PageRank(node_name) function please type 2,\n"
                        "For running get_top_nodes(n) function please type 3,\n"
                        "For running get_all_PageRank() function please type 4,\n"
                        "For exit please type 5\n")

        while command is not "1" and command is not "2" and command is not "3" and command is not "4" and command is not "5":
            print("Error - No valid command, please try again\n")
            command = input("For running calculate_page_rank() function please type 1,\n"
                            "For running get_PageRank(node_name) function please type 2,\n"
                            "For running get_top_nodes(n) function please type 3,\n"
                            "For running get_all_PageRank() function please type 4,\n"
                            "For exit please type 5\n")

        if command is "1":
            print("Calculate page rank...")
            calculate_page_rank()
            print("\n")

        if command is "2":
            node_name = input("Please type node name:")
            print(get_PageRank(node_name))
            print("\n")

        if command is "3":
            n = input("Please type number of nodes:")
            num = int(n)
            page_rank_list = get_top_nodes(num)
            if len(page_rank_list) is 0:
                print([])
            else:
                for val in page_rank_list:
                    print(val)
            print("\n")

        if command is "4":
            page_rank_list = get_all_PageRank()
            if len(page_rank_list) is 0:
                print([])
            else:
                for val in page_rank_list:
                    print(val)
            print("\n")

        if command is "5":
            exit(0)
