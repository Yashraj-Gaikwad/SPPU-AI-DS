'''
Implement Page Rank Algorithm. (Use python or beautiful soup for implementation).

viva questions
1) explain page rank algo
'''

# Import Libraries
import numpy as np

# Define Class
class PageRank:
    def __init__(self, graph, damping_factor=0.85, max_iterations=100, tolerance=1.0e-6):
        self.graph = graph  # adjacency matrix for directed graph
        self.damping_factor = damping_factor  # the probability of continuing to follow links   
        self.max_iterations = max_iterations
        self.tolerance = tolerance  # the convergence criterion
        self.num_nodes = len(graph)
        self.page_rank = np.ones(self.num_nodes) / self.num_nodes  # Initialize PageRank

    def calculate_page_rank(self):

        # Runs for max iterations
        for _ in range(self.max_iterations):
            # Initialize all nodes to 0
            new_page_rank = np.zeros(self.num_nodes)
            # Calculate pagerank for each node i
            for i in range(self.num_nodes):
                # Check link between nodes
                for j in range(self.num_nodes):
                    if self.graph[j][i] > 0:  # If there is a link from j to i
                        # update rank for node i
                        # takes current rank of node j and divides it by total no. of links node j has
                        new_page_rank[i] += (self.page_rank[j] / np.sum(self.graph[j]))  # Distribute rank

            # Apply damping factor
            # damping factor = 0.85, means 15% of the rank is distributed to all nodes equally
            new_page_rank = (1 - self.damping_factor) / self.num_nodes + self.damping_factor * new_page_rank
            
            # Check for convergence
            # checks L1 norm also known as Manhattan Norm
            if np.linalg.norm(new_page_rank - self.page_rank, ord=1) < self.tolerance:
                break
            
            # prepare for next iteration
            self.page_rank = new_page_rank
        
        # return the page rank
        return self.page_rank

# Example usage
if __name__ == "__main__":
    # Define a simple graph as an adjacency matrix
    # Example: A graph with 4 nodes (0, 1, 2, 3)
    graph = np.array([[0, 1, 1, 0],  # Node 0 links to Node 1 and Node 2
                      [0, 0, 1, 1],  # Node 1 links to Node 2 and Node 3
                      [0, 0, 0, 1],  # Node 2 links to Node 3
                      [1, 0, 0, 0]]) # Node 3 links to Node 0

    # Create an instance of class
    pagerank = PageRank(graph)
    # call function
    ranks = pagerank.calculate_page_rank()
    
    print("PageRank Values:", ranks)

'''
Output - 
PageRank Values: [0.29721007 0.16381415 0.23343507 0.30554071]

Node 3 - most imp
Node 1 - least imp
'''