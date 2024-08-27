
from apgl.util.Parameter import Parameter
from apgl.graph.AbstractMatrixGraph import AbstractMatrixGraph
from apgl.generator.AbstractGraphGenerator import AbstractGraphGenerator
import numpy
import scipy

class ErdosRenyiGenerator(AbstractGraphGenerator):
    '''
    A class which generates graphs according to the Erdos-Renyi Model. At construction time, it
    takes an empty graph, for which it creates edges.
    '''
    def __init__(self, p, selfEdges=False):
        '''
        Create an Erdos-Renyi generator with edge probability p. For all pairs
        of vertices in the graph, an edge exists with probability p. If selfEdges
        is False edges from a vertex to itself are not created.

        :param p: the probability of an edge
        :type p: :class:`float`

        :param selfEdges: whether to allow self edges
        :type selfEdges: :class:`bool` 
        '''
        self.setP(p)
        self.setSelfEdges(selfEdges)

    def setSelfEdges(self, selfEdges):
        """
        :param selfEdges: whether to allow self edges
        :type selfEdges: :class:`bool`
        """
        Parameter.checkBoolean(selfEdges)
        self.selfEdges = selfEdges

    def setP(self, p):
        '''
        :param p: the probability of an edge
        :type p: :class:`float`
        '''
        Parameter.checkFloat(p, 0.0, 1.0)
        self.p = p 

    def generate(self, graph, requireEmpty=True):
        '''
        Create an Erdos-Renyi graph from the given input graph. 

        :param graph: an empty graph to populate with edges
        :type graph: :class:`apgl.graph.AbstractMatrixGraph`

        :param requireEmpty: whether to allow non empty graphs. 
        :type requireEmpty: :class:`bool`

        :returns: The modified input graph. 
        '''
        Parameter.checkClass(graph, AbstractMatrixGraph)
        if requireEmpty and graph.getNumEdges()!= 0:
            raise ValueError("Graph must have no edges")
        
        numVertices = graph.getNumVertices()
        #This function seems slightly weird- sometimes the last cols are empty 
        W = scipy.sparse.rand(numVertices, numVertices, self.p)
        W = W/W
        
        if graph.isUndirected():
            diagW = W.diagonal()
            W = scipy.sparse.triu(W, 1)
            W = W + W.T

            if self.selfEdges:
                W.setdiag(diagW)

        if not self.selfEdges:
            W.setdiag(numpy.zeros(numVertices))

        if not requireEmpty:
            W = W + graph.getWeightMatrix()

        graph.setWeightMatrix(W)
            
        return graph
        
    def generate2(self, graph, requireEmpty=True): 
        """
        An alternative way of generating random edges which might work better 
        than generate. 
        """
        Parameter.checkClass(graph, AbstractMatrixGraph)
        if requireEmpty and graph.getNumEdges()!= 0:
            raise ValueError("Graph must have no edges")
            
        numVertices = graph.getNumVertices()
        W = numpy.random.rand(numVertices, numVertices) < self.p
        W = numpy.array(W, numpy.int)
        
        if graph.isUndirected():
            diagW = numpy.diag(W)
            W = numpy.triu(W, 1)
            W = W + W.T
  
        
        graph.setWeightMatrix(scipy.sparse.csr_matrix(W, dtype=numpy.float))
            
        return graph
        
            
    def clusteringCoefficient(self):
        '''
        Returns the clustering coefficient for the generator.
        ''' 
        return p

    def __str__(self):
        return "ErdosRenyiGenerator:p="+str(self.p)
    