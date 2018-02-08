#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from DAPICourseworkLibrary import *
from numpy import *
from math import log

################################## COURSEWORK 1 ######################################

############# QUESTION 1
# Function to compute the prior distribution of the variable root from the data set
def Prior(theData, root, noStates):
    prior = zeros((noStates[root]), float)
    
    for dataPoint in theData:
        
        # we only care about the first state of the first variable
        varRootState = dataPoint[root]

        # increment the data point's root var state
        prior[varRootState] += 1

    # get number of data points
    noDataPoints = len(theData)
    
    # now divide through by the number of data points to normalize the distribution
    prior /= noDataPoints
    
    return prior



############# QUESTION 2
# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = zeros((noStates[varC], noStates[varP]), float)

    # used to store the number of occurences of each varP state
    varPStateCounts = [0] * noStates[varP]
    
    for dataPoint in theData:
        
        # get state of varC and varP of the current data point
        varCState = dataPoint[varC]
        varPState = dataPoint[varP]

        # increment varPState
        varPStateCounts[varPState] += 1

        # increment cPT
        cPT[varCState][varPState] += 1

    for varPState in range(noStates[varP]):
        
        # divide each column of cPT by the number of occurrences othe varP of that column
        cPT[:,varPState] /= varPStateCounts[varPState]
        
    return cPT



############# QUESTION 3
# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = zeros((noStates[varRow], noStates[varCol]), float)
    
    for dataPoint in theData:

        #get state of varRow and varCol
        varRowState = dataPoint[varRow]
        varColState = dataPoint[varCol]

        # update jPT
        jPT[varRowState][varColState] += 1
    
    # get number of data points
    noDataPoints = len(theData)
    
    # divide through by the number of data points
    jPT /= noDataPoints
    
    return jPT



############# QUESTION 4
# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):

    # create a copy
    cpt = copy(aJPT)
    
    # get number of columns
    noColumns = cpt.shape[1]

    for i in range(noColumns):

        # get sum of current column
        columnSum = numpy.sum(cpt[:,i])

        # divive column through its sum
        cpt[:, i] /= columnSum
        
    return cpt




############# QUESTION 5
# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes): 
    rootPdf = zeros((naiveBayes[0].shape[0]), float)

    # get number of variables
    noVariables = len(naiveBayes)
    
    # get number of states of root
    noRootStates = len(naiveBayes[0])

    # iterage through the states of root
    for rootState in range (noRootStates):
        # calculate P(Root = rootState)
        varRootP = naiveBayes[0][rootState]

        # set the pdf for current state equal to P(Root = rootState)
        rootPdf[rootState] = varRootP

        # iterate through all non-root vars
        for varIndex in range(1, noVariables):
            varState = theQuery[varIndex - 1]
            rootPdf[rootState] *= naiveBayes[varIndex][varState][rootState]

    # now normalize
    rootPdf /= numpy.sum(rootPdf)
    
    return rootPdf



################################## COURSEWORK 2 ######################################


############# QUESTION 1
# Calculate the mutual information from the joint probability table of two variables
def MutualInformation(jP):
    mi = 0.0

    # get number of rows and columns
    noRow = jP.shape[0]
    noCol = jP.shape[1]
    
    # Compute P(Ai) & P(Bi) by marginalization
    varRowPs = jP.sum(1)
    varColPs = jP.sum(0)

    for row in range(noRow):
        for col in range(noCol):

            # get P(Ai & Bi)
            joint_p = jP[row][col]

            if (joint_p):
            
                # get P(Ai) & P(Bi)
                varRowP = varRowPs[row]
                varColP = varColPs[col]

                # add the information for this Ai, Bi pair
                mi += joint_p*log(joint_p/(varRowP*varColP), 2)

    return mi


############# QUESTION 2
# construct a dependency matrix for all the variables
def DependencyMatrix(theData, noVariables, noStates):
    MIMatrix = zeros((noVariables,noVariables))

    for i in range(noVariables):
        for j in range(i + 1, noVariables):
            
            # get the joint probability table for vars i and j
            jpt = JPT(theData, i, j, noStates)

            # compute mutual information
            mutual_information = MutualInformation(jpt)

            # add the mutual_information to the matrix
            MIMatrix[i][j] = mutual_information
    
    return MIMatrix


############# QUESTION 3
# Function to compute an ordered list of dependencies 
def DependencyList(depMatrix):
    depList=[]

    # find num of vars
    noVariables = depMatrix.shape[0]
    
    for i in range(noVariables):
        for j in range(i + 1, noVariables):

            # add [dependency, node1, node1] to the list
            depList.append([depMatrix[i][j], i, j])
        
    # sort by dependency
    depList.sort(key = lambda (dependency, node1, node2): dependency, reverse = True)

    return array(depList)


############# QUESTION 4

# Functions implementing the spanning tree algorithm
def SpanningTreeAlgorithm(depList, noVariables):
    # will be used to store the spanningTree
    spanningTree = []

    # we will build a graph as we go, always checking that we are not adding loops
    graph = {}
    
    for dependency, node1, node2 in depList:

        # we only add an arc if it does not create a loop
        if not find_path(graph, node1, node2):

            # update graph
            if node1 in graph:
                graph[node1].append(node2)
            else:
                graph[node1] = [node2]

            if node2 in graph:
                graph[node2].append(node1)
            else:
                graph[node2] = [node1]
                            
            spanningTree.append([dependency, node1, node2])
    
    return array(spanningTree)


# helper function that returns true if there is a path from source to goal
def find_path(graph, source, goal):

    # used to store the breadth-first search's agenda
    agenda = [source]

    # keeps track of already visited nodes
    visited = []

    
    while len(agenda) != 0:

        # pop first node
        current_node = agenda.pop(0)

        # check if this node is in the graph
        if current_node not in graph:
            return False
        
        # make sure we have not viside this node
        if current_node in visited:
            continue

        # add node to visited
        visited.append(current_node)

        # check if current_node is goal node
        if current_node == goal:
            return True
        
        # add all its children if it is in th
        for node in graph[current_node]:
            agenda.append(node)

    return False



################################## MAIN PROGRAM ######################################

def partII():
     # we will write to this file
    output_filename = "output/DAPIResults02.txt"

    # we will take our input from this file
    input_filename = "data/HepatitisC.txt"

    noVariables, noRoots, noStates, noDataPoints, datain = ReadFile(input_filename)
    theData = array(datain)
    
    ###### 1. A title giving my group number (just me)
    AppendString(output_filename, "Coursework Part 2 Results by Malon AZRIA\n")

    ###### 2. The dependency matrix for the HepatitisC data set
    dep_matrix = DependencyMatrix(theData, noVariables, noStates)
    AppendString(output_filename,"The dependency matrix for the HepatitisC data set")
    AppendArray(output_filename, dep_matrix)
    
    ###### 3. The dependency list for the HepatitisC data set
    dep_list = DependencyList(dep_matrix)
    AppendString(output_filename,"The dependency list for the HepatitisC data set")
    AppendArray(output_filename, dep_list)

    ###### 4. The spanning tree found for the HepatitisC data set
    spanning_tree = SpanningTreeAlgorithm(dep_list, noVariables)
    AppendString(output_filename,"The spanning tree found for the HepatitisC data set")
    AppendArray(output_filename, spanning_tree)
    
    
partII()
