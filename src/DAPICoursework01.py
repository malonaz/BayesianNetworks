#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from DAPICourseworkLibrary import *
from numpy import *

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


################################## MAIN PROGRAM ######################################
def partI():
    # we will write to this file
    output_filename = "output/DAPIResults01.txt"

    # we will take our input from this file
    input_filename = "data/Neurones.txt"
    
    # read file
    noVariables, noRoots, noStates, noDataPoints, datain = ReadFile(input_filename)
    theData = array(datain)
    
    ###### 1. A title giving my group number (just me)
    AppendString(output_filename, "Coursework Part 1 Results by Malon AZRIA\n")
    
    ###### 2. The prior probability distribution of node 0 in the data set
    AppendString(output_filename,"The prior probability of node 0")
    prior = Prior(theData, 0, noStates)
    AppendList(output_filename, prior)
    
    ###### 3. The conditional probability matrix P(2|0) calculated from the data
    AppendString(output_filename, "The conditional probability of P(2|0)")
    cpt = CPT(theData, 2, 0, noStates)
    AppendArray(output_filename, cpt)
    
    ###### 4. The joint probability matrix P(2&0) calculated from the data
    AppendString(output_filename, "The joint probability of P(2&0)")
    jpt = JPT(theData, 2, 0, noStates)
    AppendArray(output_filename, jpt)
    
    ###### 5. The conditional probability matrix P(2|0) calculated from the joint probability matrix P(2&0)
    AppendString(output_filename, "The joint probability of P(2|0) calculated from the joint probability matrix P(2&0)")
    cpt = JPT2CPT(jpt)
    AppendArray(output_filename, cpt)
    
    ###### 6. The results of queries [4, 0, 0, 0, 5] and [6, 5, 2, 5, 5] on the naive network
    # Construct naiveBayes, starting by adding prior to it.
    naiveBayes = [prior]
    
    # iterate through all variables except root
    for i in range(1, noVariables):
        # append the P(i|0) to naiveBayes
        naiveBayes.append(CPT(theData, i, 0, noStates))
        
    # carry out first query
    AppendString(output_filename, "The results of query [4, 0, 0, 0, 5] on the naive network")
    rootpdf = Query([4, 0, 0, 0, 5], naiveBayes)
    AppendList(output_filename, rootpdf)
    
    # carry out second query
    AppendString(output_filename, "The results of query [6, 5, 2, 5, 5] on the naive network")
    rootpdf = Query([6, 5, 2, 5, 5], naiveBayes)
    AppendList(output_filename, rootpdf)

    
partI()    

