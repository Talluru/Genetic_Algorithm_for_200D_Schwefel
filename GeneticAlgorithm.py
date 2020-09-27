#the intial framework for a real-valued GA
#authors: Gowtham Talluru, Charles Nicholson

#need some python libraries
import copy
import math
from random import Random, random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#to setup a random number generator, we will specify a "seed" value
seed = 5113
myPRNG = Random(seed)

dimensions = 200   #set dimensions for Schwefel Function search space
lowerBound = -500  #bounds for Schwefel Function search space
upperBound = 500   #bounds for Schwefel Function search space

#you may change anything below this line that you wish too -----------------------------------------------------------------

#Student name(s):
#Date:
def GA(populationSize, Generations, elite_group,
       crossOverRate, mutateRate, tournamentK, maxNonImproveGen):
    print("Pop_size:", populationSize)
    print("Generation:", Generations)
    print("elite_group:", elite_group)
    print("crossOverRate:", crossOverRate)
    print("mutateRate:", mutateRate)
    #create an continuous valued chromosome
    # d dimensions
    def createChromosome(d, lBnd, uBnd):  # d dimensions lBnd, uBnd lower n Upper bound
        x = []
        for i in range(d):
            x.append(myPRNG.uniform(lBnd,uBnd))   #creating a randomly located solution
        return x

    def initializePopulation(): #n is size of population; d is dimensions of chromosome
        population = []            #Population
        populationFitness = []     #Population fitness

        for i in range(populationSize):
            population.append(createChromosome(dimensions,lowerBound, upperBound)) #calling func createChromosome
            populationFitness.append(evaluate(population[i])) #calling func evaluate

        tempZip = zip(population, populationFitness)  #Merging population and fitness
        popVals = sorted(tempZip, key=lambda tempZip: tempZip[1])
        return popVals

    #implement a linear crossover
    def crossover(x1,x2):

        d = len(x1) #dimensions of solution

        #choose crossover point

        #we will choose the smaller of the two [0:crossOverPt] and [crossOverPt:d] to be unchanged
        #the other portion be linear combo of the parents

        crossOverPt = myPRNG.randint(1,d-1) #notice I choose the crossover point so that at least 1 element of parent is copied

        beta = myPRNG.uniform(0, 1)  #random number between 0 and 1


        #note: using numpy allows us to treat the lists as vectors
        #here we create the linear combination of the soltuions
        new1 = list(np.array(x1) - beta*(np.array(x1)-np.array(x2)))
        new2 = list(np.array(x2) + beta*(np.array(x1)-np.array(x2)))

        #the crossover is then performed between the original solutions "x1" and "x2" and the "new1" and "new2" solutions
        if crossOverPt<d/2:
            offspring1 = x1[0:crossOverPt] + new1[crossOverPt:d]  #note the "+" operator concatenates lists
            offspring2 = x2[0:crossOverPt] + new2[crossOverPt:d]
        else:
            offspring1 = new1[0:crossOverPt] + x1[crossOverPt:d]
            offspring2 = new2[0:crossOverPt] + x2[crossOverPt:d]

        return offspring1, offspring2  #two offspring are returned

    #function to evaluate the Schwefel Function for d dimensions
    def evaluate(x):
        val = 0
        d = len(x)
        #print("*******")
        #print(x)
        #print(d)
        for i in range(d):

            val = val + x[i]*math.sin(math.sqrt(abs(x[i])))

        #print(418.9829*d)
        #print(val)
        val = 418.9829*d - val
        #print(val)
        #print("*******")
        return val

    #function to provide the rank order of fitness values in a list
    #not currently used in the algorithm, but provided in case you want to...
    def rankOrder(anyList):

        rankOrdered = [0] * len(anyList)
        for i, x in enumerate(sorted(range(len(anyList)), key=lambda y: anyList[y])):
            rankOrdered[x] = i

        return rankOrdered

    #performs tournament selection; k chromosomes are selected (with repeats allowed) and the best advances to the mating pool
    #function returns the mating pool with size equal to the initial population
    def tournamentSelection(pop,k):

        #randomly select k chromosomes; the best joins the mating pool
        matingPool = []

        while len(matingPool)<populationSize:

            ids = [myPRNG.randint(0,populationSize-1) for i in range(k)]
            competingIndividuals = [pop[i][1] for i in ids]
            bestID=ids[competingIndividuals.index(min(competingIndividuals))]
            matingPool.append(pop[bestID][0])

        return matingPool

    #function to mutate solutions
    def mutate(x):

        xx=x[:]     #Create different object
        altIndex= myPRNG.randint(0, dimensions-1) # Random Index
        a = myPRNG.randrange(lowerBound, upperBound) # Random number
        xx[altIndex] = a
        #if(abs(x[altIndex])<400):
        #    a=myPRNG.choice([1,-1])
        #    x[altIndex]= x[altIndex] + a*10
        return xx

    def breeding(matingPool):
        children = []
        childrenFitness = []
        for i in range(0,populationSize-1,2):

            if (myPRNG.random() < crossOverRate):    #Cross over
                child1,child2=crossover(matingPool[i],matingPool[i+1])
            else:
                child1=matingPool[i]  #New offspring same as parents
                child2=matingPool[i+1]

            if (myPRNG.random() < mutateRate):  #Mutation rate
                child1= mutate(child1)
                child2= mutate(child2)

            children.append(child1)
            children.append(child2)

            childrenFitness.append(evaluate(child1))
            childrenFitness.append(evaluate(child2))

        tempZip = zip(children, childrenFitness)
        popVals = sorted(tempZip, key=lambda tempZip: tempZip[1])

        #the return object is a sorted list of tuples:
        #the first element of the tuple is the chromosome; the second element is the fitness value
        #for example:  popVals[0] is represents the best individual in the population
        #popVals[0] for a 2D problem might be  ([-70.2, 426.1], 483.3)  -- chromosome is the list [-70.2, 426.1] and the fitness is 483.3

        return popVals

    #insertion step
    def insert(pop,kids, elite):
        output=[]
        output[0:elite]=(pop[0:elite])
        output[elite: populationSize]=kids[0:(populationSize-elite)]

        output = sorted(output, key=lambda output: output[1])

        #replacing the previous generation completely...  probably a bad idea -- please implement some type of elitism
        return output

    #perform a simple summary on the population: returns the best chromosome fitness, the average population fitness, and the variance of the population fitness
    def summaryFitness(pop):
        a=np.array(list(zip(*pop))[1])
        return np.min(a), np.mean(a), np.var(a)

    #the best solution should always be the first element... if I coded everything correctly...
    def bestSolutionInPopulation(pop):
        print (pop[0])

    def plot(pop, j):
        xxx = []
        yyy = []
        zzz = []
        color = ["red", "green", "blue", "yellow","black", "orange", "violet"]
        mark = ["o", "v", "8", "s", "p", "*", "D" ]
        for l in range(populationSize):

            ax.scatter(Population[l][0][0], Population[l][0][1], Population[l][1], c=color[j], marker= mark[j])
            ax.text(Population[l][0][0], Population[l][0][1], Population[l][1], '%s' % (str(j)))

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Solution Value")

    #optional: you can output results to a file

    f = open('out.txt', 'w')

    #GA main code
    sol=[]
    Population = initializePopulation()
    fig = plt.figure()
    ax = Axes3D(fig)
    count=1

    for j in range(Generations):
        if j<2:
            plot(Population, j)
        mates=tournamentSelection(Population,tournamentK)
        Offspring = breeding(mates)

        if(Offspring[0][1]==Population[0][1]):
            count= count+1
        else:
            count=0

        Population = insert(Population, Offspring, elite_group)

        minVal,meanVal,varVal=summaryFitness(Population)
        f.write(str(minVal) + " " + str(meanVal) + " " + str(varVal) + "\n")
        #bestSolutionInPopulation(Population)
        sol.append([j, Population[0][1]])
        print(Population[0][1])
        #print(evaluate(Population[0][0]))
        #print(Population)

        if(count==maxNonImproveGen):
            print("Max non improving generation reached")
            break
    f.close()

    print (summaryFitness(Population))
    bestSolutionInPopulation(Population)
    print("Generations:", j)

    return sol

dimensions=2

q1c=GA(populationSize = 30, #size of GA population
        Generations = 1000,  #number of GA generations
        elite_group = 5,
        crossOverRate = 0.8,
        mutateRate = 0.3,
        tournamentK=3,
        maxNonImproveGen= 100)

dimensions=200

q1d=GA(populationSize = 100, #size of GA population
        Generations = 10000,  #number of GA generations
        elite_group = 10,
        crossOverRate = 0.9,
        mutateRate = 0.4,
        tournamentK=10,
        maxNonImproveGen= 1000)




plt.show()

