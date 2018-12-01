"""
copyright 2018 Preston R. Labig
"""
import random
from math import inf as Infinity
from copy import deepcopy
from math import ceil, floor


class Optimizer:
    def __init__(self, minimumIterationsToRun: int = 100, earlyStoppingIters: int = 10):
        self.numIterationsCompleted: int = 0
        self.bestScore: float = Infinity
        self.bestArtifact = None
        self.bestGenes = None
        self.numPossibleSolutions: int = 0
        self.requestedGenes: dict = {}
        self.origIndividual: Individual = None
        self.curIndividual: Individual = None
        self.curTemperature: int = 100
        self.temperatureStepSize: int = self.curTemperature / minimumIterationsToRun
        self.earlyStoppingEnabled: bool = False
        self.earlyStoppingIters: int = earlyStoppingIters
        self.earlyStoppingUnimprovedIterCount: int = 0
        self.earlyStoppingNeedToStop: bool = False

    def addGene(self, label: str, gene: object):
        """
        Adds an optimizable parameter to the Optimizer instance.

        Example:
        myOptimizer = Optimizer(10, 100)
        myOptimizer.addGene("my_parameter_to_optimize", GeneInt(1, 1000))

        :param label: A string that acts as the key to the value of the new Gene
        :param gene: An instance of the type of Gene you want to optimize
        :return: Nothing
        """
        if label is None:
            raise Exception("No label passed to addGene")
        if gene is None:
            raise Exception("No gene passed to addGene")
        self.requestedGenes[label] = gene

    def getGeneValue(self, label: str):
        """
        Retrieves the value of the Gene stored by the key "label".

        Example:
        myOptimizer = Optimizer(10, 100)
        myOptimizer.addGene("my_parameter_to_optimize", GeneInt(1, 1000))
        myOptimizer.startTraining()
        value = myOptimizer.getGeneValue("my_parameter_to_optimize")

        :param label: The string based key that access the value of the Gene
        :return: The current value of the Gene
        """
        if label is None:
            raise Exception("No label passed to getGeneValue")
        return self.curIndividual.genes[label].value

    def startTraining(self):
        self.numPossibleSolutions = 1
        for geneKey in self.requestedGenes:
            curGene = self.requestedGenes[geneKey]
            numParams = curGene.getNumParameters()
            self.numPossibleSolutions *= numParams

        print("Number of Possible Solutions: " + str(self.numPossibleSolutions))

        newIndividual = Individual()
        for key in self.requestedGenes:
            origGene = self.requestedGenes[key]
            newGene = deepcopy(origGene)
            newGene.mutate(self)
            newIndividual.genes[key] = newGene
        self.curIndividual = newIndividual

    def next(self, inputScore: float = Infinity, userArtifact: object = None):
        # Early stopping
        if self.earlyStoppingNeedToStop:
            return self.earlyStoppingNeedToStop, self.numIterationsCompleted, self.bestScore, self.bestArtifact

        self.numIterationsCompleted += 1

        # Set temperature. Linear.
        self.curTemperature -= self.temperatureStepSize
        if self.curTemperature < 0:
            self.curTemperature = 0


        # Scoring and saving of scores, artifacts, etc.
        scoreImproved = False
        self.curIndividual.score = inputScore
        if inputScore < self.bestScore:
            scoreImproved = True
            self.bestScore = inputScore
            self.bestGenes = self.curIndividual.genes
            if userArtifact is not None:
                self.bestArtifact = userArtifact

        #Early stopping
        if self.earlyStoppingEnabled:
            if scoreImproved:
                self.earlyStoppingUnimprovedIterCount = 0
            else:
                self.earlyStoppingUnimprovedIterCount += 1
                if self.earlyStoppingUnimprovedIterCount >= self.earlyStoppingIters:
                    self.earlyStoppingNeedToStop = True
                    return self.earlyStoppingNeedToStop, self.numIterationsCompleted, self.bestScore, self.bestArtifact
        #Enable early stopping
        if self.curTemperature <= 0 and self.earlyStoppingEnabled is False:
            self.earlyStoppingEnabled = True

        # The first iteration is a special case.  We have no loss to compare to, so just mutate the individual.
        if self.numIterationsCompleted == 1:
            self.origIndividual = deepcopy(self.curIndividual)
            self.mutateIndividual(self.curIndividual)
            return self.earlyStoppingNeedToStop, self.numIterationsCompleted, self.bestScore, self.bestArtifact

        # Determine which solution to use
        curScore = self.curIndividual.score
        origScore = self.origIndividual.score
        # Original solution was better.  Based on temperature and normalized difference determine if we should keep the worse score.
        if origScore <= curScore:
            normalizedCurScore = 100
            normalizedOrigScore = (origScore / curScore) * 100
            normalizedDifference = normalizedCurScore - normalizedOrigScore
            chanceOfBeingKept = self.curTemperature - normalizedDifference
            randomNum = random.uniform(0, 100)
            if randomNum < chanceOfBeingKept:
                self.origIndividual = deepcopy(self.curIndividual)
            else:
                self.curIndividual = deepcopy(self.origIndividual)
        elif curScore < origScore:
            self.origIndividual = deepcopy(self.curIndividual)

        # All of the below is ran regardless of which solution was chosen
        self.mutateIndividual(self.curIndividual)
        return self.earlyStoppingNeedToStop, self.numIterationsCompleted, self.bestScore, self.bestArtifact

    def mutateIndividual(self, individual):
        """
        NOT FOR EXTERNAL USE.
        """
        numGenesInIndividual = len(individual.genes)
        adjustedTemperature = self.curTemperature
        if adjustedTemperature < 1:
            adjustedTemperature = 1
        adjustedNumGenesInIndividual = ceil(numGenesInIndividual * (adjustedTemperature / 100))
        if adjustedNumGenesInIndividual > numGenesInIndividual:
            adjustedNumGenesInIndividual = numGenesInIndividual
        numGenesToMutate = random.randint(1, adjustedNumGenesInIndividual)
        listOfGenesToMutate = random.sample(list(individual.genes.values()), numGenesToMutate)
        for gene in listOfGenesToMutate:
            gene.mutate(self)

    def getBestParameters(self):
        """
        Returns a dictionary holding the keys and values of the best solution found so far.
        The keys are the same as the ones used to create the genes.
        :return: A dictionary with the best values found so far
        """
        bestGenesDict = deepcopy(self.bestGenes)
        dictOfValues = {}
        for key in bestGenesDict:
            value = bestGenesDict[key].value
            dictOfValues[key] = value
        return dictOfValues


class Individual:
    """
    NOT FOR EXTERNAL USE.
    """

    def __init__(self):
        self.genes = {}
        self.score = Infinity

    def getHash(self):
        stringToHash = ""
        for key in self.genes:
            curGene = self.genes[key]
            hashableValue = curGene.getHashableValue()
            stringToHash += hashableValue
        hashOfValues = hash(stringToHash)
        return hashOfValues


class GeneBool:
    """
    Gene that optimizes a boolean value.
    """

    def __init__(self):
        self.value = random.choice([True, False])

    def mutate(self, optimizer: Optimizer):
        self.value = random.choice([True, False])

    def getHashableValue(self) -> str:
        return str(self.value)

    def getNumParameters(self):
        return 2


class GeneInt:
    """
    Gene that optimizes an integer value inside a range.
    The min and max values are inclusive.
    """

    def __init__(self, min: int = 0, max: int = 100):
        self.min = min
        self.max = max
        self.value = random.randint(self.min, self.max)

    def mutate(self, optimizer: Optimizer):
        percentageOfRangeToSampleFrom = 0
        if optimizer.curTemperature < 5:
            percentageOfRangeToSampleFrom = 5 / 100
        else:
            percentageOfRangeToSampleFrom = optimizer.curTemperature / 100
        totalParameters = self.getNumParameters()
        samplingSize = ceil(totalParameters * percentageOfRangeToSampleFrom)
        samplingSizeForOneSide = ceil(samplingSize / 2)
        lowerValue = self.value - samplingSizeForOneSide
        upperValue = self.value + samplingSizeForOneSide
        possibleValue = 0
        while True:
            possibleValue = random.randint(lowerValue, upperValue)
            if possibleValue >= self.min and possibleValue <= self.max:
                break
        self.value = possibleValue

    def getHashableValue(self) -> str:
        return str(self.value)

    def getNumParameters(self):
        return (self.max - self.min) + 1


class GeneFloat:
    """
    Gene that optimizes a decimal value between two integer values.
    The min and max values are inclusive.
    """

    def __init__(self, min: int = 0, max: int = 100, numDecimalPlaces: int = 1):
        self.min = min
        self.max = max
        self.numDecimalPlaces = numDecimalPlaces
        self.value = round(random.uniform(self.min, self.max),
                           self.numDecimalPlaces)  # The rounding makes the max value inclusive

    def mutate(self, optimizer: Optimizer):
        percentageOfRangeToSampleFrom = 0
        if optimizer.curTemperature < 5:
            percentageOfRangeToSampleFrom = 5 / 100
        else:
            percentageOfRangeToSampleFrom = optimizer.curTemperature / 100
        totalParameters = self.getNumParameters()
        samplingSize = ceil(totalParameters * percentageOfRangeToSampleFrom)
        samplingSizeForOneSide = ceil(samplingSize / 2)
        adjustedFloatToLargeValue = self.value * (10 ** self.numDecimalPlaces)
        lowerValue = (adjustedFloatToLargeValue - samplingSizeForOneSide) / (10 ** self.numDecimalPlaces)
        upperValue = (adjustedFloatToLargeValue + samplingSizeForOneSide) / (10 ** self.numDecimalPlaces)
        possibleValue = 0
        while True:
            possibleValue = round(random.uniform(lowerValue, upperValue), self.numDecimalPlaces)
            if possibleValue >= self.min and possibleValue <= self.max:
                break
        self.value = possibleValue

    def getHashableValue(self) -> str:
        return str(self.value)

    def getNumParameters(self):
        numParams = ((self.max * (10 ** self.numDecimalPlaces)) - (self.min * (10 ** self.numDecimalPlaces))) + 1
        return numParams


class GeneChoice:
    """
    Gene that optimizes a choice.

    It will take a list of anything you want, and then optimize the selection of an object from that list.

    Example:
    optimizer = Optimizer(100)
    optimizer.addGene("choice_1", GeneChoice(["blah", "something", "pancakes"]))
    """

    def __init__(self, choices: list):
        self.choices = choices
        self.value = random.choice(self.choices)

    def mutate(self, optimizer: Optimizer):
        self.value = random.choice(self.choices)

    def getHashableValue(self) -> str:
        indexOfCurrentValue = self.choices.index(self.value)
        return str(indexOfCurrentValue)

    def getNumParameters(self):
        return len(self.choices)

# optim = Optimizer(minimumIterationsToRun=10000, earlyStoppingIters=1000)
# optim.addGene("gene_1", GeneBool())
# optim.addGene("gene_2", GeneInt(0, 100))
# optim.addGene("gene_3", GeneFloat(0, 2, 8))
# optim.addGene("gene_4", GeneChoice(["Relu", "Linear", "Conv2D", "Dense", "MaxPool2D"]))
# optim.startTraining()
# while True:
#     print("Bool: " + str(optim.getGeneValue("gene_1")))
#     print("Integer: " + str(optim.getGeneValue("gene_2")))
#     print("Float: " + str(optim.getGeneValue("gene_3")))
#     print("Choice: " + str(optim.getGeneValue("gene_4")))
#     print("*******************************")
#     print("TEMPERATURE: " + str(optim.curTemperature))
#     print("*******************************")
#     score = random.uniform(0, 100)
#     optimizingComplete, completedIterations, bestScore, artifact = optim.next(score)
#     if optimizingComplete:
#         print("Optimization Finished!")
#         break
