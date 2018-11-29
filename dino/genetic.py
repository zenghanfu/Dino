"""
copyright 2018 Preston R. Labig
"""
import random
from math import inf as Infinity
from copy import deepcopy
from math import ceil, floor


class Optimizer:
    def __init__(self, populationSize: int = 20, chanceOfMutation: int = 5):
        """
        The main interface to Dino.

        The populationSize defines how many Individuals are in each generation.
        For instance, if you specify 20, there will be 20 loops before each generation is finished.
        After each generation is finished, then the actual optimization occurs.  So, if the population
        is set to 20, and you run your loop 40 times, you will have run one optimized loop, as the first generation
        was created randomly, but the second generation was optimized from the first generation's results.

        :param populationSize: The number of solutions(Individuals) generated and tried per generation
        :param chanceOfMutation: A value between 1 and 100.  An integer value dictating the chance of a new Individual's chance of mutating
        """
        self.populationSize: int = populationSize
        self.numGenerationsCompleted: int = 0
        self.curGenerationIndividuals: list = []
        self.keptIndividuals: list = []
        self.requestedGenes: dict = {}
        self.curIndividual: Individual = None
        self.curIndividualNum: int = 0
        self.bestScore: float = Infinity
        self.bestArtifact = None
        self.bestGenes = None
        self.listOfHashes: list = []
        self.numPossibleSolutions: int = 0
        self.scoreImproved: bool = False
        self.baselineMutationRate: int = chanceOfMutation
        self.curMutationRate: int = self.baselineMutationRate

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
        """
        This starts the training preparation.

        Behind the scenes, this method prepares and creates all the Individuals
        and adds the requested Genes to the Individuals.

        Example:
        myOptimizer = Optimizer(10, 100)
        myOptimizer.addGene("my_parameter_to_optimize", GeneInt(1, 1000))
        myOptimizer.startTraining()
        value = myOptimizer.getGeneValue("my_parameter_to_optimize")

        :return: Nothing
        """
        self.numPossibleSolutions = 1
        for geneKey in self.requestedGenes:
            curGene = self.requestedGenes[geneKey]
            numParams = curGene.getNumParameters()
            self.numPossibleSolutions *= numParams
        if self.numPossibleSolutions < self.populationSize:
            raise Exception("FATAL: Your search space of " + str(
                self.numPossibleSolutions) + " possible solutions is smaller than your population size of " + str(
                self.populationSize) + ".  Either make your search space larger or decrease your population size to, at a minimum, the number of possible solutions.")

        print("Number of Possible Solutions: " + str(self.numPossibleSolutions))
        print("Generating initial pool of possible solutions...")
        for _ in range(self.populationSize):
            while True:
                newIndividual = Individual()
                for key in self.requestedGenes:
                    origGene = self.requestedGenes[key]
                    newGene = deepcopy(origGene)
                    newGene.mutate()
                    newIndividual.genes[key] = newGene
                individualHash = newIndividual.getHash()
                if individualHash not in self.listOfHashes:
                    self.listOfHashes.append(individualHash)
                    self.curGenerationIndividuals.append(newIndividual)
                    break
        self.curIndividual = self.curGenerationIndividuals[self.curIndividualNum]

    def next(self, inputScore: float = Infinity, userArtifact: object = None):
        """
        This method prepares the algorithm for the next loop.

        Call this method to prepare for the next loop.  It will store the user supplied score
        and also store an optional user artifact.  The user artifact could be, for instance,
        a Keras model.  It returns four things.  The first is a boolean indicating if the optimization has finished.
        It is True if optimization has exhausted the search space, False if there are still more solutions to try.
        The second is the number of completed generations, the third is the best score seen throughout all the loops
        since the beginning of optimization, and the fourth is the user artifact that was supplied with the best score.
        If an artifact was not stored None is returned instead.

        Example:
        myOptimizer = Optimizer(10, 100)
        myOptimizer.addGene("my_parameter_to_optimize", GeneInt(1, 1000))
        myOptimizer.startTraining()
        value = myOptimizer.getGeneValue("my_parameter_to_optimize")
        ...
        kerasModel = load_model(...)
        ...
        score = something
        trainingComplete, numCompletedGenerations, bestScore, bestArtifact = myOptimizer.next(score, kerasModel)


        :param inputScore: The score of the last optimization run
        :param userArtifact: An optional object that will be tied to the supplied score. IE: Keras model
        :return:
        bool: Returns True if training has finished
        Int: Number of completed generations
        Float: The best score seen so far
        Object: An artifact tied to the best score.  If not available, None.
        """
        self.numPossibleSolutions -= 1
        print("Possible solutions remaining: " + str(self.numPossibleSolutions))

        if inputScore < self.bestScore:
            self.scoreImproved = True
            self.bestScore = inputScore
            self.bestGenes = self.curIndividual.genes
            if userArtifact is not None:
                self.bestArtifact = userArtifact
        self.curIndividual.score = inputScore
        self.curIndividualNum += 1
        if self.curIndividualNum < self.populationSize:
            self.curIndividual = self.curGenerationIndividuals[self.curIndividualNum]
            return False, self.numGenerationsCompleted, self.bestScore, self.bestArtifact

        # If we reach this point then we have finished the generation.

        # Check if we are out of solutions
        if self.numPossibleSolutions <= 0:
            return True, self.numGenerationsCompleted, self.bestScore, self.bestArtifact

        self.curIndividualNum = 0

        # Check if there was score improvement in the last generation.  If not, up the mutation rate.
        # If there was improvement, drop the mutation rate back to baseline
        if self.scoreImproved is True:
            self.scoreImproved = False
            if self.curMutationRate != self.baselineMutationRate:
                self.curMutationRate = self.baselineMutationRate
        elif self.scoreImproved is False:
            if self.curMutationRate < 100:
                self.curMutationRate += 5
                if self.curMutationRate > 100:
                    self.curMutationRate = 100

        # Sort
        allIndividualsCopy = deepcopy(self.curGenerationIndividuals)
        self.curGenerationIndividuals.clear()
        self.keptIndividuals.extend(allIndividualsCopy)
        self.keptIndividuals.sort(key=lambda x: x.score)

        # Remove unfit Individuals, minus a few lucky ones.  Keeping a few is supposed to help increase "diversity".
        numGoodToKeep = ceil(self.populationSize * 0.25)
        numBadToKeep = ceil(self.populationSize * 0.10)
        totalToKeep = numGoodToKeep + numBadToKeep
        if totalToKeep > self.populationSize:
            numToSubtractFromTotal = totalToKeep - self.populationSize
            # Take from the "bad" pool first
            if numToSubtractFromTotal >= numBadToKeep:
                numToSubtractFromTotal -= numBadToKeep
                numBadToKeep = 0
            else:
                numBadToKeep -= numToSubtractFromTotal
            if numToSubtractFromTotal > 0:
                if numToSubtractFromTotal >= numGoodToKeep:
                    raise Exception(
                        "The per generation trim rate is too high for the current number of individuals.  This is fatal.")
                else:
                    numGoodToKeep -= numToSubtractFromTotal
                    numToSubtractFromTotal = 0
        totalToKeep = numGoodToKeep + numBadToKeep
        if totalToKeep < 2:
            raise Exception("The populationSize needs to be bigger.  Not enough Individuals left to breed.")

        indexesOfIndividualsToKeep = list(range(numGoodToKeep))
        while numBadToKeep > 0:
            badIndToKeepIndex = random.randint(numGoodToKeep,
                                               len(self.keptIndividuals) - 1)  # Subtract 1 because randint is inclusive
            if badIndToKeepIndex not in indexesOfIndividualsToKeep:
                indexesOfIndividualsToKeep.append(badIndToKeepIndex)
                numBadToKeep -= 1
        descendingListOfAllIndividualIndexes = list(range(len(self.keptIndividuals)))
        descendingListOfAllIndividualIndexes.sort(reverse=True)
        for index in descendingListOfAllIndividualIndexes:
            if index not in indexesOfIndividualsToKeep:
                del self.keptIndividuals[index]

        # Breeding section
        # Check to ensure there are enough remaining solutions before creating Individuals
        # otherwise we will stall indefinitely.
        if self.numPossibleSolutions < self.populationSize:
            self.populationSize = self.numPossibleSolutions
        # Set weights for breeding
        numOfKeptIndividuals = len(self.keptIndividuals)
        multiplier = 100
        for index in range(numOfKeptIndividuals):
            chanceToBreed = ceil(((numOfKeptIndividuals - index) / numOfKeptIndividuals) * multiplier)
            if chanceToBreed > 95:
                chanceToBreed = 95
            if chanceToBreed < 5:
                chanceToBreed = 5
            self.keptIndividuals[index].chanceToBreed = chanceToBreed

        numIndividualsToCreate = self.populationSize
        while numIndividualsToCreate > 0:
            motherIndex = None
            fatherIndex = None
            while True:
                possibleMotherIndex = random.randint(0,
                                                     numOfKeptIndividuals - 1)  # Subtract 1 because randint is inclusive
                mothersChanceToBreed = self.keptIndividuals[possibleMotherIndex].chanceToBreed
                randomNum = random.randint(0, 99)
                if randomNum < mothersChanceToBreed:
                    motherIndex = possibleMotherIndex
                    break
            while True:
                possiblefatherIndex = random.randint(0,
                                                     numOfKeptIndividuals - 1)  # Subtract 1 because randint is inclusive
                fathersChanceToBreed = self.keptIndividuals[possiblefatherIndex].chanceToBreed
                randomNum = random.randint(0, 99)
                if randomNum < fathersChanceToBreed:
                    fatherIndex = possiblefatherIndex
                    break
            if motherIndex is not fatherIndex:
                newIndividual = self.breedIndividuals(self.keptIndividuals[motherIndex],
                                                      self.keptIndividuals[fatherIndex])
                newIndividualHash = newIndividual.getHash()
                if newIndividualHash not in self.listOfHashes:
                    self.listOfHashes.append(newIndividualHash)
                    self.curGenerationIndividuals.append(newIndividual)
                    numIndividualsToCreate -= 1
        self.curIndividual = self.curGenerationIndividuals[0]
        self.numGenerationsCompleted += 1
        return False, self.numGenerationsCompleted, self.bestScore, self.bestArtifact

    def breedIndividuals(self, mother, father):
        """
        NOT FOR EXTERNAL USE.
        """
        newIndividual = Individual()
        motherGenes = mother.genes
        fatherGenes = father.genes
        numGenes = len(motherGenes)
        for k in motherGenes:
            geneToCopy = random.choice([motherGenes[k], fatherGenes[k]])
            copiedGene = deepcopy(geneToCopy)
            newIndividual.genes[k] = copiedGene
        self.mutateIndividual(newIndividual)  # Possibly mutate the newIndividual
        return newIndividual

    def mutateIndividual(self, individual):
        """
        NOT FOR EXTERNAL USE.
        """
        randomNumber = random.randint(0, 99)
        if randomNumber < self.curMutationRate:
            numGenesInIndividual = len(individual.genes)
            numGenesToMutate = random.randint(1, numGenesInIndividual)
            listOfGenesToMutate = random.sample(list(individual.genes.values()), numGenesToMutate)
            for gene in listOfGenesToMutate:
                gene.mutate()

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
        self.chanceToBreed = 0

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

    def mutate(self):
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

    def mutate(self):
        self.value = random.randint(self.min, self.max)

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

    def mutate(self):
        self.value = round(random.uniform(self.min, self.max), self.numDecimalPlaces)

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

    def mutate(self):
        self.value = random.choice(self.choices)

    def getHashableValue(self) -> str:
        indexOfCurrentValue = self.choices.index(self.value)
        return str(indexOfCurrentValue)

    def getNumParameters(self):
        return len(self.choices)
