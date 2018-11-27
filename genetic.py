import random
from math import inf as Infinity
from copy import deepcopy
from math import ceil, floor


class Optimizer:
    def __init__(self, populationSize: int = 20):
        """
        The main interface to Dino.

        The populationSize defines how many Individuals are in each generation.
        For instance, if you specify 20, there will be 20 loops before each generation is finished.
        After each generation is finished, then the actual optimization occurs.  So, if the population
        is set to 20, and you run your loop 40 times, you will have run one optimized loop, as the first generation
        was created randomly, but the second generation was optimized from the first generation's results.

        :param populationSize: The number of solutions generated and tried per generation
        """
        self.populationSize: int = populationSize
        self.numGenerationsCompleted: int = 0
        self.curGenerationIndividuals: list = []
        self.keptIndividuals = []
        self.requestedGenes: dict = {}
        self.curIndividual: Individual = None
        self.curIndividualNum: int = 0
        self.bestScore = Infinity
        self.bestArtifact = None

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
        for _ in range(self.populationSize):
            newIndividual = Individual()
            for key in self.requestedGenes:
                origGene = self.requestedGenes[key]
                newGene = deepcopy(origGene)
                newGene.mutate()
                newIndividual.genes[key] = newGene
            self.curGenerationIndividuals.append(newIndividual)
        self.curIndividual = self.curGenerationIndividuals[self.curIndividualNum]

    def next(self, inputScore: float = Infinity, userArtifact: object = None):
        """
        This method prepares the algorithm for the next loop.

        Call this method to prepare for the next loop.  It will store the user supplied score
        and also store an optional user artifact.  The user artifact could be, for instance,
        a Keras model.  It returns three things.  The first is the number of completed generations,
        the second is the best score seen throughout all the loops since the beginning of optimization,
        and the third is the user artifact that was supplied with the best score.  If an artifact was not stored,
        it returns None in it's place.

        Example:
        myOptimizer = Optimizer(10, 100)
        myOptimizer.addGene("my_parameter_to_optimize", GeneInt(1, 1000))
        myOptimizer.startTraining()
        value = myOptimizer.getGeneValue("my_parameter_to_optimize")
        ...
        kerasModel = load_model(...)
        ...
        score = something
        numCompletedGenerations, bestScore, bestArtifact = myOptimizer.next(score, kerasModel)


        :param inputScore: The score of the last optimization run
        :param userArtifact: An optional object that will be tied to the supplied score. IE: Keras model
        :return:
        Int: Number of completed generations
        Float: The best score seen so far
        Object: An artifact tied to the best score.  If not available, None.
        """
        if inputScore < self.bestScore:
            self.bestScore = inputScore
            if userArtifact is not None:
                self.bestArtifact = userArtifact
        self.curIndividual.score = inputScore
        self.curIndividualNum += 1
        if self.curIndividualNum < self.populationSize:
            self.curIndividual = self.curGenerationIndividuals[self.curIndividualNum]
            return self.numGenerationsCompleted, self.bestScore, self.bestArtifact

        # If we reach this code then we have finished the generation.
        self.curIndividualNum = 0

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
                newIndividual = breedIndividuals(self.keptIndividuals[motherIndex], self.keptIndividuals[fatherIndex])
                self.curGenerationIndividuals.append(newIndividual)
                numIndividualsToCreate -= 1
        self.curIndividual = self.curGenerationIndividuals[0]
        self.numGenerationsCompleted += 1
        return self.numGenerationsCompleted, self.bestScore, self.bestArtifact


class Individual:
    """
    NOT FOR EXTERNAL USE.
    """

    def __init__(self):
        self.genes = {}
        self.score = Infinity
        self.chanceToBreed = 0


class GeneBool:
    """
    Gene that optimizes a boolean value.
    """

    def __init__(self):
        self.value = random.choice([True, False])

    def mutate(self):
        """
        Not for external use
        """
        self.value = random.choice([True, False])


class GeneInt:
    """
    Gene that optimizes an integer value.
    """

    def __init__(self, min: int = 0, max: int = 100):
        self.min = min
        self.max = max
        self.value = random.randint(self.min, self.max)

    def mutate(self):
        self.value = random.randint(self.min, self.max)


class GeneFloat:
    """
    Gene that optimizes a float value
    """

    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max
        self.value = random.uniform(self.min, self.max)

    def mutate(self):
        self.value = random.uniform(self.min, self.max)


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


def breedIndividuals(mother: Individual, father: Individual) -> Individual:
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
    mutateIndividual(newIndividual)  # Possibly mutate the newIndividual
    return newIndividual


def mutateIndividual(individual: Individual):
    """
    NOT FOR EXTERNAL USE.
    """
    chanceOfMutation = 5
    for k in individual.genes:
        randNumber = random.randint(0, 99)
        if randNumber < chanceOfMutation:
            individual.genes[k].mutate()


# optim = Optimizer(100)
# optim.addGene("gene_1", GeneBool())
# optim.addGene("gene_2", GeneInt(1, 100))
# optim.addGene("gene_3", GeneFloat(0, 10))
# optim.addGene("gene_4", GeneChoice(["I'm cool", "You're cool", "We're all cool", 1]))
# optim.startTraining()
# while True:
#     print("Bool: " + str(optim.getGeneValue("gene_1")))
#     print("Integer: " + str(optim.getGeneValue("gene_2")))
#     print("Float: " + str(optim.getGeneValue("gene_3")))
#     print("Choice: " + str(optim.getGeneValue("gene_4")))
#     score = random.randint(1, 100)
#     optim.next(score)
