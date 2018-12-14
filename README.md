# Dino
Easy AutoML/AutoDL/Hyperparameter optimization via metaheuristics.

Dino includes two optimization algorithms.  One is a genetic algorithm, and the other is based on simulated annealing.  I implemented both as I believe they each have merit over the other.  Simulated annealing is good for finding a minimum within a more explicit timeframe, while a genetic algorithm may find a better solution than simulated annealing, but possibly take longer to do so.  As this was created for the use case of deep learning, which frequently has models that take quite a while to train, I personally find myself using simulated annealing more than the genetic algorithm, as I can more accurately estimate how long the training will take.

As a brief introduction to the algorithms I will give a quick overview of some of the internal workings in each below.

Genetic Algorithm: The genetic algorithm "breeds" and mutates solutions that it has seen, and also ensures any new solutions it generates are unique.  There is an emphasis on breeding together better performing solutions, however, to even out the playing field, some less performant solutions are kept and bred as well.  Specifically, the best 25% are kept, and 10% of the other solutions are kept, via random sampling.  The actual mate selection is performed on a sliding scale.  Higher performing solutions have a higher chance of being a mate, and lower performing solutions have a lower chance of being a mate.  Once two mates are selected, they create a new solution.  This new solution is then given a chance to mutate.  If it is selected for mutation, it will have a random number of it's parameters mutated.  This new solution is then compared to all previously seen solutions.  If it is not unique, we repeat the breeding process until one is.  This process will continue until the new generation is full of unique, never before seen, solutions.

Simulated Annealing: The simulated annealing algorithm is based on "temperature".  In this implementation the temperature slides down a linear scale based on the number of iterations you want to run.  The temperature is used to balance the likelihood that a worse performing solution will be accepted. IE Greediness vs Exploration.  The actual chance of the worse performing solution being accepted is calculated by normalizing the difference between the previous best solution and the current, less performant, solution.  This normalized value is then subtracted from the current iterations temperature to get the chance of the less performant solution being accepted.  As expected, better performing solutions are accepted immediately.  The temperature is also used to determine the range of values to choose from numeric ranges, and also dictates the likelihood of a new value being chosen from a list.

Currently the algorithms are seperated into their own packages and expose their own API's.  There will likely be an all encompassing frontend API in the future, but luckily enough, the interfaces are virtually identical as they stand.

Here is an example of the genetic algorithm interface.  It also doubles as a good walkthrough in general:

```python
# Creates the optimizer and sets the population size to 10.
# You can also set the minimum chance of mutation.
optim = Optimizer(populationSize=10, chanceOfMutation=5)
# Create your optimizable parameters
# The labels can be any string value you want.  They just need to be unique.
# The "Gene" objects encapsulate the optimizable parameters.  All the currently available ones are shown below.
optim.addGene("gene_1", GeneBool())
optim.addGene("gene_2", GeneInt(0, 100))
optim.addGene("gene_3", GeneFloat(0, 2, 1))
#The "choice" gene takes anything you want, inside a list, and will select from them during optimization.
optim.addGene("gene_4", GeneChoice(["Relu", "Linear", "Conv2D", "Dense", "MaxPool2D"]))
optim.startTraining() #Prepares the first generation and loads the first set of parameters.
while True:
    # You call your optimizers getGeneValue to get the value for one of the genes you created above
    # As you can see, we use the same label to access the value of the created gene.
    # In this case we are just printing the values, but you can plug them in to anything you want.
    print("Bool: " + str(optim.getGeneValue("gene_1")))
    print("Integer: " + str(optim.getGeneValue("gene_2")))
    print("Float: " + str(optim.getGeneValue("gene_3")))
    print("Choice: " + str(optim.getGeneValue("gene_4")))
    # Here we generate a random score to feed to the optimizer.  Hopefully you don't reuse this part!
    score = random.uniform(0, 100)
    # Below we feed the score to the optimizer's "next" method.
    # This let's the optimizer know how well the last Individual(possible solution) performed.
    # The next method also takes a user supplied "artifact" IE object, such as a keras model.
    # the "artifact" is tied to the score it was supplied with.
    # "next" also returns a few things.  It's all in the documentation, but here is a quick overview.
    # "solutionsExhausted" is a bool that indicates if we have run through all possible solutions.
    # That is quite unlikely for any non trivial optimization problems, but still check it to be sure!
    # "numCompletedGenerations" is just what it sounds like.  The number of completed generations.
    # Also, the "bestScore" and it's "artifact", as explained above, are returned.
    solutionsExhausted, numCompletedGenerations, bestScore, artifact = optim.next(score)
    # "getBestParams" returns a dictionary of the best parameters found so far.  The keys are the gene's labels.
    bestParams = optim.getBestParameters()
    print(bestParams)
    # And here is a check to see if the solution space has run out.
    if solutionsExhausted:
        break
```

...and if we remove the comments...

```python
optim = Optimizer(populationSize=10, chanceOfMutation=5)
optim.addGene("gene_1", GeneBool())
optim.addGene("gene_2", GeneInt(0, 100))
optim.addGene("gene_3", GeneFloat(0, 2, 1))
optim.addGene("gene_4", GeneChoice(["Relu", "Linear", "Conv2D", "Dense", "MaxPool2D"]))
optim.startTraining()
while True:
    print("Bool: " + str(optim.getGeneValue("gene_1")))
    print("Integer: " + str(optim.getGeneValue("gene_2")))
    print("Float: " + str(optim.getGeneValue("gene_3")))
    print("Choice: " + str(optim.getGeneValue("gene_4")))
    score = random.uniform(0, 100)
    solutionsExhausted, numCompletedGenerations, bestScore, artifact = optim.next(score)
    bestParams = optim.getBestParameters()
    print(bestParams)
    if solutionsExhausted:
        break
```

The simulated annealing interface is very, very, similar.  Here it is:

```python
# The "minimumIterationsToRun" is just what it sounds like.  The more the better.
# "earlyStoppingIters" dictates how many iterations to wait for a better solution to be found after "minimumIterationsToRun".
# After "minimumIterationsToRun" is exhausted the algorithm acts as a purely greedy hillclimbing algorithm.
# So if you want to capitalize on the greedy portion, set "earlyStoppingIters" to a relatively large number.
# For example, in testing, I often set "minimumIterationsToRun" to 100 and "earlyStoppingIters" to 20.
# Play with these parameters to find your ideal setup.
optim = Optimizer(minimumIterationsToRun=100, earlyStoppingIters=20)
# The genes act the same as in the genetic algorithm.
optim.addGene("gene_1", GeneBool())
optim.addGene("gene_2", GeneInt(1, 100))
optim.addGene("gene_3", GeneFloat(0, 2, 8))
optim.addGene("gene_4", GeneChoice(["Relu", "Linear", "Conv2D", "Dense", "MaxPool2D"]))
optim.startTraining()
while True:
    print("Bool: " + str(optim.getGeneValue("gene_1")))
    print("Integer: " + str(optim.getGeneValue("gene_2")))
    print("Float: " + str(optim.getGeneValue("gene_3")))
    print("Choice: " + str(optim.getGeneValue("gene_4")))
    score = random.uniform(0, 100)
    # You feed the same data to the "next" function as you do in the genetic version, including an artifact.
    # The main difference is that "optimizingComplete" tells you when all your requested iterations are used up
    # and optimization has stopped.  You get no improvement after it returns True, so stop there.
    optimizingComplete, completedIterations, bestScore, artifact = optim.next(score)
    bestParams = optim.getBestParameters()
    print(bestParams)
    if optimizingComplete:
        print("Optimization Finished!")
        break
```

As you can see, the two algorithms' frontends are very similar.  There are only minor differences due to their internal workings.

Have fun using Dino!  If you have any issues that arise, or think a new feature should be added, please do let me know!

As a note, I am not currently accepting pull requests.
