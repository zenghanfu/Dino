# Dino
Easy AutoML/AutoDL/Hyperparameter optimization via Evolutionary Computing

Dino has hopes of being "pip install"ed soon...

Dino is intended to be a simple solution to all of your difficult optimization needs.  While Dino has been designed with general optimization in mind, specifically minimization of a loss function, It was born out of my hope to make Deep Learning architecture search and hyperparameter optimization easy.

Dino is based on a few principles.  For one, using Dino has to be easy.  I didn't want a framework that was complicated to use, as I would be using it a lot.  The simpler the better.  Secondly, Dino has been designed to be adaptive.  As I assume most of those in the deep learning community know, it's no fun setting parameters, testing them, waiting two and a half lifetimes, and then trying a new set of parameters.  So, you don't want to have to do that for your model's optimizer as well, right? Dino's adaptability will likely grow over time, but it already gives good results.  Dino was also designed to be efficient.  As you will notice if you run the following code below, Dino really slows down towards the end of the solution space, and it's not due to bad programming.  Dino keeps track of the hash of all the possible solutions it has seen and doesn't allow already seen solutions to be regenerated, hence the slowdown, so you never waste computational time on already scored solutions.  This really helps when models take hours, days, or weeks to test.

Using Dino is very easy.  In fact, pretty much all of the functions have documentation and examples inside them.  However, here is a brief example:

```python
# Creates the optimizer and sets the population size to 10.
# You can also set the minimum chance of mutation.
optim = Optimizer(populationSize=10, chanceOfMutation=5)
# Create your optimizable parameters
# The labels (or keys if you prefer) can be any string value you want.  They just need to be unique.
# The "Gene" objects encapsulate the optimizable parameters.  All the currently available ones are shown below.
optim.addGene("gene_1", GeneBool())
optim.addGene("gene_2", GeneInt(0, 100))
optim.addGene("gene_3", GeneFloat(0, 2, 1))
#The "choice" gene takes anything you want, inside a list, and will select from them during optimization.
optim.addGene("gene_4", GeneChoice(["Relu", "Linear", "Conv2D", "Dense", "MaxPool2D"]))
optim.startTraining() #Prepares the first generation and loads the first set of parameters.
while True:
    # You call your optimizers getGeneValue to get the value for one of the genes you created above
    # As you can see, we use the same key to access the value of the created gene.
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

Have fun using Dino!  If you have any issues that arise, or think a new feature should be added, please do let me know!

As a note, I am not currently accepting pull requests.
