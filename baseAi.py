#!/usr/bin/env python3
# True AI Test based of CodeBullet Javascript code.
# -*- coding: utf-8 -*-

# Based of https://github.com/Code-Bullet/NEAT-Template-JavaScript

NAME = 'TRUE AI'
AUTHOR = 'CoolCat467'
__version__ = '0.0.0'

import math, random
#from threading import Thread, Event

class Node(object):
    def __init__(self, no):
        self.number = no
        self.inputSum = 0 #current sum i.e. before activation
        self.outputValue = 0 #after activation function is applied
        self.outputConnections = []
        self.layer = 0
    
    def __repr__(self):
        return 'Node(%i)' % self.number
        #return 'Node(%i, %f, %f, %s, %i)' % (self.number, self.inputSum, self.outputValue, self.outputConnections, self.layer)
    
    def sigmoid(self, x):
        return 1 / (1 + (math.e**(-4.9 * x)))
    
    def engage(self):
        # the node sends its output to the inputs of the nodes its connected to
        if self.layer != 0:#no sigmoid for the inputs and bias
            self.outputValue = self.sigmoid(self.inputSum)
        
        for i in range(len(self.outputConnections)):# for each connection
            if self.outputConnections[i].enabled:# dont do anything if not enabled
                self.outputConnections[i].toNode.inputSum += self.outputConnections[i].weight * self.outputValue
                #add the weighted output to the sum of the inputs of whatever node this node is connected to
    
    def isConnectedTo(self, node):
        if node.layer == self.layer:
            return False
        
        if node.layer < self.layer:
            for i in range(len(node.outputConnections)):
                if node.outputConnections[i].toNode == self:
                    return True
        else:
            for i in range(len(node.outputConnections)):
                if node.outputConnections[i].toNode == node:
                    return True
        return False
    
    def __copy__(self):
        clone = Node(int(self.number))
        clone.layer = int(self.layer)
        return clone
    
    def clone(self):
        return self.__copy__()
    pass

class connectionGene(object):
    def __init__(self, fromNode, toNode, weight, inno):
        self.fromNode = fromNode
        self.toNode = toNode
        self.weight = weight
        self.enabled = True
        self.innovationNo = inno
        #each connection is given a innovation number to compare genomes
    
    def __repr__(self):
        return '<connectionGene>'
        
    def mutateWeight(self):
        change = random.randint(1, 10)
        if change == 1:#10% of the time completely change the self.weight
            self.weight = random.randint(-100, 100)/100
        else:#otherwise slightly change it
            self.weight += random.gauss(0, 1) / 50
            # Keep self.weight within bounds
            self.weight = min(self.weight, 1)
            self.weight = max(self.weight, -1)
    
##    def __copy__(self):
##        copy = connectionGene(self.fromNode.__copy__(), self.toNode.__copy, self.weight, self.innovationNo)
##        copy.enabled = bool(self.enabled)
##        return copy
    
    def clone(self, fromNode, toNode):
        clone = connectionGene(fromNode, toNode, self.weight, self.innovationNo)
        clone.enabled = bool(self.enabled)
        return clone
    pass

class connectionHistory(object):
    def __init__(self, fromNode, toNode, inno, innoNos):
        self.fromNode = fromNode
        self.toNode = toNode
        self.innovationNumber = inno
        self.innovationNumbers = list(innoNos)
        # the innovation Numbers from the connections of the
        # genome which first had this mutation
        # ourself represents the genome and allows us to test if
        # another genoeme is the same
        # as ourself is before this connection was added
    
    def __repr__(self):
        return 'connectionHistory(%s, %s, %i, %s)' % (self.fromNode, self.toNode, self.innovationNumber, str(self.innovationNumbers))
    
    def matches(self, genome, fromNode, toNode):
        """Returns whether the genome matches the original genome and the connection is between the same nodes"""
        if len(genome.genes) == len(self.innovationNumbers):
            for gene in genome.genes:
                if not gene.innovationNo in self.innovationNumbers:
                    return False
            # If reached this far then innovationNumbers matches the gene
            # innovation numbers and the connection between the same nodes,
            # so it does match
            return True
        return False
    pass

class Genome(object):
    def __init__(self, inputs, outputs, crossover=False):
        self.genes = []# A list of connecteions between our nodes which represent the NN (node number?) 
        self.nodes = []
        self.inputs = inputs
        self.outputs = outputs
        self.layers = 2
        self.nextNode = 0
        self.network = []# A list of nodes in the order that they are needed to be considered in the NN
        # create input nodes
        
        if crossover:
            return
        
        for i in range(inputs):
            self.nodes.append(Node(i))
            self.nextNode += 1
            self.nodes[i].layer = 0
        
        # create output nodes
        for i in range(outputs):
            self.nodes.append(Node(i + self.inputs))
            self.nodes[i + self.inputs].layer = 1
            self.nextNode += 1
        
        self.nodes.append(Node(self.nextNode))
        self.biasNode = int(self.nextNode)
        self.nextNode += 1
        self.nodes[self.biasNode].layer = 0
    
    def __repr__(self):
        return '<Genome Object with %i layers, Bias node is %i, %i Nodes, and %i Genes>' % (self.layers, self.biasNode, len(self.nodes), len(self.genes))
    
    def getInnovationNumber(self, innovationHistory, fromNode, toNode):
        """Returns the innovation number for the new mutation"""
        isNew = True
        connectionInnovationNumber = 0
        for i in range(len(innovationHistory)):
            if innovationHistory[i].matches(self, fromNode, toNode):
                isNew = False
                connectionInnovationNumber = innovationHistory[i].innovationNumber
                break
        
        if isNew:
            # if the mutation is new then create an arrayList of varegers representing the current state of the genome
##            innoNumbers = []
##            for i in range(len(self.genes)):
##                innoNumbers.append(self.genes[i].innovationNo)
            innoNumbers = [gene.innovationNo for gene in self.genes]
            innovationHistory.append(connectionHistory(fromNode.number, toNode.number, connectionInnovationNumber, innoNumbers))
            connectionInnovationNumber += 1
        return connectionInnovationNumber
    
    def connectNodes(self):
        """Adds the connections going out of a node to that node so that it can acess the next node during feeding forward"""
        # Clear connections
        for i in range(len(self.nodes)):
            self.nodes[i].outputConnections = []
        # For each connection, add the corrosponding gene to the node.
        for i in range(len(self.genes)):
            self.genes[i].fromNode.outputConnections.append(self.genes[i])
    
    def fullyConnect(self, innovationHistory):
        
        for i in range(self.inputs):
            for ii in range(self.outputs):
                connectionInnovationNumber = self.getInnovationNumber(innovationHistory, self.nodes[i], self.nodes[len(self.nodes) - ii - 2])
                self.genes.append(connectionGene(self.nodes[i], self.nodes[len(self.nodes) - ii - 2], random(-100, 100)/100, connectionInnovationNumber))
        
        connectionInnovationNumber = self.getInnovationNumber(innovationHistory, self.nodes[self.biasNode], self.nodes[len(self.nodes) - 2])
        self.genes.append(connectionGene(self.nodes[self.biasNode], self.nodes[len(self.nodes) - 2], random(-100, 100)/100, connectionInnovationNumber))
        
        connectionInnovationNumber = self.getInnovationNumber(innovationHistory, self.nodes[self.biasNode], self.nodes[len(self.nodes) - 3])
        self.genes.append(connectionGene(self.nodes[self.biasNode], self.nodes[len(self.nodes) - 3], random(-100, 100)/100, connectionInnovationNumber))
        
        self.connectNodes()
    
    def getNode(self, nodeNumber):
        """Returns the node with a matching number, as sometimes the this.nodes will not be in order"""
        for node in self.nodes:
            if node.number == nodeNumber:
                return node
        return None
    
    def feedForward(self, inputValues):
        """Feeding in input values varo the NN and returning list"""
        # Set the outputs of the input nodes
        for i in range(len(self.inputs)):
            self.nodes[i].outputValue = inputValues[i]
        
        self.nodes[self.biasNode].outputValue = 1#Output of bias is 1
        
        for i in range(len(self.network)):
            # for each node in the network engage it
            self.network[i].engage()
        
        # the outputs are the self.nodes[inputs] to self.nodes[inputs+outputs-1]
        outs = {}
        for i in range(self.outputs):
            outs[i] = self.nodes[self.inputs + 1].outputValue
        
        # reset all nodes for the next feed forward
        for i in range(len(self.nodes)):
            self.nodes[i].inputSum = 0
        
        return [outs[k] for k in sorted(list(outs.keys()))]
    
    def generateNetwork(self):
        """sets up the NN as a list of this.nodes in order to be engaged"""
        self.connectNodes()
        self.network = []
        # For each layer add the node in that layer, since layers cannot connect to themselves there is no need to order the nodes within a layer
        for layer in range(self.layers):
            for node in self.nodes:# For each node
                if node.layer == layer:# If the node is in that layer
                    self.network.append(node)# Add that node to the network
    
    def fullyConnected(self):
        """returns whether the network is fully connected or not"""
        maxConnections = 0
        nodesInLayers = {}#Dictionary which stored the amount of nodes in each layer
        for layer in range(self.layers):
            nodesInLayers[layer] = 0
        
        # Populate dictionary
        for node in self.nodes:
            nodesInLayers[node.layer] += 1
        
        # for each layer the maximum ammount of connections is the number of the
        # layer times the number of nodes in front of it.
        # so lets add the max fir each layer together and then we will get
        # the maximum ammount of connections in the network
        for layer in range(self.layers):
            nodesInFront = 0
            for i in range(layer + 1, self.layers):# for each layer in front of this layer,
                nodesInFront += nodesInLayers[layer]# add up nodes
            
            maxConnections += nodesInLayers[layer] * nodesInFront
        #if the number of connections is equal to the max number of connections possible then it is full
        return maxConnections <= len(self.genes)
    
    def randomConnectionNodesAreBad(self, r1, r2):
        if self.nodes[r1].layer == self.nodes[r2].layer:
            return True# if the nodes are in the same layer
        if self.nodes[r1].isConnectedTo(self.nodes[r2]):
            return True#if the nodes are already connected
        return False
    
    def addConnection(self, innovationHistory):
        """Adds a connection between 2 nodes which aren't currently connected"""
        # Cannot add a connection to a fully connected network
        if self.fullyConnected():
            print('Connection failed.')
            raise LookupError('Cannot add a connection to a fully connected network.')
        
        # Get random node
        randomNode1 = random.randint(0, len(self.nodes)-1)
        randomNode2 = random.randint(0, len(self.nodes)-1)
        # If the nodes are the same or are connected, get new random nodes
        while self.randomConnectionNodesAreBad(randomNode1, randomNode2):
            randomNode1 = random.randint(0, len(self.nodes)-1)
            randomNode2 = random.randint(0, len(self.nodes)-1)
        # if the first random node is after the second, then switch them
        if self.nodes[randomNode1].layer > self.nodes[randomNode2].layer:
            randomNode1, randomNode2 = [randomNode2, randomNode1]
        # get the innovation number of the connection
        # this will be a new number if no identical genome has mutated in the same way
        connectionInnovationNumber = self.getInnovationNumber(innovationHistory, self.nodes[randomNode1], self.nodes[randomNode2])
        
        # Add the connection with a random dictionary
        self.genes.append(connectionGene(self.nodes[randomNode1], self.nodes[randomNode2], random.randint(-100, 100)/100, connectionInnovationNumber))
        self.connectNodes()
    
    def addNode(self, innovationHistory):
        """Pick a random connection to create a node between"""
        if not len(self.genes):
            self.addConnection(innovationHistory)
        
        randomConnection = random.randint(0, len(self.genes)-1)
        while self.genes[randomConnection].fromNode == self.nodes[self.biasNode] and len(self.genes) != 1:
            randomConnection = random.randint(0, len(self.genes)-1)
        
        self.genes[randomConnection].enabled = False# Disable it
        
        newNodeNo = int(self.nextNode)
        self.nodes.append(Node(newNodeNo))
        self.nextNode += 1
        
        # Add a new connection to the new node with a weight of 1
        connectionInnovationNumber = self.getInnovationNumber(innovationHistory, self.genes[randomConnection].fromNode, self.getNode(newNodeNo))
        self.genes.append(connectionGene(self.genes[randomConnection].fromNode, self.getNode(newNodeNo), 1, connectionInnovationNumber))
        
        connectionInnovationNumber = self.getInnovationNumber(innovationHistory, self.getNode(newNodeNo), self.genes[randomConnection].toNode)
##        print('This is a test from addNode. If same, bad.', self.getNode(newNodeNo).layer)
        self.getNode(newNodeNo).layer = self.genes[randomConnection].fromNode.layer + 1
##        print(self.getNode(newNodeNo).layer)
        
        connectionInnovationNumber = self.getInnovationNumber(innovationHistory, self.nodes[self.biasNode], self.getNode(newNodeNo))
        #connect the bias to the new node with a weight of 0
        self.genes.append(connectionGene(self.nodes[self.biasNode], self.getNode(newNodeNo), 0, connectionInnovationNumber))
        
        # If the layer of the new node is equal to the layer of the output node
        # of the old connection, then the new layer needs the be created more
        # accurately the layer numbers of all layers equal to or greater than
        # this new node need to be incrimented
        if self.getNode(newNodeNo).layer == self.genes[randomConnection].toNode.layer:
            for i in range(len(self.nodes)-1):
                if self.nodes[i].layer >= self.getNode(newNodeNo).layer:
                    self.nodes[i].layer += 1
            self.layers += 1
        self.connectNodes()
    
    def mutate(self, innovationHistory):
        """Mutates the genome."""
        if not len(self.genes):
            self.addConnection(innovationHistory)
        
        rand1 = random.randint(0, 100)
        if rand1 < 80:#80% of the time mutate weights
            for i in range(len(self.genes)):
                self.genes[i].mutateWeight()
        
        rand2 = random.randint(0, 100)
        if rand2 < 5:#5% of the time add a new connection
            self.addConnection(innovationHistory)

        rand3 = random.randint(0, 100)
        if rand3 < 1:#1% of the time add a node
            self.addNode(innovationHistory)
    
    def matchingGene(self, parent2, innovationNumber):
        """Returns whether or not there is a gene matching the input innovation number  in the input genome"""
        for i in range(len(parent2.genes)):
            if parent2.genes[i].innovationNo == innovationNumber:
                return i
        return None#no matching gene found
    
    def crossover(self, parent2):
        """Called when this genome is better than other parent"""
        child = Genome(int(self.inputs), int(self.outputs), True)
        child.genes = []
        child.nodes = []
        child.layers = int(self.layers)
        child.nextNode = int(self.nextNode)
        child.biasNode = int(self.biasNode)
        childGenes = []#list of genes to be inherrited from the parents
        isEnabled = []
        # All inheareted genes
        for i in range(len(self.genes)):
            setEnabled = True
            parent2gene = self.matchingGene(parent2, self.genes[i].innovationNo)
            if not parent2gene is None:#if the genes match
                if not self.genes[i].enabled or not parent2.genes[parent2gene].enabled:
                    # if either of the matching genes are disabled
                    if random.randint(0, 100) < 75:#75% of the time disable the child's gene
                        setEnabled = False
                rand = random.randint(0, 100)
                if rand > 5:#Get gene from ourselves, we are better #by the way original was <. odd.
                    childGenes.append(self.genes[i])
                else:#Otherwise, get gene from parent2 which is worse
                    childGenes.append(parent2.genes[parent2gene])
            else:#disjoit or excess gene
                childGenes.append(self.genes[i])
                setEnabled = self.genes[i].enabled
            isEnabled.append(setEnabled)
        
        # since all excess and disjovar genes are inherrited from the more
        # fit parent (this Genome) the childs structure is no different from
        # this parent, with exception of dormant connections being enabled but
        # this wont effect our nodes
        # so all the nodes can be inherrited from this parent
        for node in self.nodes:
            child.nodes.append(node.clone())
        
        # Clone all the connections so that they connect the child's new nodes
        for i in range(len(childGenes)):
            child.genes.append(childGenes[i].clone(child.getNode(childGenes[i].fromNode.number), child.getNode(childGenes[i].toNode.number)))
            child.genes[i].enabled = isEnabled[i]
        
        child.connectNodes()
        return child
        
    def printGeneome(self):
        """Prints out information about genome"""
        print('Private genome layers:', self.layers)
        print('Bias node:', self.biasNode)
        print('Nodes:')
        print(*[node.number for node in self.nodes], sep=', ')
        print('Genes:')
        for i in range(len(self.genes)):#for each connectionGene
            print("Gene " + this.genes[i].innovationNo + "From node " + this.genes[i].fromNode.number + "To node " + this.genes[i].toNode.number +
            "is enabled " + this.genes[i].enabled + "from layer " + this.genes[i].fromNode.layer + "to layer " + this.genes[i].toNode.layer + "weight: " + this.genes[i].weight)
        print()
    
    def __copy__(self):
        clone = Genome(self.inputs, self.outputs, True)
        # Copy our nodes
        for i in range(len(self.nodes)):
            clone.nodes.append(self.nodes[i].clone())
        # Copy all the connections so that they connect to the clone's new nodes
        for i in range(len(self.genes)):
            clone.genes.append(self.genes[i].clone(clone.getNode(self.genes[i].fromNode.number), clone.getNode(self.genes[i].toNode.number)))
        
        clone.layers = int(self.layers)
        clone.nextNode = int(self.nextNode)
        clone.biasNode = int(self.biasNode)
        clone.connectNodes()
        
        return clone
    
    def clone(self):
        """Returns a copy of this genome"""
        return self.__copy__()
    
    def drawGenome(startX, startY, w, h):
        raise NotImplemented
    pass

class BasePlayer(object):
    def __init__(self):
        self.fitness = 0
        self.vision = []#The input array for the nural network
        self.decision = []#The output of the nural network
        self.unadjustedFitness = 0
        self.lifespan = 0#How long the player lived for fitness
        self.bestScore = 0#Stores the score achived used for replay
        self.dead = False
        self.score = 0
        self.gen = 0
        
        self.genomeInputs = 5
        self.genomeOutputs = 2
        self.brain = None
        self.start()
    
    def __repr__(self):
        return '<BasePlayer Object>'
    
    def start(self):
        self.brain = Genome(self.genomeInputs, self.genomeOutputs)

    def show(self):
        pass
    
    def move(self):
        pass
    
    def update(self):
        pass
    
    def look(self):
        pass
    
    def think(self):
        self.decision = self.brain.feedForward(self.vision)
        self.do = self.decision.index(max(self.decision))
        pass
    
    def clone(self):
        clone = BasePlayer()
        clone.brain = self.brain.clone()
        clone.fitness = float(self.fitness)
        clone.brain.generateNetwork()
        clone.gen = int(self.gen)
        clone.bestScore = float(self.score)
        return clone
    
    def cloneForReplay(self):
        clone = BasePlayer()
        clone.brain = self.brain.clone()
        clone.fitness = float(self.fitness)
        clone.brain.generateNetwork()
        clone.gen = int(self.gen)
        clone.bestScore = float(self.score)
        return clone
    
    def calculateFitness(self):
        self.fitness = random.randint(0, 10)
    
    def crossover(self, parent2):
        child = BasePlayer()
        child.brain = self.brain.crossover(parent2.brain)
        child.brain.generateNetwork()
        return child
    pass

class Species(object):
    def __init__(self, player=None):
        self.players = []
        self.bestFitness = 0
        self.champ = None
        self.averageFitness = 0
        self.staleness = 0
        # how many generations have gone without an improvement
        self.rep = None
        
        # Co-efficiants for testing compadibility
        self.excessCoeff = 1
        self.weightDiffCoeff = 0.5
        self.compatibilityThreshold = 3
        if player:
            self.players.append(player)
            # Since it is the only one in the spicies it is by default the best
            self.bestFitness = player.fitness
            self.rep = player.brain.clone()
            self.champ = player.cloneForReplay()
    
    def __repr__(self):
        return '<Species Object>'
    
    def getExcessDisjoint(self, brain1, brain2):
        """Returns the number of excess and disjoint genes"""
        matching = 0
        for gene1 in brain1.genes:
            for gene2 in brain2.genes:
                if gene1.innovationNo == gene2.innovationNo:
                    matching += 1
                    break
        # Return number of excess and disjoint genes
        return (len(brain1.genes) + len(brain2.genes) - 2) * matching
    
    def averageWeightDiff(self, brain1, brain2):
        if not len(brain1.genes) or not len(brain2.genes):
            return 0
        matching = 0
        totalDiff = 0
        for gene1 in brain1.genes:
            for gene2 in brain2.genes:
                if gene1.innovationNo == gene2.innovationNo:
                    matching += 1
                    totalDiff += abs(gene1.weight - gene2.weight)
                    break
##        if not matching:
##            return 100#devide by zero error otherwise
        return 100 if not matching else totalDiff / matching
    
    def sameSpecies(self, genome):
        """Returns if a genime is in species"""
        excessAndDisjoint = self.getExcessDisjoint(genome, self.rep)
        averageWeightDiff = self.averageWeightDiff(genome, self.rep)
        largeGenomeNormalizer = max(len(genome.genes) - 20, 1)
        # compatibility formula
        compatibility = (self.excessCoeff * excessAndDisjoint / largeGenomeNormalizer) + (self.weightDiffCoeff * averageWeightDiff)
        return self.compatibilityThreshold > compatibility
    
    def addToSpecies(self, player):
        self.players.append(player)
    
    def sortSpecies(self):
        """sorts the species by fitness"""
        temp = []
        for i in range(len(self.players)):
            pmax = 0
            pmaxIndex = 0
            for ii in range(len(self.players)):
                if self.players[ii].fitness > pmax:
                    pmax = self.players[ii].fitness
                    pmaxIndex = ii
            temp.append(self.players[pmaxIndex])
            del self.players[pmaxIndex]
        
        self.players = temp.copy()
        if not len(self.players):
            self.staleness = 200
            return
        # if new best player
        if self.players[0].fitness > self.bestFitness:
            self.staleness = 0
            self.bestFitness = self.players[0].fitness
            #self.rep = self.players[0].cloneForReplay()
            self.rep = self.players[0].brain.clone()
        else:# If no new best player,
            self.staleness += 1
    
    def setAverage(self):
        if not len(self.players):
            self.averageFitness = 0
            return
        self.averageFitness = sum(player.fitness for player in self.players) / len(self.players)
    
    def fitnessSharing(self):
        for i in range(len(self.players)):
            self.players[i].fitness /= len(self.players)
    
    def selectPlayer(self):
        """Selects a player based on it's fitness"""
        fitnessSum = sum([player.fitness for player in self.players])
        rand = random.randint(0, int(fitnessSum))
        runningSum = 0
        for player in self.players:
            runningSum += player.fitness
            if runningSum > rand:
                return player
        return self.players[0]
    
    def giveMeBaby(self, innovationHistory):
        if random.randint(0, 100) < 25:#25% of the time there is no crossover and child is a clone of a semi-random player
            baby = self.selectPlayer().clone()
        else:
            parent1 = self.selectPlayer()
            parent2 = self.selectPlayer()
            
            # The crossover function expects the highest fitness parent
            # to be the object and the seccond parent as the argument
            if parent1.fitness < parent2.fitness:
                parent1, parent2 = [parent2, parent1]
            baby = parent1.crossover(parent2)
        baby.brain.mutate(innovationHistory)
        return baby
    
    def cull(self):
        if len(self.players) > 2:
            self.players = self.players[int(len(self.players)/2):]
    pass

class Population(object):
    def __init__(self, size):
        self.players = []
        self.bestPlayer = None
        self.bestScore = 0
        self.globalBestScore = 0
        self.gen = 1
        self.innovationHistory = []
        self.genPlayers = []
        self.species = []
        
        self.massExtinctionEvent = False
        self.newStage = True
        
        for i in range(size):
            self.players.append(BasePlayer())
            self.players[len(self.players)-1].brain.mutate(self.innovationHistory)
            self.players[len(self.players)-1].brain.generateNetwork()
    
    def __repr__(self):
        return '<Population Object with %i Players and %i Generations>' % (len(self.players), self.gen)
    
    def updateAlive(self):
        for i in range(len(self.players)):
            player = self.players[i]
            if not player.dead:
                player.look()#Get inputs for brain
                player.think()#Use outputs from neural network
                player.update()#Move the player according to the outputs from the neural network
                if not showNothing and (not showBest or not I):
                    player.show()
                if player.score > self.globalBestScore:
                    self.globalBestScore = player.score
    
    def done(self):
        """Returns true if all the players are dead. :("""
        for player in self.players:
            if not player.dead:
                return False
        return True
    
    def setBestPlayer(self):
        """Sets the best player globally and for current generation"""
        if not (self.species and self.species[0].players):
            return
        tempBest = self.species[0].players[0]
        tempBest.gen = self.gen
        
        if tempBest.score >= self.bestScore:
            bestClone = tempBest.cloneForReplay()
            self.genPlayers.append(bestClone)
            #print stuff was here
            self.bestScore = tempBest.score
            self.bestPlayer = bestClone
    
    def spectate(self):
        """Seperate players into species based on how similar they are to the leaders of the species in the previous generation"""
        # Empty current species
        for s in self.species:
            del s.players[:]
        # For each player,
        for player in self.players:
            speciesFound = False
            # For each species
            for s in self.species:
                if s.sameSpecies(player.brain):
                    s.addToSpecies(player)
                    speciesFound = True
                    break
            if not speciesFound:
                self.species.append(Species(player))
    
    def calculateFitness(self):
        """Calculate the fitness of each player"""
        for player in self.players:
            player.calculateFitness()
    
    def sortSpecies(self):
        """Sort the species to be ranked in fitness order, best first"""
        for species in self.species:
            species.sortSpecies()
        # Sort the species by a fitness of its best player
        # using selection sort like a loser
        temp = []
        for i in range(len(self.species)):
            smax = 0
            maxIndex = 0
            for ii in range(len(self.species)):
                if self.species[ii].bestFitness > smax:
                    smax = self.species[ii].bestFitness
                    maxIndex = ii
            temp.append(self.species[maxIndex])
            del self.species[maxIndex]
        self.species = temp
    
    def massExtinction(self):
        """sad"""
        for s in range(5, len(self.species)):
            del self.species[s]
    
    def cullSpecies(self):
        """Kill off the bottom half of each species"""
        for s in self.species:
            s.cull()
            s.fitnessSharing()#Also while we're at it do fitness sharing
            s.setAverage()
    
    def killStaleSpecies(self):
        """Kills all species which haven't improved in 15 generations"""
        for i in range(len(self.species)-1, -1, -1):
            if self.species[i].staleness >= 15:
                del self.species[i]
    
    def getAvgFitnessSum(self):
        return sum([s.averageFitness for s in self.species])
    
    def killBadSpecies(self):
        """Kill species which are so bad they can't reproduce"""
        averageSum = self.getAvgFitnessSum()
        if not averageSum:
            return
        for i in range(len(self.species)-1, -1, -1):
            if self.species[i].averageFitness / averageSum * len(self.players) < 1:
                del self.species[i]
    
    def naturalSelection(self):
        """This function is called when all players in the player list are dead and a new generation needs to be made"""
        previousBest = self.players[0]
        self.spectate()#Seperate players into species
        self.calculateFitness()#Calculate the fitness of each player
        self.sortSpecies()#Sort the species to be ranked in fitness order, best first
        if self.massExtinctionEvent:
            self.massExtinction()
            self.massExtinctionEvent = False
        self.cullSpecies()#Kill off the bottom half of each species
        self.setBestPlayer()#Save the best player of this generation
        self.killStaleSpecies()#Remove species which haven't improved in 15 generations
        self.killBadSpecies()#Kill species which are so bad they can't reproduce
        
        averageSum = self.getAvgFitnessSum()
        if averageSum == 0:
            averageSum = 0.1
        children = []
        for species in self.species:
            children.append(species.champ.clone())#Add champion without any mutation
            NoOfChildren = round((species.averageFitness / averageSum * len(self.players)) - 1)
            for i in range(NoOfChildren):
                children.append(species.giveMeBaby(self.innovationHistory))
        if len(children) < len(self.players):
            children.append(previousBest.clone())
        while len(children) < len(self.players):#If not enough babys
            if self.species:
                children.append(self.species[0].giveMeBaby(self.innovationHistory))#Get babys from the past generation
            else:
                clone = previousBest.clone()
                clone.brain.mutate(self.innovationHistory)
                children.append(clone)
        
        self.players = [child for child in children]
        self.gen += 1
        for player in self.players:
            player.brain.generateNetwork()
    
    def playerInBatch(self, player, worlds):
        b = self.batchNo * self.worldsPerBatch
        e = min((self.batchNo + 1) * self.worldsPerBatch, len(worlds))
        for i in range(b, e):
            if player.world == worlds[i]:
                return True
        return False
    
    def updateAliveInBatches(self, worlds, showBest):
        """Update all the players that are alive"""
        aliveCount = 0
        for i in range(len(self.players)):
            player = self.players[i]
            if self.playerInBatch(player, worlds):
                if not player.dead:
                    alive += 1
                    player.look()
                    player.think()
                    player.update()
                    if not showNothing and (showBest or i == 0):
                        player.show()
                    if player.score > self.globalBestScore:
                        self.globalBestScore = player.score
    
    def stepWorldsInBatch(self, worlds, FPS=30, arg2=10, arg3=10):
        b = self.batchNo * self.worldsPerBatch
        e = min((self.batchNo + 1) * self.worldsPerBatch, len(worlds))
        for i in range(b, e):
            worlds[i].Step(FPS, arg2, arg3)
    
    def batchDead(self):
        """Returns True if all the players in a batch are dead. :("""
        b = self.batchNo * self.worldsPerBatch
        e = min((self.batchNo + 1) * self.worldsPerBatch, len(worlds))
        for i in range(b, e):
            if not self.players[i].dead:
                return False
        return True
    pass

cat = Population(5)
[cat.naturalSelection() for i in range(500)]
print(cat)
