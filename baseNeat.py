#!/usr/bin/env python3
# NEAT Artificial Intelegence Template Module based of CodeBullet Javascript code.
# -*- coding: utf-8 -*-

# Heavily based of https://github.com/Code-Bullet/NEAT-Template-JavaScript

__title__ = 'NEAT Artificial Intelegence Template Module'
__author__ = 'CoolCat467'
__version__ = '1.1.0'
__ver_major__ = 1
__ver_minor__ = 1
__ver_patch__ = 0

import math, random
from Vector2 import *

class Node(object):
    """Represents a nuron in a brain."""
    def __init__(self, no):
        self.number = no
        self.inputSum = 0 #current sum i.e. before activation
        self.outputValue = 0 #after activation function is applied
        self.outputConnections = []
        self.layer = 0
    
    def __repr__(self):
        """Return what this object should be represented by in the python interpriter."""
        return 'Node(%i)' % self.number
        #return 'Node(%i, %f, %f, %s, %i)' % (self.number, self.inputSum, self.outputValue, self.outputConnections, self.layer)
    
    @staticmethod
    def sigmoid(x):
        """Return the sigmoid of x. AIs use this to prevent the need for keeping track of a threshold value for activation."""
##        return 1 / (1 + (math.e**(-4.9 * x)))#Original
        # No idea why the original does -4.9 * x, that is odd.
        return 1 / (1 + (math.e ** (-x)))#Modified
    
    def engage(self):
        """Calculate self.outputValue using the sigmoid function, then add output value to each connected node's inputSum."""
        # the node sends its output to the inputs of the nodes its connected to
        if self.layer != 0:#no sigmoid for the inputs and bias
            self.outputValue = self.sigmoid(self.inputSum)
        
        for i in range(len(self.outputConnections)):# for each connection
            if self.outputConnections[i].enabled:# dont do anything if not enabled
                self.outputConnections[i].toNode.inputSum += self.outputConnections[i].weight * self.outputValue
                #add the weighted output to the sum of the inputs of whatever node this node is connected to
    
    def isConnectedTo(self, node):
        """Return True if this node is connected to <node>."""
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
        """Python copy function definition."""
        clone = Node(int(self.number))
        clone.layer = int(self.layer)
        return clone
    
    def clone(self):
        """Return a clone of this node."""
        return self.__copy__()
    
    def save(self):
        """Return a list containing important data about this node."""
        connNodes = [[self.outputConnections[i].fromNode, self.outputConnections[i].toNode] for i in range(len(self.outputConnections))]
        return [self.number, self.layer, [node.number for node in sum(connNodes, [])]]

    @classmethod
    def load(cls, data):
        """Return a Node Object with data initialized from given data input."""
        node = cls(data[0])
        _, node.layer, connNodes = data
        node.outputConnections = []#Gets set with connectNodes in genome class
        return self
    pass

class connectionGene(object):
    """Object representing a connection between two node objects."""
    def __init__(self, fromNode, toNode, weight, inno):
        self.fromNode = fromNode
        self.toNode = toNode
        self.weight = weight
        self.enabled = True
        self.innovationNo = inno
        #each connection is given a innovation number to compare genomes
    
    def __repr__(self):
        """Return what this object should be represented by in the python interpriter."""
        return '<connectionGene Object>'
        
    def mutateWeight(self):
        """Mutate the weight of this connection."""
        change = random.randint(1, 10)
        if change == 1:#10% of the time completely change the self.weight
            self.weight = random.randint(-100, 100)/100
        else:#otherwise slightly change it
            self.weight += random.gauss(0, 1) / 50
            # Keep self.weight within bounds
            self.weight = min(self.weight, 1)
            self.weight = max(self.weight, -1)
    
    def __copy__(self):
        """Returns a copy of self."""
        return self.clone(self.fromNode, self.toNode)
    
    def clone(self, fromNode, toNode):
        """Returns a clone of self, but with potentially different fromNode and toNode values."""
        clone = connectionGene(fromNode, toNode, self.weight, self.innovationNo)
        clone.enabled = bool(self.enabled)
        return clone
    
    def save(self):
        """Returns a list containing important information about this connection."""
        #return [self.fromNode.save(), self.toNode.save(), float(self.weight), bool(self.enabled), int(self.innovationNo)]
        return [self.fromNode.number, self.toNode.number, float(self.weight), int(self.innovationNo), bool(self.enabled)]
    pass

class connectionHistory(object):
    """Object for storeing information about the past connections."""
    def __init__(self, fromNode, toNode, inno, innoNos):
        self.fromNode = int(fromNode)
        self.toNode = int(toNode)
        self.innovationNumber = int(inno)
        self.innovationNumbers = list(innoNos)
        # the innovation Numbers from the connections of the
        # genome which first had this mutation
        # ourself represents the genome and allows us to test if
        # another genoeme is the same
        # as ourself is before this connection was added
    
    def __repr__(self):
        """Return what this object should be represented by in the python interpriter."""
        return 'connectionHistory(%s, %s, %i, %s)' % (self.fromNode, self.toNode, self.innovationNumber, str(self.innovationNumbers))
    
    def matches(self, genome, fromNode, toNode):
        """Returns whether the genome matches the original genome and the connection is between the same nodes."""
        if len(genome.genes) == len(self.innovationNumbers):
            for gene in genome.genes:
                if not gene.innovationNo in self.innovationNumbers:
                    return False
            # If reached this far then innovationNumbers matches the gene
            # innovation numbers and the connection between the same nodes,
            # so it does match
            return True
        return False
    
    def clone(self):
        """Returns a clone of self."""
        clone = connectionHistory(self.fromNode, self.toNode, self.innovationNumber, self.innovationNumbers)
        return clone
    
    def save(self):
        """Returns a list of important information about this history object."""
        return [self.fromNode, self.toNode, self.innovationNumber, self.innovationNumbers]
    pass

class Genome(object):
    """Pretty much a brain, but it's called genome. Dunno. spagetii."""
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
        """Return what this object should be represented by in the python interpriter."""
        return '<Genome Object with %i layers, Bias node is %i, %i Nodes, and %i Genes>' % (self.layers, self.biasNode, len(self.nodes), len(self.genes))
    
    def getInnovationNumber(self, innovationHistory, fromNode, toNode):
        """Returns the innovation number for the new mutation."""
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
        """Adds the connections going out of a node to that node so that it can acess the next node during feeding forward."""
        # Clear connections
        for i in range(len(self.nodes)):
            self.nodes[i].outputConnections = []
        # For each connection, add the corrosponding gene to the node.
        for i in range(len(self.genes)):
            self.genes[i].fromNode.outputConnections.append(self.genes[i])
    
    def fullyConnect(self, innovationHistory):
        """Connects all nodes to eachother."""
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
        """Returns the node with a matching number, as sometimes the this.nodes will not be in order."""
        for node in self.nodes:
            if node.number == nodeNumber:
                return node
        return None
    
    def feedForward(self, inputValues):
        """Feeding in input values varo the NN and returning list."""
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
        """Sets up the Nural Network (finally found what that was reffering to) as a list of self.nodes in order to be engaged."""
        self.connectNodes()
        self.network = []
        # For each layer add the node in that layer, since layers cannot connect to themselves there is no need to order the nodes within a layer
        for layer in range(self.layers):
            for node in self.nodes:# For each node
                if node.layer == layer:# If the node is in that layer
                    self.network.append(node)# Add that node to the network
    
    def fullyConnected(self):
        """Returns whether the network is fully connected or not."""
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
            #for i in range(layer + 1, self.layers):# for each layer in front of this layer,
            for i in range(layer):
                nodesInFront += nodesInLayers[i]# add up nodes
            
            maxConnections += nodesInLayers[layer] * nodesInFront
        #if the number of connections is equal to the max number of connections possible then it is full
        return maxConnections <= len(self.genes)
    
    def randomConnectionNodesAreBad(self, r1, r2):
        """Returns True if the two given nodes, r1 and r2, are on the same layer or are connected to eachother. Bad code bullet for original name. Bad you."""
        if self.nodes[r1].layer == self.nodes[r2].layer:
            return True# if the nodes are in the same layer
        if self.nodes[r1].isConnectedTo(self.nodes[r2]):
            return True#if the nodes are already connected
        return False
    
    def addConnection(self, innovationHistory):
        """Adds a connection between two nodes which aren't currently connected."""
        # Cannot add a connection to a fully connected network
        if self.fullyConnected():
            print('Connection failed.')
            raise RuntimeError('Cannot add a connection to a fully connected network.')
        
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
        """Pick a random connection to create a node between."""
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
        self.getNode(newNodeNo).layer = self.genes[randomConnection].fromNode.layer + 1
        
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
        """Mutates the genome in random ways."""
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
        """Returns whether or not there is a gene matching the input innovation number in the input genome."""
        for i in range(len(parent2.genes)):
            if parent2.genes[i].innovationNo == innovationNumber:
                return i
        return None#no matching gene found
    
    def crossover(self, parent2):
        """Called when this genome is better than other parent."""
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
        """Prints out information about genome."""
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
        """Returns a copy of self."""
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
    
    def drawGenome(self, screen, startX, startY, w, h):
        """Draws the nurons in this genome to the screen."""
        raise NotImplemented('drawGenome has not been implemented, as the system is running headless at the moment.')
    
    def save(self):
        """Returns important information about this Genome Object."""
        genes = [gene.save() for gene in self.genes]
        nodes = [node.save() for node in self.nodes]
        return [self.inputs, self.outputs, genes, nodes, self.layers, self.nextNode, self.biasNode]
    
    @classmethod
    def load(cls, data):
        """Returns a new Genome Object based on save data input."""
        self = cls(*data[:2], False)
        self.inputs, self.outputs, genes, nodes, self.layers, self.nextNode, self.biasNode = data
        self.nodes = [Node.load(i) for i in nodes]
        tmpgenes = [[connectionGene(self.getNode(frm), self.getNode(to), w, i), e] for frm, to, w, i, e in genes]
        self.genes = []
        for gene, e in tmpgenes:
            gene.enabled = bool(e)
            self.genes.append(gene)
        #self.connectNodes() already called in generateNetwork
        self.generateNetwork()
        return self
    pass

class GameEntity(object):
    """Base Class for all entities. Stolen from my (me is CoolCat467) base2d module."""
    def __init__(self, world, name, image, **kwargs):
        self.world = world
        self.name = name
        self.image = image
        self.base_image = image
        
        self.location = Vector2()
        self.destination = Vector2()
        self.speed = 0
        self.scan = 100
        if not self.image is None:
            self.scan = int(get_surf_len(self.image)/2) + 2
        
        self.showhitbox = False
        self.show = True
        self.doprocess = True
        
        keys = list(kwargs.keys())
        if 'location' in keys:
            self.location = Vector2(*kwargs['location'])
        if 'destination' in keys:
            self.location = Vector2(*kwargs['destination'])
        if 'speed' in keys:
            self.speed = kwargs['speed']
        if 'hitbox' in keys:
            self.showhitbox = bool(kwargs['hitbox'])
        if 'scan' in keys:
            self.scan = int(kwargs['scan'])
        if 'show' in keys:
            self.show = bool(kwargs['show'])
        
        self.brain = None
        
        self.id = 0
    
    def __repr__(self):
        return '<%s GameEntity>' % self.name.title()
    
    def __str__(self):
        return self.__repr__
    
    def render(self, surface):
        """Render an entity and it's hitbox if showhitbox is True, and blit it to the surface"""
        x, y = list(self.location)
        try:
            x, y = float(x), float(y)
        except TypeError as e:
            print(x, y)
            print('TypeError in Render!')
            raise TypeError(str(e))
        w, h = self.image.get_size()
        if self.show:
            surface.blit(self.image, (x-w/2, y-h/2))
        if self.showhitbox:
            pygame.draw.rect(surface, [0]*3, self.get_col_rect(), 1)
            if self.scan:
                pygame.draw.circle(surface, [0, 0, 60], toint([x, y]), self.scan, 1)
        
    
    def process(self, time_passed):
        """Process brain and move according to time passed if speed > 0 and not at destination"""
        if self.doprocess:
            self.brain.think()
            
            if self.speed > 0 and self.location != self.destination:
                #vec_to_dest = self.destination - self.location
                #distance_to_dest = vec_to_dest.get_length()
                vec_to_dest = Vector2.from_points(self.location, self.destination)
                distance_to_dest = self.location.get_distance_to(self.destination)
                heading = vec_to_dest.get_normalized()
                # prevent going back and forward really fast once it make it close to destination
                travel_distance = min(distance_to_dest, (time_passed * self.speed))
                self.location += heading * round(travel_distance)
    
    def get_xywh(self):
        """Return x and y position and width and height of self.image for collision"""
        # Return x pos, y pos, width, and height for collision
        x, y = self.location
        w, h = (0, 0)
        if not self.image is None:
            w, h = self.image.get_size()
        x -= w/2
        y -= h/2
        return x, y, w, h
    
    def get_col_rect(self):
        """Return a rect for collision"""
        rect = pygame.rect.Rect(*self.get_xywh())
        return rect
    
    def is_over(self, point):
        """Return True if point is over self.image"""
        # Return True if a point is over image
        point_x, point_y = point
        x, y, w, h = self.get_xywh()
        
        in_x = point_x >= x and point_x < x + w
        in_y = point_y >= y and point_y < y + h
        
        return in_x and in_y
    
    def collision(self, sprite):
        """Return True if a sprite's image is over self.image"""
        # Return True if a sprite's image is over our image
        rs = self.get_col_rect()#rect self
        ro = sprite.get_col_rect()#rect other
        
        return bool(rs.colliderect(ro))
    
    def collide(self, entityname, action):
        """For every entity with the name of entityname, call action(self, entity)"""
        for entity in self.world.get_type(entityname):
            if entity is not None:
                if self.collision(entity):
                    action(self, entity)
    
    def relative_side(self, entity):
        """Return what side of an entity we are on, to be used with collision"""
        # THIS IS VERY BROKEN PLZ FIX
        wall = hasattr(entity, 'side')
        sides = ['top', 'right', 'bottom', 'left']# 45:'top right', 135:'bottom right', 225:'bottom left', 315:'top left'}
        sloc = self.location
        eloc = entity.location
        rect = entity.get_col_rect()
        vec = Vector2.from_points(sloc, eloc)
        if not wall:
            final = [i*90 for i in range(4)]
            first = [i*45 for i in range(8)]
            rsides = [rect.midtop, rect.midright, rect.midbottom, rect.midleft]
            #rsides = [Vector2(x, y) for x, y in rsides]
            #side_deg = [atan2(v.y, v.x) for v in rsides]
            side_deg = [round((heading_to_degrees( Vector2.from_points(entity.location, Vector2(x, y)).get_normalized()) + 360) % 360) for x, y in rsides]
            deg = 360 - round((heading_to_degrees(vec.get_normalized())) % 360)
            if deg <= 45:
                deg = 0
            sdeg = closest(round(deg), side_deg)
            num = side_deg.index(sdeg)
            
            return sides[num]
        elif wall:
            return entity.side
        return None
    pass

class BasePlayer(GameEntity):
    """Base class for a player object. Many functions simply pass instead of doing stuff."""
    def __init__(self, world, name, image, **kwargs):
        GameEntity.__init__(self, world, name, image, **kwargs)
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
    
    def start(self):
        """Function that gets called during initialization, sets brain to a Genome Object with self.genomeInputs, self.genomeOutputs as arguments."""
        self.brain = Genome(self.genomeInputs, self.genomeOutputs)

    def show(self, screen):
        """Renders this player to the screen."""
        self.render(screen)
    
    def move(self, time_passed_secconds):
        """Updates the player's position."""
        self.process(time_passed_secconds)
    
    def update(self, time_passed_secconds):
        """Updates the player."""
        self.move(time_passed_secconds)
    
    def look(self):
        pass
    
    def think(self):
        self.decision = self.brain.feedForward(self.vision)
        self.do = self.decision.index(max(self.decision))
        pass
    
    def clone(self):
        """Returns a clone of self."""
        clone = BasePlayer()
        clone.brain = self.brain.clone()
        clone.fitness = float(self.fitness)
        clone.brain.generateNetwork()
        clone.gen = int(self.gen)
        clone.bestScore = float(self.score)
        return clone
    
    def cloneForReplay(self):
        """Returns a clone for a non-existant replay."""
        return self.clone()
    
    def calculateFitness(self):
        """Calculates the fitness of the AI."""
        self.fitness = random.randint(0, 10)
    
    def crossover(self, parent2):
        """Returns a BasePlayer object by crossing over our brain and parent2's brain."""
        child = BasePlayer()
        child.brain = self.brain.crossover(parent2.brain)
        child.brain.generateNetwork()
        return child
    
    def save(self):
        """Returns a list containing important information about ourselves."""
        return [self.brain.save(), self.gen, self.dead, self.bestScore, self.score]
    
    @classmethod
    def load(cls, data):
        """Returns a BasePlayer Object with save data given."""
        self = cls()
        brain, self.gen, self.dead, self.bestScore, self.score = data
        self.genomeInputs, self.genomeOutputs = brain[:2]
        self.brain = Genome.load(brain)
        return self
    pass

class Species(object):
    """Species object, containing large groups of players."""
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
        """Return what this object should be represented by in the python interpriter."""
        return '<Species Object>'
    
    @staticmethod
    def getExcessDisjoint(brain1, brain2):
        """Returns the number of excess and disjoint genes."""
        matching = 0
        for gene1 in brain1.genes:
            for gene2 in brain2.genes:
                if gene1.innovationNo == gene2.innovationNo:
                    matching += 1
                    break
        # Return number of excess and disjoint genes
        return (len(brain1.genes) + len(brain2.genes) - 2) * matching
    
    @staticmethod
    def averageWeightDiff(brain1, brain2):
        """Returns the average weight difference between two brains."""
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
        """Returns if a genime is in this species."""
        excessAndDisjoint = self.getExcessDisjoint(genome, self.rep)
        averageWeightDiff = self.averageWeightDiff(genome, self.rep)
        largeGenomeNormalizer = max(len(genome.genes) - 20, 1)
        # compatibility formula
        compatibility = (self.excessCoeff * excessAndDisjoint / largeGenomeNormalizer) + (self.weightDiffCoeff * averageWeightDiff)
        return self.compatibilityThreshold > compatibility
    
    def addToSpecies(self, player):
        """Adds player to this species."""
        self.players.append(player)
    
    def sortSpecies(self):
        """Sorts the species by their fitness."""
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
        """Calculates the average fitness of this species."""
        if not len(self.players):
            self.averageFitness = 0
            return
        self.averageFitness = sum(player.fitness for player in self.players) / len(self.players)
    
    def fitnessSharing(self):
        """Divides each player's fitness by the number of players."""
        for i in range(len(self.players)):
            self.players[i].fitness /= len(self.players)
    
    def selectPlayer(self):
        """Selects a player based on it's fitness."""
        fitnessSum = sum([player.fitness for player in self.players])
        rand = random.randint(0, int(fitnessSum))
        runningSum = 0
        for player in self.players:
            runningSum += player.fitness
            if runningSum > rand:
                return player
        return self.players[0]
    
    def giveMeBaby(self, innovationHistory):
        """Returns a baby by either cloneing the best player or crossing over the two highest graded fitness players."""
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
        """Kill half of the players."""
        if len(self.players) > 2:
            self.players = self.players[int(len(self.players)/2):]
    
    def clone(self):
        """Returns a clone of self."""
        clone = Species()
        clone.players = [player.clone() for player in self.players]
        clone.bestFitness = float(self.bestFitness)
        clone.champ = self.champ.clone()
        clone.setAverage()
        clone.staleness = int(self.staleness)
        clone.excessCoeff = float(self.excessCoeff)
        clone.weightDiffCoeff = float(self.weightDiffCoeff)
        clone.compatibilityThreshold = float(self.compatibilityThreshold)
        return clone
    
    def __copy__(self):
        """Returns a copy of self."""
        return self.clone()
    
    def save(self):
        """Returns a list containing important information about this species."""
        players = [player.save() for player in self.players]
        champ = self.champ.save()
        rep = self.rep.save()
        return [players, self.bestFitness, champ, self.staleness, self.excessCoeff, self.weightDiffCoeff, self.compatibilityThreshold]
    pass

class Population(object):
    """Population Object, stores groups of species."""
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
        """Return what this object should be represented by in the python interpriter."""
        return '<Population Object with %i Players and %i Generations>' % (len(self.players), self.gen)
    
    def updateAlive(self):
        """Updates all of the players that are alive."""
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
        """Returns True if all the players are dead. :("""
        for player in self.players:
            if not player.dead:
                return False
        return True
    
    def setBestPlayer(self):
        """Sets the best player globally and for current generation."""
        if not (self.species and self.species[0].players):
            return
        tempBest = self.species[0].players[0]
        tempBest.gen = self.gen
        
        if tempBest.score >= self.bestScore:
            bestClone = tempBest.cloneForReplay()
            self.genPlayers.append(bestClone)
            #print stuff was here, removed
            self.bestScore = tempBest.score
            self.bestPlayer = bestClone
    
    def spectate(self):
        """Seperate players into species based on how similar they are to the leaders of the species in the previous generation."""
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
        """Calculate the fitness of each player."""
        for player in self.players:
            player.calculateFitness()
    
    def sortSpecies(self):
        """Sort the species to be ranked in fitness order, best first."""
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
        """For all the species but the top five, kill them all."""
        for s in range(5, len(self.species)):
            del self.species[s]
    
    def cullSpecies(self):
        """Kill off the bottom half of each species."""
        for s in self.species:
            s.cull()
            s.fitnessSharing()#Also while we're at it do fitness sharing
            s.setAverage()
    
    def killStaleSpecies(self):
        """Kills all species which haven't improved in 15 generations."""
        for i in range(len(self.species)-1, -1, -1):
            if self.species[i].staleness >= 15:
                del self.species[i]
    
    def getAvgFitnessSum(self):
        """Returns the sum of the average fitness for each species."""
        return sum([s.averageFitness for s in self.species])
    
    def killBadSpecies(self):
        """Kill species which are so bad they can't reproduce."""
        averageSum = self.getAvgFitnessSum()
        if not averageSum:
            return
        for i in range(len(self.species)-1, -1, -1):
            if self.species[i].averageFitness / averageSum * len(self.players) < 1:
                del self.species[i]
    
    def naturalSelection(self):
        """This function is called when all players in the player list are dead and a new generation needs to be made."""
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
        """Returns True if a player is in worlds...???"""
        b = self.batchNo * self.worldsPerBatch
        e = min((self.batchNo + 1) * self.worldsPerBatch, len(worlds))
        for i in range(b, e):
            if player.world == worlds[i]:
                return True
        return False
    
    def updateAliveInBatches(self, worlds, showBest):
        """Update all the players that are alive."""
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
        """For each world, call world.step(FPS, arg2, arg3)"""
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
    
    def clone(self):
        """Returns a clone of self."""
        clone = Population(len(self.players))
        clone.players = [player.clone() for player in self.players]
        clone.bestPlayer = self.bestPlayer.clone()
        clone.bestScore = float(self.bestScore)
        clone.globalBestScore = float(self.globalBestScore)
        clone.gen = int(self.gen)
        clone.innovationHistory = [ih.clone() for ih in self.innovationHistory]
        clone.genPlayers = [player.clone() for player in self.genPlayers]
        clone.species = [sep.clone() for sep in self.species]
        return clone
    
    def __copy__(self):
        """Returns a copy of self."""
        return self.clone()
    
    def save(self):
        """Returns a list containing all important data."""
        players = [player.save() for player in self.players]
        bestp = self.bestPlayer.save()
        innoh = [innohist.save() for innohist in self.innovationHistory]
        genplayers = [gplayer.save() for gplayer in self.genPlayers]
        species = [specie.save() for specie in self.species]
        return [players, bestp, self.bestScore, self.globalBestScore, self.gen, innoh, genplayers, species]
    
    @classmethod
    def load(cls, data):
        """Returns a Population Object using save data."""
        self = cls(len(data[0]))
        players, bestp, self.bestScore, self.globalBestScore, self.gen, innoh, genplayers, species = data
        self.players = [BasePlayer.load(pdat) for pdat in players]
        self.bestPlayer = BasePlayer.load(bestp)
        self.innovationHistory = [connectionHistory(*i) for i in innoh]
        return self
    pass

def typ(x):
    """Returns 0 if type is a list, returns 1 if type is a string, returns 2 if type is int or float, returns 3 if type is boolian, None if no matches."""
##    if hasattr(x, '__reversed__'):
##        return 0
##    elif hasattr(x, 'capitalize'):
##        return 1
##    elif hasattr(x, 'real'):
##        if str(x) in ('True', 'False'):
##            return 3
##        return 2
    thing = str(type(x))
    if thing in [str(type([])), str(type([]))]:
        return 0
    if thing == str(type(str('Why hello there'))):
        return 1
    if thing == str(type(bool(1))):
        return 3
    if thing == str(type(8675309)):
        return 2
    return None

def save(data, filename):
    """Save data to a file."""
    infl = lambda x: int(x) if round(float(x)) == float(x) else float(x)
    def sv(filedata, incount, lst):
        filedata.append(' '*incount+'{\n')
        incount += 2
        for thing in lst:
            t = typ(thing)
            if t == 0:
                if len(thing):
                    filedata, incount = sv(filedata, incount, thing)
                else:
                    filedata.append(' '*incount+'{}\n')
                #filedata.append(' '*incount+'/'+'&'.join(thing)+'/\n')
            elif t == 1:
                filedata.append(' '*incount+'$'+thing+'$\n')
            elif t == 2:
                filedata.append(' '*incount+'N'+str(infl(thing))+'N\n')
            elif t == 3:
                filedata.append(' '*incount+'@'+str(thing)+'@\n')
        incount -= 2
        filedata.append(' '*incount+'}\n')
        return filedata, incount
    filedata, _ = sv([], 0, data)
    with open(filename, 'w') as saveFile:
        saveFile.write(''.join(filedata))
        saveFile.close()

def load(filename):
    """Return data retreved from a file."""
    infl = lambda x: int(x) if round(float(x)) == float(x) else float(x)
    def ld(data, lst, idx):
        add = []
        while True:
            if idx >= len(lst):
                break
            line = lst[idx]
            sp = line.count(' ')
            text = line[sp:]
            t = text[0]
            if text == '{}':
                add.append([])
                idx += 1
                continue
            if t == '{':
                idx += 1
                fromNext, idx = ld(add, lst, idx)
                #print(fromNext, add)
                add.append(fromNext)
            elif t == '}':
                break
            elif t == '$':
                add.append(text[1:-1])
            elif t == 'N':
                add.append(infl(text[1:-1]))
            elif t == '@':
                add.append(True if text[1:-1] == 'True' else False)
            idx += 1
        return add, idx
    with open(filename, 'r') as loadFile:
        filedata = loadFile.read().splitlines()
        loadFile.close()
    data, _ = ld([], filedata, 0)
    return data[0]

if __name__ == '__main__':
    print('Starting example.')
    try:
        data = load('AI_Data.txt')
    except FileNotFoundError:
        cat = Population(5)
    else:
        print('AI Data loaded from AI_Data.txt')
        cat = Population.load(data)
    print('Running Natural Selection program 100 times...')
    [cat.naturalSelection() for i in range(100)]
    print('Natural Selection done.')
    print(cat)
    print('Saving AI data to AI_Data.txt...')
    save(cat.save(), 'AI_Data.txt')
    print('Done.')
