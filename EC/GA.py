# -*- coding: utf-8 -*-
import random
import os
import numpy as np

from GAN.options import Options
from GAN.solvers import create_solver
from FER.recognizer import Recognizer
from EC.individual import Individual


class GA(object):

    def __init__(self, aCrossRate, aMutationRage, aLifeCount, aGeneLength, aTar, aInterpolation):

        self.opt = Options().parse()
        self.solver = create_solver(self.opt)
        self.solver.init_test_setting(self.opt)
        self.recognizer = Recognizer()
        self.croessRate = aCrossRate
        self.mutationRate = aMutationRage
        self.lifeCount = aLifeCount
        self.geneLength = aGeneLength
        self.interpolation = aInterpolation - 1
        self.tar = aTar
        self.pop = []  # 种群
        self.best = None  # 保存这一代中最好的个体
        self.generation = 0
        self.bounds = 0.0  # 适配值之和，用于选择是计算概率

        self.initPopulation()

    def initPopulation(self):
        self.pop = []
        self.best = None
        self.generation = 0
        for i in range(self.lifeCount):
            gene = []
            for j in range(self.geneLength):
                gene.append(random.random())
            individual = Individual(gene)
            self.pop.append(individual)

    def evalResnum(self, res, tar):
        score = 0
        for i in range(7):
            score += (res[i] - tar[i]) * (res[i] - tar[i])
        return score

    def saveBest(self):
        self.solver.expression = np.array(self.best.gene)
        img = self.solver.test_ops()
        self.recognizer.recognize(img).tolist()
        self.solver.test_save_imgs(img, "best")
        self.recognizer.save_res_img("best_rec")

    def eval(self, individual, i):
        path = str(self.generation) + "/" + str(i)
        exp = np.array(individual)
        self.solver.expression = exp
        img = self.solver.test_ops()
        res = self.recognizer.recognize(img).tolist()
        self.solver.test_save_imgs(img, path)
        self.recognizer.save_res_img(path + "_rec")
        return (3 - self.evalResnum(res, self.tar)) / 3

    def judge(self):
        self.bounds = 0.0
        self.best = self.pop[0]
        i = 0
        for individual in self.pop:
            individual.score = self.eval(individual.gene, i)
            self.bounds += individual.score
            if self.best.score < individual.score:
                self.best = individual
            i += 1

    def cross(self, parent1, parent2):
        index1 = random.randint(0, self.geneLength - 1)
        index2 = random.randint(index1, self.geneLength - 1)
        newGene = []
        for i in range(self.geneLength):
            if i < index1 or i > index2:
                newGene.append(parent1.gene[i])
            else:
                newGene.append(parent2.gene[i])
        return newGene

    def mutation(self, gene):
        index = random.randint(0, self.geneLength - 1)
        index2 = random.randint(0, 1)

        newGene = gene[:]

        if index2 == 0:
            newGene[index] /= 2
        else:
            newGene[index] *= 2
        return newGene

    def getOne(self):
        r = random.uniform(0, self.bounds)
        for life in self.pop:
            r -= life.score
            if r <= 0:
                return life

    def newChild(self):
        parent1 = self.getOne()
        rate = random.random()

        if rate < self.croessRate:
            parent2 = self.getOne()
            gene = self.cross(parent1, parent2)
        else:
            gene = parent1.gene
        rate = random.random()
        if rate < self.mutationRate:
            gene = self.mutation(gene)

        return Individual(gene)

    def next(self):

        newLives = []
        newLives.append(self.best)  # 把最好的个体加入下一代
        while len(newLives) < self.lifeCount:
            newLives.append(self.newChild())
        self.pop = newLives
        self.generation += 1
        self.judge()

    def run(self):

        os.system("mkdir results/0")
        print("generation: 0")
        self.initPopulation()
        self.judge()

        for i in range(self.interpolation):
            print("generation: " + str(i + 1))
            os.system("mkdir results/" + str(i + 1))
            self.next()
        self.saveBest()

        return self.pop


