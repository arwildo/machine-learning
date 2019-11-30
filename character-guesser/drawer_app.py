#!/bin/python3
from tkinter import *
import tensorflow as tf
import matplotlib.pyplot as plt
import pygame


class pixel(object):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = (255, 255, 255)
        self.neighbors = []

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, (self.x, self.y,
            self.x + self.width, self.y + self.height))

    def getNeighbors(self, g):
        j = self.x // 20
        i = self.y // 20
        rows = 28
        cols = 28
        

