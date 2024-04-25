import pygame as pg
from pygame import *
import numpy as np
import asyncio
from time import sleep
import time
import tensorflow as tf
import copy
import contextlib
import random


class Bug:
    def __init__(self, w, h, x_pos=-1, y_pos=-1):
        self.width = w
        self.height = h

        self.x_pos = w/2
        self.y_pos = h/2
        if x_pos != -1:
            self.x_pos = x_pos
        if y_pos != -1:
            self.y_pos = y_pos

        self.speed = 75
        self.size = 25

        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

        self.brain = self.build_model()
        self.speedbrain = self.build_speed_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            # tf.keras.layers.Dense(units=2, activation='relu', input_shape=(2,)),
            tf.keras.layers.Dense(units=4, activation='linear')  # Three outputs for jet actions
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.load_weights('movement_weights.h5')
        return model
    
    def build_speed_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            # tf.keras.layers.Dense(units=2, activation='relu', input_shape=(2,)),
            tf.keras.layers.Dense(units=2, activation='softmax')  # Three outputs for jet actions
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def move(self, L, R, U, D, S, time):
        if S:
            self.speed += 100 * time
            if self.speed > 1000:
                self.speed = 1000
        else:
            self.speed -= 100 * time
            if self.speed < 0:
                self.speed = 0
        print(self.speed)
        to_move = self.speed * time
        if L:
            self.x_pos -= to_move
        if R:
            self.x_pos += to_move
        if U:
            self.y_pos -= to_move
        if D:
            self.y_pos += to_move

    def get_actions(self, distance_away, threshold = .5):
        pred = self.brain.predict(np.array([distance_away]), verbose=0)[0]
        decisions = [a > threshold for a in pred]
        return decisions   # [0.0, -150.0]

    def get_speed_action(self, distance_away):
        pred = self.speedbrain.predict(np.array([distance_away]), verbose=0)[0]
        if np.argmax(pred) == 0:
            return False
        else:
            return True

    def paint(self, screen):
        pg.draw.circle(screen, self.color, (self.x_pos, self.y_pos), self.size)

    def train(self, x_train, y_train, reward):
        # self.brain.fit(x_train, y_train, epochs=1, verbose=0)
        self.brain.train_on_batch(x_train, y_train, sample_weight=reward)

    def get_copy(self):
        copy_bug = Bug(self.width, self.height, self.x_pos, self.y_pos)
        copy_bug.brain = self.brain
        copy_bug.color = self.color
        return copy_bug


class Game:
    def __init__(self):
        self.width = 800
        self.height = 600

        self.p = [self.width * .5, self.height * .25]

        pg.init()
        self.screen = pg.display.set_mode((self.width, self.height))

        self.running = True

        self.bug = Bug(self.width+0, self.height+0)
        self.fps = 60
        self.bugs = []

        self.BUG = 0
        self.REWARD = 1

        self.text_font = pg.font.SysFont("Arial", 30)

    def paint_screen(self, bugs, gen):
        self.screen.fill((0, 0, 0))

        for bug_ind in range(len(bugs)):
            bug = bugs[bug_ind][self.BUG]
            bug.paint(self.screen)

        pg.draw.circle(self.screen, (200, 200, 200), (self.p[0], self.p[1]), 10)

        img = self.text_font.render(str(gen), True, (255, 255, 255))
        self.screen.blit(img, (25, 25))

        pg.display.flip()

    async def run():
        pass

    async def run_generation(self, bugs, steps, gen):
        pg.init()
        self.running = True
        quadrant = random.randint(0, 3)
        # self.p = [ random.randint(0, self.width), random.randint(0, self.height) ]
        self.p = [0, 0]
        if quadrant == 0 or quadrant == 1:
            self.p[1] = self.height * .25
        else:
            self.p[1] = self.height * .75
        if quadrant == 0 or quadrant == 2:
            self.p[0] = self.width * .25
        else:
            self.p[0] = self.width * .75

        last_times = [time.time() for _ in range(len(bugs))]

        for _ in range(steps):
            for event in pg.event.get():
                if event.type == QUIT:
                    self.running = False
            if self.running:
                for bug_ind in range(len(bugs)):
                    bug = bugs[bug_ind][self.BUG]

                    x_distance = self.p[0] - bug.x_pos
                    y_distance = self.p[1] - bug.y_pos
                    actions = bug.get_actions([x_distance, y_distance])
                    L, R, U, D = actions

                    S = bug.get_speed_action([x_distance, y_distance])
                    bug.move(L, R, U, D, S, time.time() - last_times[bug_ind])
                    last_times[bug_ind] = time.time()

                    distance_to_target = np.linalg.norm(np.array([self.p[0],self.p[1]]) - np.array([bug.x_pos, bug.y_pos]))
                    reward = 1.0 / (1.0 + distance_to_target)

                    bugs[bug_ind][self.REWARD] += reward

                self.paint_screen(bugs, gen)
                if len(bugs) < 3:
                    await asyncio.sleep(1 / self.fps)

        return bugs

    def get_best_bugs(self, bugs):
        bugs = sorted(bugs, key=lambda item: item[self.REWARD], reverse=True)
        return bugs[:4]
    
    def create_child(self, bug, mutation_rate=.1):
        child = bug.get_copy()

        weights = child.speedbrain.get_weights()

        for i, layer in enumerate(weights):
            weights[i] += np.random.normal(scale=mutation_rate, size=layer.shape)
        
        child.speedbrain.set_weights = weights

        return child

    def reset_bugs_position(self):
        print(self.bugs)
        for bug_ind in range(len(self.bugs)):
            print(self.bugs[bug_ind][self.BUG])
            self.bugs[bug_ind][self.BUG].x_pos = self.width/2
            self.bugs[bug_ind][self.BUG].y_pos = self.height/2

    def copy_bugs(self, bugs):
        new_bugs = []
        for bug_ind in range(len(bugs)):
            new_bugs.append([bugs[bug_ind][self.BUG].get_copy(), bugs[bug_ind][self.REWARD] + 0])
        return new_bugs

    async def train(self):
        bug_count = 10
        generations = 100

        # self.bugs = [(Bug(self.width, self.height), 0.0) for _ in range(bug_count)]
        self.bugs = [[Bug(self.width, self.height), 0.0] for _ in range(0, bug_count)]

        for generation in range(generations):
            if self.running:
                self.reset_bugs_position()
                print(f'\nStarting Generation {generation}!\n')
                
                new_bugs = await self.run_generation(self.copy_bugs(self.bugs), 25, generation)
                best_bugs = self.get_best_bugs(new_bugs)

                self.bugs = self.copy_bugs(best_bugs)

                self.bugs[0][self.BUG].speedbrain.save_weights('speed_weights.h5')

                self.bugs[0][self.REWARD] = 0
                self.bugs[1][self.REWARD] = 0
                self.bugs[2][self.REWARD] = 0
                self.bugs[3][self.REWARD] = 0
                for _ in range(1):
                    self.bugs.append([self.create_child(self.bugs[0][self.BUG].get_copy()), 0])
                # self.bugs.append([self.create_child(self.bugs[0][self.BUG].get_copy()), 0])
                for _ in range(1):
                    self.bugs.append([self.create_child(self.bugs[1][self.BUG].get_copy()), 0])
                for _ in range(1):
                    self.bugs.append([self.create_child(self.bugs[2][self.BUG].get_copy()), 0])
                for _ in range(1):
                    self.bugs.append([self.create_child(self.bugs[3][self.BUG].get_copy()), 0])
                # self.bugs.append([self.create_child(self.bugs[1][self.BUG].get_copy()), 0])
                for _ in range(2):
                    self.bugs.append([Bug(self.width, self.height), 0])

                print(f'\nGeneration {generation} Done!\n')
            else:
                break

        # self.bugs = dict(list(new_bugs.items())[:2])
        self.reset_bugs_position()
        self.bugs = self.get_best_bugs(self.copy_bugs(self.bugs))
        await self.run_generation(self.bugs, 200, 9999999)

        pg.quit()

    async def test(self):
        self.fps = 60
        self.running = True
        bugs = [[Bug(self.width, self.height), 0.0] for _ in range(0, 1)]
        bugs[0][self.BUG].speedbrain.load_weights('speed_weights.h5')
        # bugs[0][self.BUG].speed = 200
        last_times = [time.time() for _ in range(len(bugs))]

        while self.running:
            last_time = time.time()
            self.p = pg.mouse.get_pos()
            for event in pg.event.get():
                if event.type == QUIT:
                    self.running = False
            
            for bug_ind in range(len(bugs)):
                bug = bugs[bug_ind][self.BUG]

                x_distance = self.p[0] - bug.x_pos
                y_distance = self.p[1] - bug.y_pos
                actions = bug.get_actions([x_distance, y_distance])
                L, R, U, D = actions

                # distance_to_target = np.linalg.norm(np.array([self.p[0],self.p[1]]) - np.array([bug.x_pos, bug.y_pos]))
                S = bug.get_speed_action([x_distance, y_distance])
                print(S)
                bug.move(L, R, U, D, S, time.time() - last_times[bug_ind])
                last_times[bug_ind] = time.time()

            self.paint_screen(bugs, gen = 0)
            await asyncio.sleep(1 / self.fps)
            
            
            




async def main():
    game = Game()
    # await asyncio.gather(game.run_main(), game.train())
    test = True

    if test:
        await game.test()
    else:
        await game.train()

    # game.train()


if __name__ == '__main__':
    asyncio.run(main())

















