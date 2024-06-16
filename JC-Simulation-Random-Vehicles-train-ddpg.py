import pygame  # pip install pygame
import random
import threading
import time
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

#########################################################################################################
# Define constants
#########################################################################################################

defaultGreen = {0: 10, 1: 10, 2: 10, 3: 10}
defaultRed = 150
defaultYellow = 5

signals = []
noOfSignals = 4
currentGreen = 0   # Indicates which signal is green currently
nextGreen = (currentGreen + 1) % noOfSignals    # Indicates which signal will turn green next
currentYellow = 0   # Indicates whether yellow signal is on or off

# Define the average startup speeds (in m/s) for each vehicle type
speeds = {'car': 2.47, 'truck': 1.53, 'bus': 1.53, 'motorcycle': 4.21}

# Coordinates of vehicles' start
x = {'right': [0, 0, 0], 'down': [755, 727, 697], 'left': [1400, 1400, 1400], 'up': [602, 627, 657]}
y = {'right': [348, 372, 397], 'down': [0, 0, 0], 'left': [498, 466, 436], 'up': [800, 800, 800]}

vehicles = {'right': {0: [], 1: [], 2: [], 'crossed': 0}, 'down': {0: [], 1: [], 2: [], 'crossed': 0},
            'left': {0: [], 1: [], 2: [], 'crossed': 0}, 'up': {0: [], 1: [], 2: [], 'crossed': 0}}
vehicleTypes = {0: 'car', 1: 'bus', 2: 'truck', 3: 'motorcycle'}
directionNumbers = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

# Coordinates of signal image, timer, and vehicle count
signalCoods = [(530, 230), (810, 230), (810, 570), (530, 570)]
signalTimerCoods = [(530, 210), (810, 210), (810, 550), (530, 550)]

# Coordinates of stop lines
stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
defaultStop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}

# Gap between vehicles
stoppingGap = 25    # stopping gap
movingGap = 25   # moving gap

# set allowed vehicle types here
allowedVehicleTypes = {'car': True, 'bus': True, 'truck': True, 'motorcycle': True}
allowedVehicleTypesList = []
vehiclesTurned = {'right': {0: [], 1: [], 2: []}, 'down': {0: [], 1: [], 2: []}, 'left': {0: [], 1: [], 2: []},
                  'up': {0: [], 1: [], 2: []}}
vehiclesNotTurned = {'right': {0: [], 1: [], 2: []}, 'down': {0: [], 1: [], 2: []}, 'left': {0: [], 1: [], 2: []},
                     'up': {0: [], 1: [], 2: []}}
rotationAngle = 3
mid = {'right': {'x': 705, 'y': 445}, 'down': {'x': 695, 'y': 450}, 'left': {'x': 695, 'y': 425},
       'up': {'x': 695, 'y': 400}}
# set random or default green signal time here point me
randomGreenSignalTimer = False
# set random green signal time range here
randomGreenSignalTimerRange = [5, 10]

# Vehicle counters for each direction
vehicleCounters = {'right': 0, 'down': 0, 'left': 0, 'up': 0}
vehiclesLeftInDirection = {'right': 0, 'down': 0, 'left': 0, 'up': 0}

# Total vehicle counts for each direction
totalVehicles4Direction = {'right': 0, 'down': 0, 'left': 0, 'up': 0}

# DDPG related classes and functions
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q = torch.relu(self.l1(torch.cat([state, action], 1)))
        q = torch.relu(self.l2(q))
        return self.l3(q)

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.storage = []
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.storage, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def save(self, filename):
        np.save(filename, self.storage)

    def load(self, filename):
        self.storage = np.load(filename, allow_pickle=True).tolist()

class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).cuda()
        self.actor_target = Actor(state_dim, action_dim, max_action).cuda()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).cuda()
        self.critic_target = Critic(state_dim, action_dim).cuda()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer(max_size=1000000)
        self.batch_size = 64
        self.discount = 0.99
        self.tau = 0.005

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).cuda()
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self):
        if len(self.replay_buffer.storage) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).cuda()
        action = torch.FloatTensor(action).cuda()
        reward = torch.FloatTensor(reward).cuda()
        next_state = torch.FloatTensor(next_state).cuda()
        done = torch.FloatTensor(done).cuda()

        target_q = self.critic_target(next_state, self.actor_target(next_state))
        target_q = reward + ((1 - done) * self.discount * target_q).detach()

        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add((state, action, reward, next_state, done))

    def save(self, actor_path, critic_path, replay_buffer_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        self.replay_buffer.save(replay_buffer_path)

    def load(self, actor_path, critic_path, replay_buffer_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        self.replay_buffer.load(replay_buffer_path)

def generateVehicles():
    while(True):
        vehicle_type = random.choice(allowedVehicleTypesList)
        lane_number = random.randint(0,2)
        will_turn = 0
        if(lane_number == 0):
            temp = random.randint(0,99)
            if(temp<40):
                will_turn = 1
        elif(lane_number == 2):
            temp = random.randint(0,99)
            if(temp<40):
                will_turn = 1
        else: 
            will_turn = 0
        temp = random.randint(0,99)
        direction_number = 0
        dist = [25,50,75,100]
        if(temp<dist[0]):
            direction_number = 0
        elif(temp<dist[1]):
            direction_number = 1
        elif(temp<dist[2]):
            direction_number = 2
        elif(temp<dist[3]):
            direction_number = 3
        Vehicle(lane_number, vehicleTypes[vehicle_type], directionNumbers[direction_number], will_turn)
        vehiclesLeftInDirection[directionNumbers[direction_number]] += 1
        time.sleep(1)

pygame.init()
simulation = pygame.sprite.Group()

class TrafficSignal:
    def __init__(self, red, yellow, green):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.signalText = ""

class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction, will_turn):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        self.willTurn = will_turn
        self.turned = 0
        self.rotateAngle = 0
        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1
        self.crossedIndex = 0
        path = "images/" + direction + "/" + vehicleClass + ".png"
        self.originalImage = pygame.image.load(path)
        self.image = pygame.image.load(path)

        if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index - 1].crossed == 0:
            if direction == 'right':
                self.stop = vehicles[direction][lane][self.index - 1].stop - vehicles[direction][lane][self.index - 1].image.get_rect().width - stoppingGap
            elif direction == 'left':
                self.stop = vehicles[direction][lane][self.index - 1].stop + vehicles[direction][lane][self.index - 1].image.get_rect().width + stoppingGap
            elif direction == 'down':
                self.stop = vehicles[direction][lane][self.index - 1].stop - vehicles[direction][lane][self.index - 1].image.get_rect().height - stoppingGap
            elif direction == 'up':
                self.stop = vehicles[direction][lane][self.index - 1].stop + vehicles[direction][lane][self.index - 1].image.get_rect().height + stoppingGap
        else:
            self.stop = defaultStop[direction]

        if direction == 'right':
            temp = self.image.get_rect().width + stoppingGap
            x[direction][lane] -= temp
        elif direction == 'left':
            temp = self.image.get_rect().width + stoppingGap
            x[direction][lane] += temp
        elif direction == 'down':
            temp = self.image.get_rect().height + stoppingGap
            y[direction][lane] -= temp
        elif direction == 'up':
            temp = self.image.get_rect().height + stoppingGap
            y[direction][lane] += temp

        simulation.add(self)

    def render(self, screen):
        screen.blit(self.image, (self.x, self.y))

    def move(self):
        global vehicleCounters, vehiclesLeftInDirection

        if self.direction == 'right':
            self.handle_right_direction()
        elif self.direction == 'down':
            self.handle_down_direction()
        elif self.direction == 'left':
            self.handle_left_direction()
        elif self.direction == 'up':
            self.handle_up_direction()

    def handle_right_direction(self):
        if self.crossed == 0 and self.x + self.image.get_rect().width > stopLines[self.direction]:
            self.crossed = 1
            vehicles[self.direction]['crossed'] += 1
            vehicleCounters[self.direction] += 1
            vehiclesLeftInDirection[self.direction] -= 1
            if self.willTurn == 0:
                vehiclesNotTurned[self.direction][self.lane].append(self)
                self.crossedIndex = len(vehiclesNotTurned[self.direction][self.lane]) - 1

        if self.willTurn == 1:
            if self.lane == 0:
                self.turn_right_lane_1()
            elif self.lane == 2:
                self.turn_right_lane_3()
        else:
            self.go_straight_right()

    def handle_down_direction(self):
        if self.crossed == 0 and self.y + self.image.get_rect().height > stopLines[self.direction]:
            self.crossed = 1
            vehicles[self.direction]['crossed'] += 1
            vehicleCounters[self.direction] += 1
            vehiclesLeftInDirection[self.direction] -= 1
            if self.willTurn == 0:
                vehiclesNotTurned[self.direction][self.lane].append(self)
                self.crossedIndex = len(vehiclesNotTurned[self.direction][self.lane]) - 1

        if self.willTurn == 1:
            if self.lane == 0:
                self.turn_down_lane_1()
            elif self.lane == 2:
                self.turn_down_lane_3()
        else:
            self.go_straight_down()

    def handle_left_direction(self):
        if self.crossed == 0 and self.x < stopLines[self.direction]:
            self.crossed = 1
            vehicles[self.direction]['crossed'] += 1
            vehicleCounters[self.direction] += 1
            vehiclesLeftInDirection[self.direction] -= 1
            if self.willTurn == 0:
                vehiclesNotTurned[self.direction][self.lane].append(self)
                self.crossedIndex = len(vehiclesNotTurned[self.direction][self.lane]) - 1

        if self.willTurn == 1:
            if self.lane == 0:
                self.turn_left_lane_1()
            elif self.lane == 2:
                self.turn_left_lane_3()
        else:
            self.go_straight_left()

    def handle_up_direction(self):
        if self.crossed == 0 and self.y < stopLines[self.direction]:
            self.crossed = 1
            vehicles[self.direction]['crossed'] += 1
            vehicleCounters[self.direction] += 1
            vehiclesLeftInDirection[self.direction] -= 1
            if self.willTurn == 0:
                vehiclesNotTurned[self.direction][self.lane].append(self)
                self.crossedIndex = len(vehiclesNotTurned[self.direction][self.lane]) - 1

        if self.willTurn == 1:
            if self.lane == 0:
                self.turn_up_lane_1()
            elif self.lane == 2:
                self.turn_up_lane_3()
        else:
            self.go_straight_up()

    def turn_right_lane_1(self):
        if self.crossed == 0 or self.x + self.image.get_rect().width < stopLines[self.direction] + 40:
            if (self.x + self.image.get_rect().width <= self.stop or (currentGreen == 0 and currentYellow == 0) or self.crossed == 1) and (
                    self.index == 0 or self.x + self.image.get_rect().width < (vehicles[self.direction][self.lane][self.index - 1].x - movingGap) or
                    vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                self.x += self.speed
        else:
            if self.turned == 0:
                self.rotateAngle += rotationAngle
                self.image = pygame.transform.rotate(self.originalImage, self.rotateAngle)
                self.x += 2.4
                self.y -= 2.8
                if self.rotateAngle == 90:
                    self.turned = 1
                    vehiclesTurned[self.direction][self.lane].append(self)
                    self.crossedIndex = len(vehiclesTurned[self.direction][self.lane]) - 1
            else:
                if self.crossedIndex == 0 or (
                        self.y > (vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].y + vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].image.get_rect().height + movingGap)):
                    self.y -= self.speed

    def turn_right_lane_3(self):
        if self.crossed == 0 or self.x + self.image.get_rect().width < mid[self.direction]['x'] + 40:
            if (self.x + self.image.get_rect().width <= self.stop or (currentGreen == 0 and currentYellow == 0) or self.crossed == 1) and (
                    self.index == 0 or self.x + self.image.get_rect().width < (vehicles[self.direction][self.lane][self.index - 1].x - movingGap) or
                    vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                self.x += self.speed
        else:
            if self.turned == 0:
                self.rotateAngle -= rotationAngle  # Ensure correct direction of rotation
                self.image = pygame.transform.rotate(self.originalImage, self.rotateAngle)
                self.x += 2.0
                self.y += 1.8
                if self.rotateAngle == -90:  # Ensure correct rotation angle
                    self.turned = 1
                    vehiclesTurned[self.direction][self.lane].append(self)
                    self.crossedIndex = len(vehiclesTurned[self.direction][self.lane]) - 1
            else:
                if self.crossedIndex == 0 or ((self.y + self.image.get_rect().height) < (vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].y - movingGap)):
                    self.y += self.speed

    def turn_down_lane_1(self):
        if self.crossed == 0 or self.y + self.image.get_rect().height < stopLines[self.direction] + 50:
            if (self.y + self.image.get_rect().height <= self.stop or (currentGreen == 1 and currentYellow == 0) or self.crossed == 1) and (
                    self.index == 0 or self.y + self.image.get_rect().height < (vehicles[self.direction][self.lane][self.index - 1].y - movingGap) or
                    vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                self.y += self.speed
        else:
            if self.turned == 0:
                self.rotateAngle += rotationAngle
                self.image = pygame.transform.rotate(self.originalImage, self.rotateAngle)
                self.x += 1.2
                self.y += 1.8
                if self.rotateAngle == 90:
                    self.turned = 1
                    vehiclesTurned[self.direction][self.lane].append(self)
                    self.crossedIndex = len(vehiclesTurned[self.direction][self.lane]) - 1
            else:
                if self.crossedIndex == 0 or ((self.x + self.image.get_rect().width) < (vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].x - movingGap)):
                    self.x += self.speed

    def turn_down_lane_3(self):
        if self.crossed == 0 or self.y + self.image.get_rect().height < mid[self.direction]['y'] + 40:
            if (self.y + self.image.get_rect().height <= self.stop or (currentGreen == 1 and currentYellow == 0) or self.crossed == 1) and (
                    self.index == 0 or self.y + self.image.get_rect().height < (vehicles[self.direction][self.lane][self.index - 1].y - movingGap) or
                    vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                self.y += self.speed
        else:
            if self.turned == 0:
                self.rotateAngle -= rotationAngle  # Ensure correct direction of rotation
                self.image = pygame.transform.rotate(self.originalImage, self.rotateAngle)
                self.x -= 2.5
                self.y += 2.0
                if self.rotateAngle == -90:  # Ensure correct rotation angle
                    self.turned = 1
                    vehiclesTurned[self.direction][self.lane].append(self)
                    self.crossedIndex = len(vehiclesTurned[self.direction][self.lane]) - 1
            else:
                if self.crossedIndex == 0 or (self.x > (vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].x + vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].image.get_rect().width + movingGap)):
                    self.x -= self.speed

    def turn_left_lane_1(self):
        if self.crossed == 0 or self.x > stopLines[self.direction] - 70:
            if (self.x >= self.stop or (currentGreen == 2 and currentYellow == 0) or self.crossed == 1) and (
                    self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index - 1].x + vehicles[self.direction][self.lane][self.index - 1].image.get_rect().width + movingGap) or
                    vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                self.x -= self.speed
        else:
            if self.turned == 0:
                self.rotateAngle += rotationAngle
                self.image = pygame.transform.rotate(self.originalImage, self.rotateAngle)
                self.x -= 1
                self.y += 1.2
                if self.rotateAngle == 90:
                    self.turned = 1
                    vehiclesTurned[self.direction][self.lane].append(self)
                    self.crossedIndex = len(vehiclesTurned[self.direction][self.lane]) - 1
            else:
                if self.crossedIndex == 0 or ((self.y + self.image.get_rect().height) < (vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].y - movingGap)):
                    self.y += self.speed

    def turn_left_lane_3(self):
        if self.crossed == 0 or self.x > mid[self.direction]['x'] - 40:
            if (self.x >= self.stop or (currentGreen == 2 and currentYellow == 0) or self.crossed == 1) and (
                    self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index - 1].x + vehicles[self.direction][self.lane][self.index - 1].image.get_rect().width + movingGap) or
                    vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                self.x -= self.speed
        else:
            if self.turned == 0:
                self.rotateAngle -= rotationAngle  # Ensure correct direction of rotation
                self.image = pygame.transform.rotate(self.originalImage, self.rotateAngle)
                self.x -= 1.8
                self.y -= 2.5
                if self.rotateAngle == -90:  # Ensure correct rotation angle
                    self.turned = 1
                    vehiclesTurned[self.direction][self.lane].append(self)
                    self.crossedIndex = len(vehiclesTurned[self.direction][self.lane]) - 1
            else:
                if self.crossedIndex == 0 or (self.y > (vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].y + vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].image.get_rect().height + movingGap)):
                    self.y -= self.speed

    def turn_up_lane_1(self):
        if self.crossed == 0 or self.y > stopLines[self.direction] - 60:
            if (self.y >= self.stop or (currentGreen == 3 and currentYellow == 0) or self.crossed == 1) and (
                    self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index - 1].y + vehicles[self.direction][self.lane][self.index - 1].image.get_rect().height + movingGap) or
                    vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                self.y -= self.speed
        else:
            if self.turned == 0:
                self.rotateAngle += rotationAngle
                self.image = pygame.transform.rotate(self.originalImage, self.rotateAngle)
                self.x -= 2
                self.y -= 1.2
                if self.rotateAngle == 90:
                    self.turned = 1
                    vehiclesTurned[self.direction][self.lane].append(self)
                    self.crossedIndex = len(vehiclesTurned[self.direction][self.lane]) - 1
            else:
                if self.crossedIndex == 0 or (self.x > (vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].x + vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].image.get_rect().width + movingGap)):
                    self.x -= self.speed

    def turn_up_lane_3(self):
        if self.crossed == 0 or self.y > mid[self.direction]['y'] - 40:
            if (self.y >= self.stop or (currentGreen == 3 and currentYellow == 0) or self.crossed == 1) and (
                    self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index - 1].y + vehicles[self.direction][self.lane][self.index - 1].image.get_rect().height + movingGap) or
                    vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                self.y -= self.speed
        else:
            if self.turned == 0:
                self.rotateAngle -= rotationAngle  # Ensure correct direction of rotation
                self.image = pygame.transform.rotate(self.originalImage, self.rotateAngle)
                self.x -= 1.6
                self.y -= 0.8
                if self.rotateAngle == -90:  # Ensure correct rotation angle
                    self.turned = 1
                    vehiclesTurned[self.direction][self.lane].append(self)
                    self.crossedIndex = len(vehiclesTurned[self.direction][self.lane]) - 1
            else:
                if self.crossedIndex == 0 or (self.x < (vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].x - vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].image.get_rect().width - movingGap)):
                    self.x += self.speed

    def go_straight_right(self):
        if self.crossed == 0:
            if (self.x + self.image.get_rect().width <= self.stop or (currentGreen == 0 and currentYellow == 0)) and (
                    self.index == 0 or self.x + self.image.get_rect().width < (vehicles[self.direction][self.lane][self.index - 1].x - movingGap)):
                self.x += self.speed
        else:
            if self.crossedIndex == 0 or (self.x + self.image.get_rect().width < (vehiclesNotTurned[self.direction][self.lane][self.crossedIndex - 1].x - movingGap)):
                self.x += self.speed

    def go_straight_down(self):
        if self.crossed == 0:
            if (self.y + self.image.get_rect().height <= self.stop or (currentGreen == 1 and currentYellow == 0)) and (
                    self.index == 0 or self.y + self.image.get_rect().height < (vehicles[self.direction][self.lane][self.index - 1].y - movingGap)):
                self.y += self.speed
        else:
            if self.crossedIndex == 0 or (self.y + self.image.get_rect().height < (vehiclesNotTurned[self.direction][self.lane][self.crossedIndex - 1].y - movingGap)):
                self.y += self.speed

    def go_straight_left(self):
        if self.crossed == 0:
            if (self.x >= self.stop or (currentGreen == 2 and currentYellow == 0)) and (
                    self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index - 1].x + vehicles[self.direction][self.lane][self.index - 1].image.get_rect().width + movingGap)):
                self.x -= self.speed
        else:
            if self.crossedIndex == 0 or (self.x > (vehiclesNotTurned[self.direction][self.lane][self.crossedIndex - 1].x + vehiclesNotTurned[self.direction][self.lane][self.crossedIndex - 1].image.get_rect().width + movingGap)):
                self.x -= self.speed

    def go_straight_up(self):
        if self.crossed == 0:
            if (self.y >= self.stop or (currentGreen == 3 and currentYellow == 0)) and (
                    self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index - 1].y + vehicles[self.direction][self.lane][self.index - 1].image.get_rect().height + movingGap)):
                self.y -= self.speed
        else:
            if self.crossedIndex == 0 or (self.y > (vehiclesNotTurned[self.direction][self.lane][self.crossedIndex - 1].y + vehiclesNotTurned[self.direction][self.lane][self.crossedIndex - 1].image.get_rect().height + movingGap)):
                self.y -= self.speed

# Initialization of signals with default values
def initialize():
    minTime = randomGreenSignalTimerRange[0]
    maxTime = randomGreenSignalTimerRange[1]
    if(randomGreenSignalTimer):
        ts1 = TrafficSignal(0, defaultYellow, random.randint(minTime,maxTime))
        signals.append(ts1)
        ts2 = TrafficSignal(ts1.red+ts1.yellow+ts1.green, defaultYellow, random.randint(minTime,maxTime))
        signals.append(ts2)
        ts3 = TrafficSignal(defaultRed, defaultYellow, random.randint(minTime,maxTime))
        signals.append(ts3)
        ts4 = TrafficSignal(defaultRed, defaultYellow, random.randint(minTime,maxTime))
        signals.append(ts4)
    else:
        ts1 = TrafficSignal(0, defaultYellow, defaultGreen[0])
        signals.append(ts1)
        ts2 = TrafficSignal(ts1.yellow+ts1.green, defaultYellow, defaultGreen[1])
        signals.append(ts2)
        ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen[2])
        signals.append(ts3)
        ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen[3])
        signals.append(ts4)
    repeat()

# Initialize DDPG agent
state_dim = 4  # Number of vehicle types
action_dim = 1  # Green light duration
max_action = 60  # Maximum green light duration
agent = DDPG(state_dim, action_dim, max_action)

if os.path.exists("ddpg_actor.pth") and os.path.exists("ddpg_critic.pth") and os.path.exists("replay_buffer.npy"):
    agent.load("ddpg_actor.pth", "ddpg_critic.pth", "replay_buffer.npy")
else:
    agent = DDPG(state_dim, action_dim, max_action)

def updateGreenSignalTiming():
    lane_number = 3
    lane_length = 100
    scaling_factor = 20

    for i in range(noOfSignals):
        direction = directionNumbers[i]
        vehicle_count = totalVehicles4Direction[direction] - vehicleCounters[direction]
        
        # Calculate the number of each vehicle type left on the lane
        N_car = sum(1 for lane in range(3) for vehicle in vehicles[direction][lane] if vehicle.vehicleClass == 'car')
        N_motorcycle = sum(1 for lane in range(3) for vehicle in vehicles[direction][lane] if vehicle.vehicleClass == 'motorcycle')
        N_truck = sum(1 for lane in range(3) for vehicle in vehicles[direction][lane] if vehicle.vehicleClass == 'truck')
        N_bus = sum(1 for lane in range(3) for vehicle in vehicles[direction][lane] if vehicle.vehicleClass == 'bus')

        # Deduct the vehicles of each type that have already crossed
        N_car = max(N_car - sum(1 for lane in range(3) for vehicle in vehicles[direction][lane] if vehicle.vehicleClass == 'car' and vehicle.crossed), 0)
        N_motorcycle = max(N_motorcycle - sum(1 for lane in range(3) for vehicle in vehicles[direction][lane] if vehicle.vehicleClass == 'motorcycle' and vehicle.crossed), 0)
        N_truck = max(N_truck - sum(1 for lane in range(3) for vehicle in vehicles[direction][lane] if vehicle.vehicleClass == 'truck' and vehicle.crossed), 0)
        N_bus = max(N_bus - sum(1 for lane in range(3) for vehicle in vehicles[direction][lane] if vehicle.vehicleClass == 'bus' and vehicle.crossed), 0)

        # State representation for DDPG
        state = np.array([N_car, N_motorcycle, N_truck, N_bus])

        # Select action using DDPG
        action = agent.select_action(state)
        green_signal_timing = action[0]  # DDPG action

        print(f"Direction {direction}: DDPG green_signal_timing = {green_signal_timing}")

        if green_signal_timing == 0:
            signals[i].green = 0  # Skip green signal timing if no vehicles
            signals[i].yellow = 0  # Skip yellow signal timing if no vehicles
        else:
            signals[i].green = math.ceil(green_signal_timing)
            signals[i].yellow = defaultYellow  # Maintain yellow signal timing

        print(f"Direction {direction}: Assigned green time = {signals[i].green}s")

# Update the repeat function to call updateGreenSignalTiming at the end
def repeat():
    global currentGreen, currentYellow, nextGreen

    updateGreenSignalTiming()
    print('\n')
    print(f"{totalVehicles4Direction[directionNumbers[currentGreen]] - vehicleCounters[directionNumbers[currentGreen]]} of vehicles on the lane")
    print(f"Direction {directionNumbers[currentGreen]} assigned {signals[currentGreen].green}s for green signal timing")

    while signals[currentGreen].green > 0:   # while the timer of current green signal is not zero
        updateValues()
        time.sleep(1)
    currentYellow = 1   # set yellow signal on
    # reset stop coordinates of lanes and vehicles
    for i in range(3):
        for vehicle in vehicles[directionNumbers[currentGreen]][i]:
            vehicle.stop = defaultStop[directionNumbers[currentGreen]]
    while signals[currentGreen].yellow > 0:  # while the timer of current yellow signal is not zero
        updateValues()
        time.sleep(1)
    currentYellow = 0   # set yellow signal off

    vehiclesLeftInDirection[directionNumbers[currentGreen]] = totalVehicles4Direction[directionNumbers[currentGreen]] - vehicleCounters[directionNumbers[currentGreen]]
    print(f"Direction {directionNumbers[currentGreen]} turned red. Vehicles passed: {vehicleCounters[directionNumbers[currentGreen]]}, Vehicles left in direction: {vehiclesLeftInDirection[directionNumbers[currentGreen]]}")

    # reset all signal times of current signal to default/random times
    if randomGreenSignalTimer:
        signals[currentGreen].green = random.randint(randomGreenSignalTimerRange[0], randomGreenSignalTimerRange[1])
    else:
        signals[currentGreen].green = defaultGreen[currentGreen]
    signals[currentGreen].yellow = defaultYellow
    signals[currentGreen].red = defaultRed

    currentGreen = nextGreen  # set next signal as green signal
    nextGreen = (currentGreen + 1) % noOfSignals    # set next green signal
    signals[nextGreen].red = signals[currentGreen].yellow + signals[currentGreen].green    # set the red time of next to next signal as (yellow time + green time) of next signal

    # Train DDPG
    agent.train()

    repeat()

def updateValues():
    for i in range(noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                signals[i].green -= 1
            else:
                signals[i].yellow -= 1
        else:
            signals[i].red -= 1

def save_ddpg_model():
    agent.save("ddpg_actor.pth", "ddpg_critic.pth", "replay_buffer.npy")

class Main:
    global allowedVehicleTypesList
    i = 0
    for vehicleType in allowedVehicleTypes:
        if allowedVehicleTypes[vehicleType]:
            allowedVehicleTypesList.append(i)
        i += 1
    thread1 = threading.Thread(name="initialization", target=initialize, args=())    # initialization
    thread1.daemon = True
    thread1.start()

    # Colours
    black = (0, 0, 0)
    white = (255, 255, 255)

    # Screensize
    screenWidth = 1400
    screenHeight = 800
    screenSize = (screenWidth, screenHeight)

    # Setting background image i.e. image of intersection
    background = pygame.image.load('images/3-lane-4-waysIntersection.png')
    # background = pygame.image.load('images/4-waysIntersection.png')

    screen = pygame.display.set_mode(screenSize)
    pygame.display.set_caption("SIMULATION")

    # Loading signal images and font
    redSignal = pygame.image.load('images/signals/red.png')
    yellowSignal = pygame.image.load('images/signals/yellow.png')
    greenSignal = pygame.image.load('images/signals/green.png')
    font = pygame.font.Font(None, 30)
    thread2 = threading.Thread(name="generateVehicles", target=generateVehicles, args=())    # Generating vehicles
    thread2.daemon = True
    thread2.start()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_ddpg_model()
                sys.exit()

        screen.blit(background, (0, 0))   # display background in simulation
        for i in range(noOfSignals):  # display signal and set timer according to current status: green, yellow, or red
            if i == currentGreen:
                if currentYellow == 1:
                    signals[i].signalText = signals[i].yellow
                    screen.blit(yellowSignal, signalCoods[i])
                else:
                    signals[i].signalText = signals[i].green
                    screen.blit(greenSignal, signalCoods[i])
            else:
                if signals[i].red <= 10:
                    signals[i].signalText = signals[i].red
                else:
                    signals[i].signalText = "---"
                screen.blit(redSignal, signalCoods[i])
        signalTexts = ["", "", "", ""]

        # display signal timer
        for i in range(noOfSignals):
            signalTexts[i] = font.render(str(signals[i].signalText), True, white, black)
            screen.blit(signalTexts[i], signalTimerCoods[i])

        # display the vehicles
        for vehicle in simulation:
            screen.blit(vehicle.image, [vehicle.x, vehicle.y])
            vehicle.move()
        pygame.display.update()

if __name__ == "__main__":
    Main()
