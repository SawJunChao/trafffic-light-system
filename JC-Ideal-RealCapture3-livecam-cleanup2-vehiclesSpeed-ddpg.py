import tkinter as tk
from tkinter import Label, Frame, Canvas
from threading import Thread
import cv2
from PIL import Image, ImageTk
import time
import torch
import os
from ultralytics import YOLO
import logging
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

# DDPG
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
from collections import deque
# DDPG

# Load the YOLO model
model = YOLO("yolov8n_trained.pt").to('cuda')

# Define vehicle types (COCO dataset IDs)
vehicleTypes = {2: 'car', 7: 'truck', 5: 'bus', 3: 'motorcycle'}

# Speeds dictionary
speeds = {'car': 2.47, 'truck': 1.53, 'bus': 1.53, 'motorcycle': 4.21}

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

    def save(self, filename):
        torch.save(self.actor.state_dict(), os.path.join(filename + "_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(filename + "_critic.pth"))
        torch.save(self.actor_target.state_dict(), os.path.join(filename, "actor_target.pth"))
        torch.save(self.critic_target.state_dict(), os.path.join(filename, "critic_target.pth"))
        self.replay_buffer.save(os.path.join(filename, 'replay_buffer.npy'))

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.critic.load_state_dict(torch.load(filename + "_critic.pth"))
        self.actor_target.load_state_dict(torch.load(filename + "_actor.pth"))
        self.critic_target.load_state_dict(torch.load(filename + "_critic.pth"))
        self.replay_buffer.load(os.path.join(filename, 'replay_buffer.npy'))

class TrafficLightGUI:
    def __init__(self, master, row, col):
        self.master = master
        self.junction_frame = Frame(master, width=100, height=250)
        self.junction_frame.grid(row=row, column=col, padx=10, pady=10)
        self.junction_frame.grid_propagate(False)

        self.canvas = Canvas(self.junction_frame, width=100, height=250, bg='white')
        self.canvas.pack(side='top')

        circle_diameter = 50
        circle_radius = circle_diameter // 2
        center_x = 50
        top_y = 50
        middle_y = 125
        bottom_y = 200

        self.red_light = self.canvas.create_oval(
            center_x - circle_radius, top_y - circle_radius,
            center_x + circle_radius, top_y + circle_radius, fill='red')

        self.orange_light = self.canvas.create_oval(
            center_x - circle_radius, middle_y - circle_radius,
            center_x + circle_radius, middle_y + circle_radius, fill='grey')

        self.green_light = self.canvas.create_oval(
            center_x - circle_radius, bottom_y - circle_radius,
            center_x + circle_radius, bottom_y + circle_radius, fill='grey')

        self.countdown_text = self.canvas.create_text(center_x, bottom_y + 30, text="0s", fill="black")

    def update_lights(self, active, orange=False):
        if orange:
            self.canvas.itemconfig(self.red_light, fill='grey')
            self.canvas.itemconfig(self.green_light, fill='grey')
            self.canvas.itemconfig(self.orange_light, fill='orange')
        else:
            if active:
                self.canvas.itemconfig(self.red_light, fill='grey')
                self.canvas.itemconfig(self.orange_light, fill='grey')
                self.canvas.itemconfig(self.green_light, fill='green')
            else:
                self.canvas.itemconfig(self.red_light, fill='red')
                self.canvas.itemconfig(self.orange_light, fill='grey')
                self.canvas.itemconfig(self.green_light, fill='grey')

    def update_counter(self, time_left):
        self.canvas.itemconfig(self.countdown_text, text=f"{time_left}s")

class VehicleDetection:
    def __init__(self, frame):
        self.frame = frame

    def detect_vehicles(self):
        conf_threshold = 0.5
        detected_vehicles = {vehicle_type: 0 for vehicle_type in vehicleTypes.values()}

        # Predict on the current frame
        results = model(self.frame)

        # Extract detection information
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # Move tensor to CPU and convert to NumPy
                conf = box.conf[0].cpu().numpy()  # Move tensor to CPU and convert to NumPy
                cls = int(box.cls[0].cpu().numpy())  # Extract class value

                # Filter out detections with low confidence
                if conf > conf_threshold:
                    # Check if the class value corresponds to a valid vehicle type
                    if cls in vehicleTypes:
                        detected_vehicles[vehicleTypes[cls]] += 1  # Increment count for the detected vehicle type
                        # Draw bounding box on the frame
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{vehicleTypes[cls]}: {conf:.2f}"
                        cv2.putText(self.frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return detected_vehicles, self.frame

class Junction:
    def __init__(self, master, name, row, col, use_live_cam=False, live_cam_index=0, video_path=''):
        self.name = name
        self.use_live_cam = use_live_cam
        self.live_cam_index = live_cam_index
        self.video_path = video_path
        self.video_label = None
        self.traffic_light = TrafficLightGUI(master, row, col + 1)
        self.vehicle_count = 0
        self.green_time = 10
        self.red_time = 5
        self.orange_time = 3  # Orange light duration

        self.setup_ui(master, row, col)
        self.start_video_thread()

        # DDPG agent
        self.state_dim = 4  # Adjust as per the state representation
        self.action_dim = 1  # Single action (green light duration)
        self.max_action = 30  # Maximum green light duration
        self.agent = DDPG(self.state_dim, self.action_dim, self.max_action)

        self.last_state = None
        self.last_action = None

    def setup_ui(self, master, row, col):
        label_frame = Frame(master, width=600, height=30)
        label_frame.grid(row=row, column=col, sticky='nw')
        label = Label(label_frame, text=self.name, font=('Arial', 14))
        label.pack()

        video_frame = Frame(master, width=600, height=400, bg='black')
        video_frame.grid(row=row, column=col, padx=10, pady=(40, 10), sticky='n')
        self.video_label = Label(video_frame)
        self.video_label.pack(fill='both', expand=True)

        vehicle_count_frame = Frame(master, width=600, height=50)
        vehicle_count_frame.grid(row=row, column=col, pady=(450, 0))
        self.vehicle_count_label = Label(vehicle_count_frame, text=f"Vehicles counted : 0. Green Signal Timing assigned : {self.green_time}s", font=('Arial', 12))
        self.vehicle_count_label.pack()

    def update_video(self):
        if self.use_live_cam:
            cap = cv2.VideoCapture(self.live_cam_index)
        else:
            cap = cv2.VideoCapture(self.video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (600, 400))
                detector = VehicleDetection(frame)
                detected_vehicles, frame = detector.detect_vehicles()
                total_vehicles = sum(detected_vehicles.values())
                self.vehicle_count = total_vehicles

                self.vehicle_counts = detected_vehicles

                # State representation for DDPG
                state = np.array([detected_vehicles['car'], detected_vehicles['motorcycle'], detected_vehicles['truck'], detected_vehicles['bus']])

                # Select action using DDPG
                action = self.agent.select_action(state)
                self.green_time = int(action[0])  # Update Timing based on DDPG action

                if self.last_state is not None:
                    reward = -total_vehicles  # Example reward function (minimize vehicles)
                    done = False
                    self.agent.store_transition(self.last_state, self.last_action, reward, state, done)
                    self.agent.train()

                self.last_state = state
                self.last_action = action

                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                self.vehicle_count_label.config(text=f"Vehicles counted : {total_vehicles}. Green Signal Timing assigned : {self.green_time}s")
            else:
                # If the video ends or no live feed, use default green time
                self.green_time = 10
            time.sleep(0.03)
        cap.release()

    def start_video_thread(self):
        thread = Thread(target=self.update_video)
        thread.daemon = True
        thread.start()

class TrafficSystemApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Traffic Light System (YOLOv8 + DDPG)")
        self.master.geometry('1500x950')

        self.junctions = [
            Junction(master, 'Junction 1', 0, 0, use_live_cam=False, live_cam_index=0, video_path='videos/11.mp4'),
            Junction(master, 'Junction 2', 0, 2, use_live_cam=False, live_cam_index=1, video_path='videos/2_2.mp4'),
            Junction(master, 'Junction 3', 1, 0, use_live_cam=False, live_cam_index=2, video_path='videos/33.mp4'),
            Junction(master, 'Junction 4', 1, 2, use_live_cam=False, live_cam_index=3, video_path='videos/44.mp4')
        ]

        self.start_traffic_light_thread()

    def update_traffic_lights(self):
        active_junction = 0
        while True:
            for i, junction in enumerate(self.junctions):
                if i == active_junction:
                    junction.traffic_light.update_lights(True)
                    for time_left in range(junction.green_time, 0, -1):
                        junction.traffic_light.update_counter(time_left)
                        time.sleep(1)
                    junction.traffic_light.update_counter(0)

                    if junction.vehicle_count > 0:
                        junction.traffic_light.update_lights(False, orange=True)
                        for time_left in range(junction.orange_time, 0, -1):
                            junction.traffic_light.update_counter(time_left)
                            time.sleep(1)
                        junction.traffic_light.update_counter(0)

                    junction.traffic_light.update_lights(False)
            
            active_junction = (active_junction + 1) % 4
            time.sleep(1)  # Ensure smooth transition

    def start_traffic_light_thread(self):
        thread = Thread(target=self.update_traffic_lights)
        thread.daemon = True
        thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSystemApp(root)
    root.mainloop()
