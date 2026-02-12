# TODO: Fix collisions, connect to pygame for visualization

import numpy as np

class Environment:
    def __init__(self):
        self.width = 400
        self.height = 600
        
        self.speed = 10
        self.done = False

        self.attacker = Attacker(np.array([self.width / 2, self.height - 50], np.float16))
        self.defender = Defender(np.array([self.width / 2 - 25, self.height], np.float16))

        self.reset()

    def reset(self):
        self.attacker.reset()
        self.defender.reset()

        self.done = False

        return self.get_obs()

    def step(self, attacker_action, defender_action):
        # attacker_action: {(-180, 180), None}
        # defender_action: {-1, 1, None}
        if self.done:
            return self.get_obs(), (None, None), True
        
        self.attacker.move(attacker_action)
        self.defender.move(defender_action)

        attacker_reward = -0.01
        defender_reward = -0.01

        distance = np.linalg.norm(self.attacker.position - self.defender.position)

        if distance < 25: # Fix
            defender_reward += 1
            attacker_reward -= 1
            self.done = True

        elif self.attacker[1] < self.height / 2:
            attacker_reward += 1
            defender_reward -= 1
            self.done = True

        return self.get_obs, (attacker_reward, defender_reward), self.done

    def get_obs(self):
        attacker_obs = np.array([
            self.attacker[0] / self.width,
            self.attacker[1] / self.height,
            (self.defender[0] - self.attacker[0]) / self.width,
            (self.defender[1] - self.attacker[1]) / self.height,
        ])

        defender_obs = np.array([
            self.defender[0] / self.width,
            (self.attacker[0] - self.defender[0]) / self.width,
            (self.attacker[1] - self.defender[1]) / self.height,
        ])

        return {
            "attacker": attacker_obs.astype(np.float16),
            "defender": defender_obs.astype(np.float16)
        }

class Attacker:
    def __init__(self, position):
        self.position = np.array((self.screen_width / 2, self.screen_height - 50), np.float16)
        self.angle = -90
        self.radius = 10
        self.speed = 10

        self.screen_width = 400
        self.screen_height = 600
        self.uv = np.array([1, 0], np.float16)

    def __getitem__(self, key):
        return self.position[key]

    def move(self, angle):
        if angle > -180 and angle <= 180:
            self.angle += angle
            if self.angle > 180:
                self.angle -= 360
            elif self.angle <= 180:
                self.angle += 360

            theta = np.radians(self.angle)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            distance = R @ self.uv

            new_position = self.position + (distance * self.speed)

            new_position[0] = max(self.radius, min(new_position[0], self.screen_width - self.radius))
            new_position[1] = max(self.radius, min(new_position[1], self.screen_height - self.radius))

            if new_position[0] != self.position[0] + (distance[0] * self.speed) or new_position[0] != self.position[0] + (distance[0] * self.speed):
                print("Attacker is touching the boundary.")

            self.position = new_position
        
        else:
            self.stay()

    def stay(self):
        pass

    def reset(self):
        self.position = np.array((self.screen_width / 2, self.screen_height - 50), np.float16)
        self.angle = -90

class Defender:
    def __init__(self, position):
        self.position = position
        self.width = 50
        self.height = 10
        self.speed = 10

        self.screen_width = 400
        self.screen_height = 600

    def __getitem__(self, key):
        return self.position[key]

    def move(self, direction):
        if direction == 1 or direction == -1:
            new_x = max(0, min(self.position[0] + direction * self.speed, self.screen_width - self.width))
            if new_x != self.position[0] + direction * self.speed:
                print("Defender is touching the boundary!")
            self.position[0] = new_x

        else:
            self.stay()

    def stay(self):
        pass

    def reset(self):
        self.position = np.array([self.screen_width / 2 - self.width / 2, self.screen_height], np.float16)