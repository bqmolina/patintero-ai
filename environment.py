import numpy as np
import pygame

class Environment:
    def __init__(self):
        self.width = 400
        self.height = 600
        
        self.speed = 10
        self.done = False

        self.attacker = Attacker(np.array([self.width / 2, self.height - 50], np.float32))
        self.defender = Defender(np.array([self.width / 2 - 25, self.height / 2], np.float32))

        self.reset()

    def __check_collision(self):
        distance = np.linalg.norm(self.attacker.position - (self.defender.position + np.array([self.defender.width / 2, 0])))
        return distance < (self.attacker.radius + self.defender.width / 2)
    
    def __debug_controls(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            self.defender.move(-1)
        elif keys[pygame.K_d]:
            self.defender.move(1)
        if keys[pygame.K_LEFT]:
            self.attacker.move(350)
        elif keys[pygame.K_RIGHT]:
            self.attacker.move(10)
        elif keys[pygame.K_UP]:
            self.attacker.move(0)

    def reset(self):
        self.attacker.reset()
        self.defender.reset()

        self.done = False

        return self.get_obs()

    def step(self, attacker_action, defender_action):
        if self.done:
            return self.get_obs(), (None, None), True
        
        # self.attacker.move(attacker_action)
        # self.defender.move(defender_action)
        self.__debug_controls()

        attacker_reward = -0.01
        defender_reward = -0.01

        if self.__check_collision():
            attacker_reward -= 1
            defender_reward += 1
            self.done = True
            print("Defender gains a point")

        elif self.attacker[1] < self.height / 2:
            attacker_reward += 1
            defender_reward -= 1
            self.done = True
            print("Attacker gains a point")

        return self.get_obs(), (attacker_reward, defender_reward), self.done

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
            "attacker": attacker_obs.astype(np.float32),
            "defender": defender_obs.astype(np.float32)
        }
    
class Renderer:
    def __init__(self, env):
        pygame.init()
        self.screen = pygame.display.set_mode((400, 600))
        pygame.display.set_caption("Patintero AI")
        self.clock = pygame.time.Clock()
        self.fps = 30
        self.env = env 

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
        self.screen.fill("black")

        self.env.attacker.render(self.screen)
        self.env.defender.render(self.screen)

        pygame.display.flip()
        self.clock.tick(self.fps)

    def quit(self):
        pygame.quit()

class Attacker:
    def __init__(self, position):
        self.position = position
        self.radius = 10
        self.speed = 10
        self.forward = np.array([0, -1], np.float32)

        self.screen_width = 400
        self.screen_height = 600

    def __getitem__(self, key):
        return self.position[key]

    def move(self, angle):
        if angle >= 0 and angle < 360:
            theta = np.radians(angle)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            self.forward = R @ self.forward

            new_position = self.position + (self.forward * self.speed)

            new_position[0] = max(self.radius, min(new_position[0], self.screen_width - self.radius))
            new_position[1] = max(self.radius, min(new_position[1], self.screen_height - self.radius))

            if new_position[0] != self.position[0] + (self.forward[0] * self.speed) or new_position[1] != self.position[1] + (self.forward[1] * self.speed):
                print("Attacker is touching the boundary.")

            self.position = new_position
        
        else:
            self.stay()

    def stay(self):
        pass

    def reset(self):
        self.position = np.array((self.screen_width / 2, self.screen_height - 50), np.float32)
        self.forward = np.array([0, -1], np.float32)

    def render(self, screen):
        pygame.draw.circle(
            screen, "blue", self.position.tolist(), self.radius
        )

        pygame.draw.line(
            screen, "cyan", self.position.tolist(), 
            (self.position + self.forward * 20).tolist(), 2
        )

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
        self.position = np.array([self.screen_width / 2 - self.width / 2, self.screen_height / 2], np.float32)
    
    def render(self, screen):
        pygame.draw.rect(
            screen, "red", (int(self.position[0]), int(self.position[1]), int(self.width), int(self.height))
        )