import numpy as np
import pygame

class Environment:
    def __init__(self):
        self.width = 400
        self.height = 600
        self.fps = 15
        self.time_limit_seconds = 10
        self.max_frames = self.fps * self.time_limit_seconds
        self.attacker_progress_reward_scale = 0.2
        self.defender_tracking_reward_scale = 0.2
        
        self.speed = 10
        self.done = False
        self.frame_count = 0
        self.attacker_score = 0
        self.defender_score = 0
        self.episode_number = 1
        self.last_step_info = {}

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
        self.frame_count = 0

        return self.get_obs()

    def step(self, attacker_action, defender_action):
        if self.done:
            return self.get_obs(), (None, None), True

        prev_attacker_x = float(self.attacker[0])
        prev_attacker_y = float(self.attacker[1])
        prev_defender_center_x = float(self.defender[0] + self.defender.width / 2)
        
        self.attacker.move(attacker_action)
        self.defender.move(defender_action)
        self.frame_count += 1

        curr_attacker_x = float(self.attacker[0])
        curr_attacker_y = float(self.attacker[1])
        curr_defender_center_x = float(self.defender[0] + self.defender.width / 2)

        attacker_reward = -0.01
        defender_reward = -0.01
        terminal_reason = None

        # Heuristic 1 (attacker): reward progress toward the midline.
        midline_y = self.height / 2
        prev_midline_distance = max(prev_attacker_y - midline_y, 0.0) / self.height
        curr_midline_distance = max(curr_attacker_y - midline_y, 0.0) / self.height
        attacker_reward += self.attacker_progress_reward_scale * (prev_midline_distance - curr_midline_distance)

        # Heuristic 2 (defender): reward reducing horizontal distance to attacker.
        prev_tracking_distance = abs(prev_attacker_x - prev_defender_center_x) / self.width
        curr_tracking_distance = abs(curr_attacker_x - curr_defender_center_x) / self.width
        defender_reward += self.defender_tracking_reward_scale * (prev_tracking_distance - curr_tracking_distance)

        if self.__check_collision(): # Defender tags attacker
            attacker_reward -= 1
            defender_reward += 1
            self.defender_score += 1
            self.done = True
            terminal_reason = "tag"
            print("Defender gains a point")

        elif self.attacker[1] < self.height / 2: # Attacker passes through border
            attacker_reward += 1
            defender_reward -= 1
            self.attacker_score += 1
            self.done = True
            terminal_reason = "cross"
            print("Attacker gains a point")

        elif self.frame_count >= self.max_frames: # Defender wins on timeout
            attacker_reward -= 1
            defender_reward += 1
            self.defender_score += 1
            self.done = True
            terminal_reason = "timeout"
            print("Defender gains a point (timeout)")

        terminal = self.done
        self.last_step_info = {
            "frame_count": int(self.frame_count),
            "attacker_x": float(curr_attacker_x),
            "attacker_y": float(curr_attacker_y),
            "defender_x": float(self.defender[0]),
            "defender_y": float(self.defender[1]),
            "attacker_reward": float(attacker_reward),
            "defender_reward": float(defender_reward),
            "done": bool(terminal),
            "terminal_reason": terminal_reason,
        }

        if terminal:
            self.episode_number += 1
            # Immediately reset after a point so the next frame starts a fresh round.
            next_obs = self.reset()
            return next_obs, (attacker_reward, defender_reward), True

        return self.get_obs(), (attacker_reward, defender_reward), False

    def get_obs(self):
        attacker_obs = np.array([ # normalized x, y, and distance between attacker and defender
            self.attacker[0] / self.width,
            self.attacker[1] / self.height,
            (self.defender[0] - self.attacker[0]) / self.width,
            (self.defender[1] - self.attacker[1]) / self.height,
        ])

        defender_obs = np.array([ # normalized x and distance between attacker and defender
            self.defender[0] / self.width,
            (self.attacker[0] - self.defender[0]) / self.width,
            (self.attacker[1] - self.defender[1]) / self.height,
        ])

        return {
            "attacker": attacker_obs.astype(np.float32),
            "defender": defender_obs.astype(np.float32)
        }
    
class Renderer:
    def __init__(self, env, fps=15):
        pygame.init()
        self.screen = pygame.display.set_mode((400, 600))
        pygame.display.set_caption("Patintero AI")
        self.clock = pygame.time.Clock()
        self.fps = float(fps)
        self.env = env 
        self.font = pygame.font.SysFont("consolas", 20)
        self.small_font = pygame.font.SysFont("consolas", 16)

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
        self.screen.fill("black")

        hud_title = self.font.render(f"Episode: {self.env.episode_number}", True, (240, 240, 240))
        attacker_text = self.small_font.render(f"Attacker: {self.env.attacker_score}", True, (80, 160, 255))
        defender_text = self.small_font.render(f"Defender: {self.env.defender_score}", True, (255, 110, 110))
        self.screen.blit(hud_title, (10, 10))
        self.screen.blit(attacker_text, (10, 36))
        self.screen.blit(defender_text, (10, 58))

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

            # if new_position[0] != self.position[0] + (self.forward[0] * self.speed) or new_position[1] != self.position[1] + (self.forward[1] * self.speed):
            #     print("Attacker is touching the boundary.")

            self.position = new_position
        
        else:
            self.stay()

    def stay(self):
        pass

    def reset(self):
        random_x = np.random.uniform(self.radius, self.screen_width - self.radius)
        self.position = np.array((random_x, self.screen_height - 50), np.float32)
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
            # if new_x != self.position[0] + direction * self.speed:
            #     print("Defender is touching the boundary!")
            self.position[0] = new_x

        else:
            self.stay()

    def stay(self):
        pass

    def reset(self):
        random_x = np.random.uniform(0, self.screen_width - self.width)
        self.position = np.array([random_x, self.screen_height / 2], np.float32)
    
    def render(self, screen):
        pygame.draw.rect(
            screen, "red", (int(self.position[0]), int(self.position[1]), int(self.width), int(self.height))
        )