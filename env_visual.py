import pygame

class Game:
    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode((400, 600))
        pygame.display.set_caption("Patintero AI")

        self.running = True

        self.clock = pygame.time.Clock()
        self.fps = 30

        self.attacker = Attacker([self.screen.get_width() / 2, self.screen.get_height() - 50], self.screen)
        self.defender = Defender([self.screen.get_width() / 2 - 25, self.screen.get_height()], self.screen)

        self.attacker_score = 0
        self.defender_score = 0
        self.episode = 1

    def run(self):
        self.round_start_time = pygame.time.get_ticks()
        while self.running:
            if pygame.time.get_ticks() - self.round_start_time > 10000:
                self.reset()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            self.screen.fill("black")

            # Game logic
            keys = pygame.key.get_pressed()
            if keys[pygame.K_a]:
                self.defender.move(-1)
            elif keys[pygame.K_d]:
                self.defender.move(1)
            if keys[pygame.K_LEFT]:
                self.attacker.move(-10)
            elif keys[pygame.K_RIGHT]:
                self.attacker.move(10)
            elif keys[pygame.K_UP]:
                self.attacker.move(0)

            if self.__check_collision():
                print("Collision detected!")
                self.reset()
                self.defender_score += 1

            if self.__check_passing():
                print("Attacker passed through!")
                self.reset()
                self.attacker_score += 1

            self.__render()

        pygame.quit()

    def reset(self):
        self.episode += 1
        self.attacker.reset()
        self.defender.reset()
        self.round_start_time = pygame.time.get_ticks()

    def __check_collision(self):
        attacker_rect = pygame.Rect(self.attacker.position[0] - self.attacker.radius, 
                                    self.attacker.position[1] - self.attacker.radius, 
                                    self.attacker.radius * 2, self.attacker.radius * 2)
        defender_rect = pygame.Rect(self.defender.position[0], self.defender.screen_height / 2, 
                                    self.defender.width, self.defender.height)

        return attacker_rect.colliderect(defender_rect)
    
    def __check_passing(self):
        return self.attacker.position[1] < self.defender.screen_height / 2
    
    def __font(self):
        font = pygame.font.Font(None, 36)
        episode_text = font.render(f"Episode: {self.episode}", True, "white")
        attacker_score_text = font.render(f"Attacker: {self.attacker_score}", True, "white")
        defender_score_text = font.render(f"Defender: {self.defender_score}", True, "white")
        time_text = font.render(f"Time: {(10000 - (pygame.time.get_ticks() - self.round_start_time)) // 1000 + 1}", True, "white")
        self.screen.blit(attacker_score_text, (10, 10))
        self.screen.blit(defender_score_text, (10, 50))
        self.screen.blit(time_text, (10, 90))
        self.screen.blit(episode_text, (10, 130))
    
    def __render(self):
            self.attacker.render(self.screen)
            self.defender.render(self.screen)
            self.__font()

            pygame.display.flip()
            self.clock.tick(self.fps)

class Attacker:
    def __init__(self, position, screen):
        # Action Set: {move(x) where x is (-180, 180], stay()}
        # Local Space Movement
        self.screen = screen
        self.screen_width = screen.get_width()
        self.screen_height = screen.get_height()

        self.radius = 10
        self.position = position
        self.direction = pygame.math.Vector2(0, -1)
        self.angle = -90

    def move(self, angle):
        self.angle += angle
        direction = pygame.math.Vector2(1, 0).rotate(self.angle)
        
        new_x = self.position[0] + direction[0] * 10
        new_y = self.position[1] + direction[1] * 10
        
        new_x = max(self.radius, min(new_x, self.screen_width - self.radius))
        new_y = max(self.radius, min(new_y, self.screen_height - self.radius))
        
        if new_x != self.position[0] + direction[0] * 10 or new_y != self.position[1] + direction[1] * 10:
            print("Attacker is touching the boundary!")
        
        self.position = [new_x, new_y]
    
    def stay(self):
        pass
    
    def render(self, screen):
        pygame.draw.circle(screen, "blue", self.position, self.radius)
        end_x = self.position[0] + 20 * pygame.math.Vector2(1, 0).rotate(self.angle)[0]
        end_y = self.position[1] + 20 * pygame.math.Vector2(1, 0).rotate(self.angle)[1]
        pygame.draw.line(screen, "cyan", self.position, (end_x, end_y), 2)

    def reset(self):
        self.position = [self.screen_width / 2, self.screen_height - 50]
        self.direction = pygame.math.Vector2(0, -1)
        self.angle = -90

class Defender:
    def __init__(self, position, screen):
        # Action Set: {move(-1), move(1), stay()}
        self.screen = screen
        self.screen_width = screen.get_width()
        self.screen_height = screen.get_height()

        self.position = position
        self.width = 50
        self.height = 10

    def move(self, direction):
        if direction == 1 or direction == -1:
            new_x = max(0, min(self.position[0] + direction * 10, self.screen_width - self.width))
            if new_x != self.position[0] + direction * 10:
                print("Defender is touching the boundary!")
            self.position[0] = new_x

    def stay(self):
        pass

    def render(self, screen):
        pygame.draw.rect(screen, "red", (self.position[0], self.screen_height / 2, self.width, self.height))

    def reset(self):
        self.position = [self.screen_width / 2 - self.width / 2, self.screen_height]