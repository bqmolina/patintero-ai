import numpy as np
import pygame


class Environment:
    def __init__(self, num_attackers=5, num_defenders=5):
        self.width = 400
        self.height = 600
        self.fps = 15
        self.time_limit_seconds = 60
        self.max_frames = self.fps * self.time_limit_seconds
        self.num_attackers = int(num_attackers)
        self.num_defenders = int(num_defenders)
        self.crosswise_line_ys = np.linspace(self.height * 0.2, self.height * 0.8, 4).astype(np.float32)
        self.lengthwise_line_x = self.width / 2
        self.lengthwise_line_top = float(self.crosswise_line_ys[0])
        self.lengthwise_line_bottom = float(self.crosswise_line_ys[-1])
        self.starting_area_y_min = self.lengthwise_line_bottom
        self.return_area_y_max = self.lengthwise_line_top
        self.attacker_progress_reward_scale = 0.4
        self.attacker_return_area_entry_reward = 0.5
        self.attacker_spin_penalty_scale = 0.05
        self.defender_tracking_reward_scale = 0.2

        self.speed = 10
        self.done = False
        self.frame_count = 0
        self.attacker_score = 0
        self.defender_score = 0
        self.episode_number = 1
        self.last_step_info = {}
        self.attacker_reached_return_area = np.zeros(self.num_attackers, dtype=bool)
        self.attacker_crossed_lines = np.zeros((self.num_attackers, len(self.crosswise_line_ys)), dtype=bool)
        self.attacker_return_crossed_lines = np.zeros((self.num_attackers, len(self.crosswise_line_ys)), dtype=bool)

        self.attackers = [Attacker(np.array([self.width / 2, self.height - 50], dtype=np.float32)) for _ in range(self.num_attackers)]
        self.defenders = [
            Defender(
                np.array([self.width / 2 - 30, self.crosswise_line_ys[idx]], dtype=np.float32),
                orientation="crosswise",
                lane_position=float(self.crosswise_line_ys[idx]),
                lane_min=0.0,
                lane_max=float(self.width - 60.0),
            )
            for idx in range(min(4, self.num_defenders))
        ]
        if self.num_defenders > 4:
            self.defenders.append(
                Defender(
                    np.array([self.lengthwise_line_x - 5.0, self.lengthwise_line_top], dtype=np.float32),
                    orientation="lengthwise",
                    lane_position=float(self.lengthwise_line_x),
                    lane_min=float(self.lengthwise_line_top),
                    lane_max=float(self.lengthwise_line_bottom),
                )
            )
        self.attacker = self.attackers[0]
        self.defender = self.defenders[0]

        self.reset()

    def __check_collision(self):
        closest_pair = None
        closest_distance = None

        for attacker_idx, attacker in enumerate(self.attackers):
            for defender_idx, defender in enumerate(self.defenders):
                rect_left = float(defender.position[0])
                rect_top = float(defender.position[1])
                rect_right = rect_left + float(defender.width)
                rect_bottom = rect_top + float(defender.height)

                closest_x = np.clip(float(attacker.position[0]), rect_left, rect_right)
                closest_y = np.clip(float(attacker.position[1]), rect_top, rect_bottom)
                dx = float(attacker.position[0]) - float(closest_x)
                dy = float(attacker.position[1]) - float(closest_y)
                distance = np.sqrt(dx * dx + dy * dy)

                if distance <= float(attacker.radius):
                    if closest_distance is None or distance < closest_distance:
                        closest_distance = distance
                        closest_pair = (attacker_idx, defender_idx)

        return closest_pair

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

    def _spawn_attackers(self):
        attacker_radius = float(self.attackers[0].radius) if self.attackers else 10.0
        x_min = attacker_radius
        x_max = self.width - attacker_radius
        y_min = max(self.starting_area_y_min + (4 * attacker_radius), attacker_radius)
        y_max = max(y_min, self.height - (2 * attacker_radius))
        min_center_distance = 2.0 * attacker_radius
        max_attempts_per_attacker = 200

        positions = []
        for _ in range(self.num_attackers):
            placed = False

            for _ in range(max_attempts_per_attacker):
                candidate = np.array(
                    [
                        np.random.uniform(x_min, x_max),
                        np.random.uniform(y_min, y_max),
                    ],
                    dtype=np.float32,
                )
                if all(np.linalg.norm(candidate - existing) >= min_center_distance for existing in positions):
                    positions.append(candidate)
                    placed = True
                    break

            if placed:
                continue

            # Fallback: choose the candidate that maximizes distance from existing attackers.
            best_candidate = None
            best_score = -1.0
            for _ in range(300):
                candidate = np.array(
                    [
                        np.random.uniform(x_min, x_max),
                        np.random.uniform(y_min, y_max),
                    ],
                    dtype=np.float32,
                )
                if not positions:
                    best_candidate = candidate
                    break
                score = min(np.linalg.norm(candidate - existing) for existing in positions)
                if score > best_score:
                    best_score = score
                    best_candidate = candidate

            positions.append(best_candidate if best_candidate is not None else np.array([x_min, y_min], dtype=np.float32))

        return positions

    def _spawn_defenders(self):
        return [defender.random_spawn_position() for defender in self.defenders]

    @staticmethod
    def _crossed_line(prev_y, curr_y, line_y):
        return (prev_y - line_y) * (curr_y - line_y) < 0

    def _update_attacker_progress(self, prev_positions, curr_positions):
        illegal_recross = None

        for attacker_idx in range(self.num_attackers):
            prev_y = float(prev_positions[attacker_idx, 1])
            curr_y = float(curr_positions[attacker_idx, 1])
            was_return_phase = bool(self.attacker_reached_return_area[attacker_idx])

            for line_idx, line_y in enumerate(self.crosswise_line_ys):
                if not self._crossed_line(prev_y, curr_y, float(line_y)):
                    continue

                if was_return_phase:
                    if self.attacker_return_crossed_lines[attacker_idx, line_idx]:
                        illegal_recross = (attacker_idx, line_idx)
                        return illegal_recross

                    self.attacker_return_crossed_lines[attacker_idx, line_idx] = True
                    continue

                if self.attacker_crossed_lines[attacker_idx, line_idx]:
                    illegal_recross = (attacker_idx, line_idx)
                    return illegal_recross

                self.attacker_crossed_lines[attacker_idx, line_idx] = True

            # Promote to return phase only after validating crossings for this step.
            if (not was_return_phase) and curr_y <= self.return_area_y_max:
                self.attacker_reached_return_area[attacker_idx] = True

        return illegal_recross

    def reset(self):
        for attacker, spawn_position in zip(self.attackers, self._spawn_attackers()):
            attacker.reset(spawn_position)
        for defender, spawn_position in zip(self.defenders, self._spawn_defenders()):
            defender.reset(spawn_position)

        self.attacker = self.attackers[0]
        self.defender = self.defenders[0]
        self.done = False
        self.frame_count = 0
        self.attacker_reached_return_area.fill(False)
        self.attacker_crossed_lines.fill(False)
        self.attacker_return_crossed_lines.fill(False)

        return self.get_obs()

    def step(self, attacker_action, defender_action):
        if self.done:
            return self.get_obs(), (
                np.zeros(self.num_attackers, dtype=np.float32),
                np.zeros(self.num_defenders, dtype=np.float32),
            ), True

        attacker_actions = np.asarray(attacker_action, dtype=np.float32).reshape(-1)
        defender_actions = np.asarray(defender_action, dtype=np.float32).reshape(-1)

        if attacker_actions.size != self.num_attackers:
            raise ValueError(f"Expected {self.num_attackers} attacker actions, got {attacker_actions.size}")
        if defender_actions.size != self.num_defenders:
            raise ValueError(f"Expected {self.num_defenders} defender actions, got {defender_actions.size}")

        prev_attacker_positions = np.stack([attacker.position.copy() for attacker in self.attackers], axis=0)
        prev_attacker_forwards = np.stack([attacker.forward.copy() for attacker in self.attackers], axis=0)
        prev_reached_return_area = self.attacker_reached_return_area.copy()
        prev_defender_centers = np.stack(
            [defender.position + np.array([defender.width / 2, defender.height / 2], dtype=np.float32) for defender in self.defenders],
            axis=0,
        )

        for attacker, action in zip(self.attackers, attacker_actions):
            attacker.move(float(action))
        for defender, action in zip(self.defenders, defender_actions):
            defender.move(int(action))

        self.frame_count += 1

        curr_attacker_positions = np.stack([attacker.position.copy() for attacker in self.attackers], axis=0)
        illegal_recross = self._update_attacker_progress(prev_attacker_positions, curr_attacker_positions)
        curr_defender_centers = np.stack(
            [defender.position + np.array([defender.width / 2, defender.height / 2], dtype=np.float32) for defender in self.defenders],
            axis=0,
        )

        attacker_rewards = np.full(self.num_attackers, -0.01, dtype=np.float32)
        defender_rewards = np.full(self.num_defenders, -0.01, dtype=np.float32)
        terminal_reason = None
        winning_attacker = None
        winning_defender = None

        for attacker_idx in range(self.num_attackers):
            if self.attacker_reached_return_area[attacker_idx]:
                target_y = float(self.starting_area_y_min)
            else:
                target_y = float(self.return_area_y_max)

            prev_target_distance = abs(float(prev_attacker_positions[attacker_idx, 1]) - target_y) / self.height
            curr_target_distance = abs(float(curr_attacker_positions[attacker_idx, 1]) - target_y) / self.height
            attacker_rewards[attacker_idx] += self.attacker_progress_reward_scale * (
                prev_target_distance - curr_target_distance
            )

        newly_reached_return_area = np.logical_and(~prev_reached_return_area, self.attacker_reached_return_area)
        if np.any(newly_reached_return_area):
            attacker_rewards[newly_reached_return_area] += self.attacker_return_area_entry_reward

        for attacker_idx in range(self.num_attackers):
            prev_forward = prev_attacker_forwards[attacker_idx]
            curr_forward = self.attackers[attacker_idx].forward
            cosine = float(np.clip(np.dot(prev_forward, curr_forward), -1.0, 1.0))
            turn_angle = float(np.degrees(np.arccos(cosine)))
            attacker_rewards[attacker_idx] -= self.attacker_spin_penalty_scale * (turn_angle / 180.0)

        for defender_idx in range(self.num_defenders):
            prev_tracking_distance = np.min(np.linalg.norm(prev_attacker_positions - prev_defender_centers[defender_idx], axis=1))
            curr_tracking_distance = np.min(np.linalg.norm(curr_attacker_positions - curr_defender_centers[defender_idx], axis=1))
            defender_rewards[defender_idx] += self.defender_tracking_reward_scale * (
                (prev_tracking_distance - curr_tracking_distance) / self.width
            )

        collision_pair = self.__check_collision()
        if collision_pair is not None:
            winning_attacker, winning_defender = collision_pair
            attacker_rewards -= 0.5
            defender_rewards += 0.5
            attacker_rewards[winning_attacker] -= 0.5
            defender_rewards[winning_defender] += 0.5
            self.defender_score += 1
            self.done = True
            terminal_reason = "tag"
            print("Defender gains a point")
        elif illegal_recross is not None:
            winning_attacker = int(illegal_recross[0])
            attacker_rewards -= 1.0
            defender_rewards += 1.0
            self.defender_score += 1
            self.done = True
            terminal_reason = "invalid_recross"
            print("Defender gains a point (illegal attacker recross before return)")
        else:
            crossing_attackers = [
                idx
                for idx, attacker in enumerate(self.attackers)
                if self.attacker_reached_return_area[idx] and attacker.position[1] >= self.starting_area_y_min
            ]
            if crossing_attackers:
                winning_attacker = crossing_attackers[0]
                attacker_rewards += 1.0
                defender_rewards -= 1.0
                attacker_rewards[winning_attacker] += 1.0
                self.attacker_score += 1
                self.done = True
                terminal_reason = "return"
                print("Attacker gains a point (successful return)")

        if not self.done and self.frame_count >= self.max_frames:
            attacker_rewards -= 1
            defender_rewards += 1
            self.defender_score += 1
            self.done = True
            terminal_reason = "timeout"
            print("Defender gains a point (timeout)")

        terminal = self.done
        self.last_step_info = {
            "frame_count": int(self.frame_count),
            "attacker_positions": curr_attacker_positions.tolist(),
            "defender_positions": np.stack([defender.position.copy() for defender in self.defenders], axis=0).tolist(),
            "attacker_rewards": attacker_rewards.tolist(),
            "defender_rewards": defender_rewards.tolist(),
            "done": bool(terminal),
            "terminal_reason": terminal_reason,
            "winning_attacker": None if winning_attacker is None else int(winning_attacker),
            "winning_defender": None if winning_defender is None else int(winning_defender),
            "attacker_reached_return_area": self.attacker_reached_return_area.tolist(),
        }

        if terminal:
            self.episode_number += 1
            return self.get_obs(), (attacker_rewards, defender_rewards), True

        return self.get_obs(), (attacker_rewards, defender_rewards), False

    def get_obs(self):
        attackers = np.array(
            [
                [
                    attacker.position[0] / self.width,
                    attacker.position[1] / self.height,
                    attacker.forward[0],
                    attacker.forward[1],
                ]
                for attacker in self.attackers
            ],
            dtype=np.float32,
        )
        defenders = np.array(
            [
                [
                    defender.position[0] / self.width,
                    defender.position[1] / self.height,
                    0.0 if defender.orientation == "crosswise" else 1.0,
                    defender.lane_position / (self.height if defender.orientation == "crosswise" else self.width),
                ]
                for defender in self.defenders
            ],
            dtype=np.float32,
        )
        board = np.array([self.lengthwise_line_x / self.width, *list(self.crosswise_line_ys / self.height)], dtype=np.float32)
        return {
            "attackers": attackers,
            "defenders": defenders,
            "board": board,
            "state": self._build_state(),
        }

    def _build_state(self):
        attacker_state = []
        for attacker in self.attackers:
            attacker_state.extend(
                [
                    attacker.position[0] / self.width,
                    attacker.position[1] / self.height,
                    attacker.forward[0],
                    attacker.forward[1],
                ]
            )

        defender_state = []
        for defender in self.defenders:
            defender_state.extend(
                [
                    defender.position[0] / self.width,
                    defender.position[1] / self.height,
                    0.0 if defender.orientation == "crosswise" else 1.0,
                    defender.lane_position / (self.height if defender.orientation == "crosswise" else self.width),
                ]
            )

        board_state = [self.lengthwise_line_x / self.width, *list(self.crosswise_line_ys / self.height)]
        scoreboard_state = [self.attacker_score / 10.0, self.defender_score / 10.0, self.frame_count / max(self.max_frames, 1)]
        return np.array(attacker_state + defender_state + board_state + scoreboard_state, dtype=np.float32)


class Renderer:
    def __init__(self, env, fps=15):
        pygame.init()
        self.screen = pygame.display.set_mode((400, 600))
        pygame.display.set_caption("Patintero AI 5v5")
        self.clock = pygame.time.Clock()
        self.fps = float(fps)
        self.env = env
        self.font = pygame.font.SysFont("consolas", 20)
        self.small_font = pygame.font.SysFont("consolas", 16)
        self.running = True

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
        self.screen.fill("black")

        line_color = (45, 45, 45)
        for y in self.env.crosswise_line_ys:
            pygame.draw.line(self.screen, line_color, (20, int(y)), (self.env.width - 20, int(y)), 2)
        pygame.draw.line(
            self.screen,
            line_color,
            (int(self.env.lengthwise_line_x), int(self.env.lengthwise_line_top)),
            (int(self.env.lengthwise_line_x), int(self.env.lengthwise_line_bottom)),
            2,
        )

        hud_title = self.font.render(f"Episode: {self.env.episode_number}", True, (240, 240, 240))
        attacker_text = self.small_font.render(f"Attacker: {self.env.attacker_score}", True, (80, 160, 255))
        defender_text = self.small_font.render(f"Defender: {self.env.defender_score}", True, (255, 110, 110))
        self.screen.blit(hud_title, (10, 10))
        self.screen.blit(attacker_text, (10, 36))
        self.screen.blit(defender_text, (10, 58))

        for attacker in self.env.attackers:
            attacker.render(self.screen)
        for defender in self.env.defenders:
            defender.render(self.screen)

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
            self.position = new_position
        else:
            self.stay()

    def stay(self):
        pass

    def reset(self, position=None):
        if position is None:
            random_x = np.random.uniform(self.radius, self.screen_width - self.radius)
            self.position = np.array((random_x, self.screen_height - 50), np.float32)
        else:
            self.position = np.array(position, np.float32)
        self.forward = np.array([0, -1], np.float32)

    def render(self, screen):
        pygame.draw.circle(screen, "blue", self.position.tolist(), self.radius)
        pygame.draw.line(
            screen,
            "cyan",
            self.position.tolist(),
            (self.position + self.forward * 20).tolist(),
            2,
        )


class Defender:
    def __init__(self, position, orientation="crosswise", lane_position=0.0, lane_min=0.0, lane_max=0.0):
        self.position = position
        self.orientation = orientation
        self.lane_position = float(lane_position)
        self.lane_min = float(lane_min)
        self.lane_max = float(lane_max)
        self.width = 60 if orientation == "crosswise" else 10
        self.height = 10 if orientation == "crosswise" else 60
        self.speed = 10

        self.screen_width = 400
        self.screen_height = 600
        self._snap_to_lane()

    def __getitem__(self, key):
        return self.position[key]

    def _snap_to_lane(self):
        if self.orientation == "crosswise":
            self.position[1] = self.lane_position - self.height / 2
        else:
            self.position[0] = self.lane_position - self.width / 2

    def move(self, direction):
        if direction == 1 or direction == -1:
            if self.orientation == "crosswise":
                new_x = max(0, min(self.position[0] + direction * self.speed, self.screen_width - self.width))
                self.position[0] = new_x
                self._snap_to_lane()
            else:
                new_y = max(self.lane_min, min(self.position[1] + direction * self.speed, self.lane_max - self.height))
                self.position[1] = new_y
                self._snap_to_lane()
        else:
            self.stay()

    def stay(self):
        pass

    def reset(self, position=None):
        if position is None:
            if self.orientation == "crosswise":
                random_x = np.random.uniform(0, self.screen_width - self.width)
                self.position = np.array([random_x, self.lane_position - self.height / 2], np.float32)
            else:
                random_y = np.random.uniform(self.lane_min, self.lane_max - self.height)
                self.position = np.array([self.lane_position - self.width / 2, random_y], np.float32)
        else:
            self.position = np.array(position, np.float32)
        self._snap_to_lane()

    def random_spawn_position(self):
        if self.orientation == "crosswise":
            random_x = np.random.uniform(0, self.screen_width - self.width)
            return np.array([random_x, self.lane_position - self.height / 2], np.float32)
        random_y = np.random.uniform(self.lane_min, self.lane_max - self.height)
        return np.array([self.lane_position - self.width / 2, random_y], np.float32)

    def render(self, screen):
        pygame.draw.rect(
            screen,
            "red",
            (int(self.position[0]), int(self.position[1]), int(self.width), int(self.height)),
        )
