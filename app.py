"""
Pikachu Volleyball - Human vs AI Desktop App

Play Pikachu Volleyball against a PPO-trained AI agent.
The AI uses an ONNX-exported neural network for real-time inference.

Controls:
  Arrow keys: Move (left/right), Jump (up)
  Space:      Power hit
  Combine arrows with Space for directional hits:
    Space + Up:         Lob (high arc)
    Space + Down:       Spike (sharp downward, best near net)
    Space + Left/Right: Horizontal power hit
    Space + Arrow + Arrow: Combined hits (e.g., toward net + up + power_hit)

  D:     Cycle difficulty (Easy / Medium / Hard / Impossible)
  S:     Switch sides (play as left or right Pikachu)
  R:     Restart game
  ESC:   Quit
"""

import os
import sys
import numpy as np
import pygame
import onnxruntime as ort

from pikazoo.env.pikazoo_env import raw_env

# ============================================================================
# AI PLAYER (ONNX inference with observation pipeline)
# ============================================================================


class AIPlayer:
    """ONNX-based AI player that replicates the training observation pipeline.

    During training, the agent saw observations through:
      raw_env -> SimplifyAction -> MirrorP2Observation -> NormalizeObservation -> FrameStack(2)
    This class replicates Mirror + NormalizeObservation + FrameStack manually,
    and maps ONNX output back to raw_env's 18-action space.

    When playing as player_2, raw observations are mirrored so the network
    always sees the world from the player_1 (left side) perspective.
    """

    DIFFICULTIES = {
        'Easy':       3.0,   # high temperature -> near-random
        'Medium':     1.5,
        'Hard':       0.5,
        'Impossible': 0.0,   # argmax, purely deterministic
    }
    DIFFICULTY_ORDER = ['Easy', 'Medium', 'Hard', 'Impossible']

    # SimplifyAction mapping: simplified_idx -> original_action_idx
    ACTION_MAP = {
        "player_1": (0, 1, 2, 3, 4, 6, 7, 10, 11, 12, 13, 14, 16),
        "player_2": (0, 1, 2, 4, 3, 7, 6, 10, 12, 11, 13, 15, 17),
    }

    # MirrorP2Observation constants (must match training wrapper exactly)
    GROUND_WIDTH = 432
    X_INDICES = [0, 13, 26, 28, 30]        # self_x, opp_x, ball_x, ball_prev_x, ball_prev_prev_x
    NEGATE_INDICES = [3, 16, 32]            # self_diving_dir, opp_diving_dir, ball_x_vel

    def __init__(self, onnx_path, agent_id="player_2", difficulty="Hard"):
        self.session = ort.InferenceSession(onnx_path)
        self.agent_id = agent_id
        self.difficulty = difficulty

        # Observation normalization bounds (from pika-zoo observation_space)
        self.obs_low = np.array([
            32, 108, -15, -1, -2, 0, 0, 0, 0, 0, 0, 0, 0,
            32, 108, -15, -1, -2, 0, 0, 0, 0, 0, 0, 0, 0,
            20, 0, 0, 0, 0, 0, -20, -124, 0
        ], dtype=np.float32)
        self.obs_high = np.array([
            400, 244, 16, 1, 3, 4, 4, 1, 1, 1, 1, 1, 1,
            400, 244, 16, 1, 3, 4, 4, 1, 1, 1, 1, 1, 1,
            432, 252, 432, 252, 432, 252, 20, 124, 1
        ], dtype=np.float32)
        self.obs_range = self.obs_high - self.obs_low
        self.obs_range[self.obs_range == 0] = 1.0  # prevent division by zero

        self.prev_obs = None

    def reset(self):
        self.prev_obs = None

    def normalize(self, obs):
        """Replicate NormalizeObservation wrapper: min-max to [0, 1]."""
        return np.clip((obs.astype(np.float32) - self.obs_low) / self.obs_range, 0.0, 1.0)

    def frame_stack(self, obs_normalized):
        """Replicate FrameStack(2): concatenate [prev_frame, current_frame]."""
        if self.prev_obs is None:
            self.prev_obs = obs_normalized.copy()
        stacked = np.concatenate([self.prev_obs, obs_normalized])
        self.prev_obs = obs_normalized.copy()
        return stacked

    def _mirror(self, obs):
        """Mirror x-coordinates for player_2 (matches MirrorP2Observation).

        Applied BEFORE normalization on raw integer observations.
        """
        mirrored = obs.copy()
        for idx in self.X_INDICES:
            mirrored[idx] = self.GROUND_WIDTH - obs[idx]
        for idx in self.NEGATE_INDICES:
            mirrored[idx] = -obs[idx]
        return mirrored

    def get_action(self, raw_obs):
        """Full pipeline: raw obs -> [mirror if P2] -> normalize -> frame_stack -> ONNX -> action."""
        processed = raw_obs.copy()
        if self.agent_id == "player_2":
            processed = self._mirror(processed)
        obs_norm = self.normalize(processed)
        obs_stacked = self.frame_stack(obs_norm)
        obs_input = obs_stacked.reshape(1, -1).astype(np.float32)

        logits = self.session.run(None, {'observation': obs_input})[0][0]

        temperature = self.DIFFICULTIES[self.difficulty]
        if temperature <= 0.01:
            simplified_action = int(np.argmax(logits))
        else:
            scaled = logits / temperature
            scaled -= scaled.max()  # numerical stability
            probs = np.exp(scaled) / np.exp(scaled).sum()
            simplified_action = int(np.random.choice(len(probs), p=probs))

        return self.ACTION_MAP[self.agent_id][simplified_action]

    def cycle_difficulty(self):
        idx = self.DIFFICULTY_ORDER.index(self.difficulty)
        self.difficulty = self.DIFFICULTY_ORDER[(idx + 1) % len(self.DIFFICULTY_ORDER)]
        return self.difficulty


# ============================================================================
# HUMAN INPUT
# ============================================================================


# Keyboard state [left, right, up, down, power_hit] -> action_key_map index
# Built from pikazoo's action_key_map:
# action_key_map[i] = [left, right, up, down, power_hit]
KEYS_TO_ACTION = {
    (0, 0, 0, 0, 0): 0,
    (0, 0, 0, 0, 1): 1,
    (0, 0, 1, 0, 0): 2,
    (0, 1, 0, 0, 0): 3,
    (1, 0, 0, 0, 0): 4,
    (0, 0, 0, 1, 0): 5,
    (0, 1, 1, 0, 0): 6,
    (1, 0, 1, 0, 0): 7,
    (0, 1, 0, 1, 0): 8,
    (1, 0, 0, 1, 0): 9,
    (0, 0, 1, 0, 1): 10,
    (0, 1, 0, 0, 1): 11,
    (1, 0, 0, 0, 1): 12,
    (0, 0, 0, 1, 1): 13,
    (0, 1, 1, 0, 1): 14,
    (1, 0, 1, 0, 1): 15,
    (0, 1, 0, 1, 1): 16,
    (1, 0, 0, 1, 1): 17,
}


def get_human_action():
    """Read keyboard state and return raw_env action index (0-17)."""
    keys = pygame.key.get_pressed()
    left = int(keys[pygame.K_LEFT])
    right = int(keys[pygame.K_RIGHT])
    up = int(keys[pygame.K_UP])
    down = int(keys[pygame.K_DOWN])
    power = int(keys[pygame.K_SPACE] or keys[pygame.K_z])

    # Cancel out conflicting directions
    if left and right:
        left = right = 0
    if up and down:
        up = down = 0

    key_tuple = (left, right, up, down, power)
    return KEYS_TO_ACTION.get(key_tuple, 0)


# ============================================================================
# MAIN APP
# ============================================================================


SCALE = 2           # render scale (432x304 -> 864x608)
FPS = 25            # game speed (original game runs at ~25 FPS)
WINNING_SCORE = 15


def resource_path(relative_path):
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative_path)


ONNX_MODEL = resource_path("models/pikachu_actor.onnx")


def ensure_pikazoo_assets():
    """Copy project sprites into pikazoo package (required for render_mode).

    Training uses raw_env() without rendering, so sprites are not loaded there.
    pikazoo always reads from its own env/img/ directory.
    """
    import pikazoo.env.pikazoo_env as pikazoo_env

    pkg_img = os.path.join(os.path.dirname(pikazoo_env.__file__), "img")
    project_img = resource_path("assets/img")
    marker = os.path.join(pkg_img, "ball_hyper.png")

    if os.path.isfile(marker):
        return

    if not os.path.isdir(project_img):
        print(f"Error: assets not found at {project_img}")
        sys.exit(1)

    os.makedirs(pkg_img, exist_ok=True)
    import shutil

    for name in os.listdir(project_img):
        if name.endswith(".png"):
            shutil.copy2(os.path.join(project_img, name), os.path.join(pkg_img, name))


def draw_hud(screen, scores, difficulty, human_side, font):
    """Draw scores and info overlay."""
    w = screen.get_width()

    # Scores
    p1_label = "YOU" if human_side == "player_1" else "AI"
    p2_label = "YOU" if human_side == "player_2" else "AI"
    score_text = f"{p1_label}: {scores[0]}   {p2_label}: {scores[1]}"
    surf = font.render(score_text, True, (255, 255, 255))
    rect = surf.get_rect(center=(w // 2, 16))
    # Background for readability
    bg = pygame.Surface((rect.width + 16, rect.height + 8))
    bg.set_alpha(160)
    bg.fill((0, 0, 0))
    screen.blit(bg, (rect.x - 8, rect.y - 4))
    screen.blit(surf, rect)

    # Difficulty + controls hint
    info = f"[D] Difficulty: {difficulty}  |  [S] Switch sides  |  [R] Restart  |  [ESC] Quit"
    info_surf = font.render(info, True, (200, 200, 200))
    info_rect = info_surf.get_rect(center=(w // 2, screen.get_height() - 14))
    bg2 = pygame.Surface((info_rect.width + 16, info_rect.height + 6))
    bg2.set_alpha(140)
    bg2.fill((0, 0, 0))
    screen.blit(bg2, (info_rect.x - 8, info_rect.y - 3))
    screen.blit(info_surf, info_rect)


def draw_game_over(screen, winner_text, font_big, font_small):
    """Draw game over overlay."""
    w, h = screen.get_size()

    overlay = pygame.Surface((w, h))
    overlay.set_alpha(150)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))

    text = font_big.render(winner_text, True, (255, 255, 50))
    rect = text.get_rect(center=(w // 2, h // 2 - 20))
    screen.blit(text, rect)

    restart = font_small.render("Press R to restart", True, (200, 200, 200))
    screen.blit(restart, restart.get_rect(center=(w // 2, h // 2 + 30)))


def main():
    if not os.path.exists(ONNX_MODEL):
        print(f"Error: {ONNX_MODEL} not found. Train the model first.")
        sys.exit(1)

    ensure_pikazoo_assets()
    pygame.init()
    screen_w, screen_h = 432 * SCALE, 304 * SCALE
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Pikachu Volleyball - Human vs AI")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("monospace", 14 * SCALE, bold=True)
    font_big = pygame.font.SysFont("monospace", 24 * SCALE, bold=True)
    font_small = pygame.font.SysFont("monospace", 12 * SCALE)

    human_side = "player_1"
    ai_side = "player_2"

    ai = AIPlayer(ONNX_MODEL, agent_id=ai_side, difficulty="Hard")

    env = raw_env(render_mode='rgb_array', winning_score=WINNING_SCORE)
    obs, _ = env.reset()
    ai.reset()

    game_over = False
    winner_text = ""

    running = True
    while running:
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_d:
                    new_diff = ai.cycle_difficulty()
                    pygame.display.set_caption(f"Pikachu Volleyball - {new_diff}")
                elif event.key == pygame.K_s:
                    human_side, ai_side = ai_side, human_side
                    ai.agent_id = ai_side
                    ai.reset()
                    obs, _ = env.reset()
                    game_over = False
                elif event.key == pygame.K_r:
                    ai.reset()
                    obs, _ = env.reset()
                    game_over = False

        if not game_over:
            # --- Get actions ---
            human_action = get_human_action()
            ai_action = ai.get_action(obs[ai_side])

            actions = {
                human_side: human_action,
                ai_side: ai_action,
            }

            obs, rewards, terms, truncs, infos = env.step(actions)

            if terms.get("player_1", False):
                game_over = True
                scores = infos.get("player_1", {}).get("scores",
                          [env.scores[0], env.scores[1]] if hasattr(env, 'scores') else [0, 0])
                if rewards[human_side] > 0:
                    winner_text = "YOU WIN!"
                else:
                    winner_text = "AI WINS!"

        # --- Render ---
        frame = env.render()  # (304, 432, 3) numpy array
        # Scale up: create pygame surface and blit scaled
        surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        scaled = pygame.transform.scale(surf, (screen_w, screen_h))
        screen.blit(scaled, (0, 0))

        # HUD overlay
        scores = [0, 0]
        if hasattr(env, 'scores'):
            scores = list(env.scores)
        draw_hud(screen, scores, ai.difficulty, human_side, font)

        if game_over:
            draw_game_over(screen, winner_text, font_big, font_small)

        pygame.display.flip()
        clock.tick(FPS)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
