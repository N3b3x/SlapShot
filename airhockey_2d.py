
import pygame
import sys
import numpy as np

# initialize

pygame.init()
WIDTH, HEIGHT = 800, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Air Hockey")
clock = pygame.time.Clock()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PUCK_COLOR = (255, 50, 50)
PADDLE1_COLOR = (50, 150, 255)
TARGET_COLOR = (0, 255, 0)


# set up entities
puck_radius = 15
paddle_radius= 20

puck_pos = np.array([WIDTH / 2, HEIGHT / 2], dtype=float)
puck_vel = np.array([4, 2], dtype=float)

paddle1_pos = np.array([WIDTH - 100, HEIGHT / 2], dtype=float)
target_pos = np.array([WIDTH / 2, HEIGHT / 2], dtype=float)


paddle1_max_x = WIDTH // 3


def handle_wall_collision(puck_pos, puck_vel, puck_radius):
    if puck_pos[0] - puck_radius <= 0 or puck_pos[0] + puck_radius >= WIDTH:
        puck_vel[0] *= -1
    if puck_pos[1] - puck_radius <=0 or puck_pos[1] + puck_radius >= HEIGHT:
        puck_vel[1] *= -1

def move_paddle_towards_target(paddle_pos, target_pos, speed=5):

    d_pos = target_pos - paddle_pos
    dist = np.linalg.norm(d_pos)

    if dist > 1:
        paddle_pos += speed * d_pos / dist

def handle_paddle_collision(puck_pos, puck_vel, paddle_pos):
    delta = puck_pos - paddle_pos
    dist = np.linalg.norm(delta)
    min_dist = puck_radius + paddle_radius

    if dist < min_dist:
        normal = delta / dist
        vel_dot = np.dot(puck_vel, normal)
        puck_vel -= 2 * vel_dot * normal
        overlap = min_dist - dist
        # push puck outside of overlap
        puck_pos += normal * overlap


defense_line1 = WIDTH // 4

def predict_intercept(puck_pos, puck_vel, table_width, table_height, radius, intercept_x):
    pos = np.copy(puck_pos)
    vel = np.copy(puck_vel)

    while True:
        pos += vel
        if pos[1] - radius <=0 or pos[1] + radius >= table_height:
            vel[1] *= -1

        if ((vel[0] > 0 and pos[0] >= intercept_x) 
            or (vel[0] < 0 and pos[0] <= intercept_x)):

            return np.array([intercept_x, pos[1]])


while True:
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    #update puck position

    puck_pos += puck_vel

    handle_wall_collision(puck_pos, puck_vel, puck_radius)

    target_pos = predict_intercept(puck_pos, puck_vel, WIDTH, HEIGHT, puck_radius, defense_line1)

    move_paddle_towards_target(paddle1_pos, target_pos)
    paddle1_pos[0] = min(paddle1_pos[0], paddle1_max_x)

    handle_paddle_collision(puck_pos, puck_vel, paddle1_pos)

    pygame.draw.circle(screen, PUCK_COLOR, (int(puck_pos[0]), int(puck_pos[1])), puck_radius)
    pygame.draw.circle(screen, PADDLE1_COLOR, (int(paddle1_pos[0]), int(paddle1_pos[1])), paddle_radius)
    pygame.draw.circle(screen, TARGET_COLOR, (int(target_pos[0]), int(target_pos[1])), 5)

    pygame.display.flip()
    clock.tick(60)