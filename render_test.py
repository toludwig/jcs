import pygame
from Juggler_gym import Juggler, OptimalAgent

# Initialize Pygame
pygame.init()
pygame.display.init()

# Create your environment with render_mode="human"
env = Juggler(pattern=[3, 3, 3], render_mode="human")
agent = OptimalAgent(pattern=[3, 3, 3])

# Reset the environment to initialize everything
observations, info = env.reset()

running = True
done = False
step = 0
while running and not done:
    print(step)
    step += 1

    # get action
    action = agent.act(observations, info)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # step
    observations, reward, done, _, info = env.step(action)

    # Call your render function
    env.render()

    # Optional: control frame rate
    if env.clock:
        env.clock.tick(30)  # 30 FPS

# Clean up
env.close()
pygame.quit()