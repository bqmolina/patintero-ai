from environment import Environment, Renderer

if __name__ == "__main__":
    env = Environment()
    renderer = Renderer(env)
    renderer.running = True

    while renderer.running:
        step = env.step(1, 2)
        renderer.render()
        if step[2]:
            env.reset()
    renderer.quit()

    print("Game closed.") 