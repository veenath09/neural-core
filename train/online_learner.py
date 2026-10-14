class OnlineLearner:
    def __init__(self):
        self.episode_count = 0

    def update(self, reward):
        self.episode_count += 1

        return {
            "episode": self.episode_count,
            "reward": reward
        }