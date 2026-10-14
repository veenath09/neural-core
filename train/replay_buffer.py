class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []

    def add(self, episode):
        self.buffer.append(episode)

        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, size=10):
        return self.buffer[-size:]