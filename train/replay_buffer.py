class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def add(self, episode):
        self.buffer.append(episode)

        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def increment_ages(self) -> None:
        for episode in self.buffer:
            episode["age"] = episode.get("age", 0) + 1

    def age_band(self, age: int) -> str:
        if age <= 2:
            return "recent"

        if age <= 5:
            return "medium"

        return "old"

    def sample_age_banded(self, size=10):
        if size <= 0:
            return []

        bands = {
            "recent": [],
            "medium": [],
            "old": [],
        }

        for episode in self.buffer:
            band = self.age_band(episode.get("age", 0))
            bands[band].append(episode)

        sampled = []

        # First, try to take one episode from every age band.
        for band_name in ("recent", "medium", "old"):
            if bands[band_name] and len(sampled) < size:
                sampled.append(bands[band_name][-1])

        # If more samples are requested, fill from remaining episodes.
        if len(sampled) < size:
            for episode in reversed(self.buffer):
                if episode not in sampled:
                    sampled.append(episode)

                if len(sampled) >= size:
                    break

        return sampled

    def sample(self, size=10):
        return self.buffer[-size:]  