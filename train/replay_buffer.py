from numpy.ma import size


class ReplayBuffer:
    def __init__(
        self,
        capacity=1000,
        per_domain_capacity=None,
    ):
        
        if capacity <= 0:
             raise ValueError(
                "capacity must be greater than 0"
        )

        if (
            per_domain_capacity is not None
            and per_domain_capacity <= 0
        ):
            raise ValueError(
                "per_domain_capacity must be greater than 0"
        )

        self.capacity = capacity
        self.per_domain_capacity = per_domain_capacity
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def add(self, episode):
        self.buffer.append(episode)

        # Limit how many episodes one domain can occupy.
        if self.per_domain_capacity is not None:
            domain = episode.get("domain")

            same_domain_indices = [
                index
                for index, stored_episode in enumerate(self.buffer)
                if stored_episode.get("domain") == domain
            ]

            if len(same_domain_indices) > self.per_domain_capacity:
                oldest_same_domain_index = same_domain_indices[0]
                self.buffer.pop(oldest_same_domain_index)

        # Global buffer capacity.
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
                already_sampled = any(
                    episode is selected_episode
                    for selected_episode in sampled
                )

                if not already_sampled:
                    sampled.append(episode)

                if len(sampled) >= size:
                    break
        return sampled

    
    def sample_domain_balanced(self, size: int = 10) -> list[dict]:
        if size <= 0 or not self.buffer:
            return []

        target_size = min(size, len(self.buffer))

        # Group stored episodes by domain.
        domain_buckets = {}

        for episode in self.buffer:
            domain = episode.get("domain")

            if domain not in domain_buckets:
                domain_buckets[domain] = []

            domain_buckets[domain].append(episode)

        selected = []

        # First take one episode from every available domain.
        # Use the newest episode from each domain first.
        for domain in domain_buckets:
            if len(selected) >= target_size:
                break

            selected.append(
                domain_buckets[domain][-1]
        )

        # If more samples are required, continue in a
        # round-robin manner across the domains.
        depth = 2

        while len(selected) < target_size:
            added_episode = False

            for domain in domain_buckets:
                bucket = domain_buckets[domain]

                if len(bucket) >= depth:
                    selected.append(
                        bucket[-depth]
                    )

                    added_episode = True

                    if len(selected) >= target_size:
                        break

            if not added_episode:
                break

            depth += 1

                
        return selected    

    

    def sample(self, size=10):
        if size < 0:
            raise ValueError("sample size must be non-negative")

        if size == 0:
            return []

        return self.buffer[-size:]