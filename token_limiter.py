import time

class TokenLimiter:
    def __init__(self, rate_limit=100000):
        self.tokens_per_min = rate_limit
        self.used_tokens = 0
        self.start = time.time()

    def wait_for_slot(self, tokens_needed):
        if tokens_needed > self.tokens_per_min:
            raise ValueError("To many tokens required.")

        now = time.time()
        if now - self.start >= 60:
            self.start = now 
            self.used_tokens = 0
        
        if self.tokens_per_min <= self.used_tokens + tokens_needed:
            wait_time = 60 - (now - self.start)
            time.sleep(wait_time)
            self.used_tokens = 0
            self.start = time.time()

        self.used_tokens += tokens_needed 
        