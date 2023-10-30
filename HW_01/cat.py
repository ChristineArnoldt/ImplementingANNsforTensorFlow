class cat:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, name):
        self.name = name
        self.greeted = False
    
    def greet(self, partner):
        if self.greeted == False:
            print(f"Hello, I am {self.name}! I see you are also cool fluffy kitty {partner.name}, let's together purr at the human, so that they shall give us food.")
            partner.greeted = True
            partner.greet(self)
        else:
            print(f"Hello, {partner.name}! Nice to meet you. I am {self.name}. I'd love to spend some time together.")
            self.greeted = False