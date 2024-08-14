class Transmitter:
    def __init__(self, user_id, reaction_time, perseverance, authority_level, sensitivity):
        self.user_id = user_id
        self.reaction_time = reaction_time
        self.perseverance = perseverance
        self.authority_level = authority_level
        self.sensitivity = sensitivity

    def __repr__(self):
        return f"Transmitter(user_id={self.user_id}, reaction_time={self.reaction_time}, perseverance={self.perseverance}, authority_level={self.authority_level}, sensitivity={self.sensitivity})"

class Receiver:
    def __init__(self, user_id, attitude, number_of_messages, source_authority):
        self.user_id = user_id
        self.attitude = attitude
        self.number_of_messages = number_of_messages
        self.source_authority = source_authority

    def __repr__(self):
        return f"Receiver(user_id={self.user_id}, attitude={self.attitude}, number_of_messages={self.number_of_messages}, source_authority={self.source_authority})"

# Example usage
transmitter1 = Transmitter(user_id='user1', reaction_time=0.5, perseverance=0.7, authority_level=1000, sensitivity='believe-and-forward')
receiver1 = Receiver(user_id='user2', attitude='adopt', number_of_messages=10, source_authority=1000)

print(transmitter1)
print(receiver1)
