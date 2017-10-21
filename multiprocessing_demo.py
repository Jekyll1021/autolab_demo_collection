from multiprocessing import Pool
from multiprocessing import Process
from agents import *

agent1 = sequenceAgent([moveAgent((7, 7))])
agent2 = sequenceAgent([moveAgent((5, 7))])

def action(agent):
	agent.pre_train()
	return agent.execute()

if __name__ == '__main__':
    p1 = Process(target=action, args=(agent1,))
    p1.start()
    p2 = Process(target=action, args=(agent2,))
    p2.start()