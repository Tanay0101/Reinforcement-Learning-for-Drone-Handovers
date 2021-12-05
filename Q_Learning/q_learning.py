import numpy as np
from Environment import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

env = Environment(6000, 5000, 50, 1, 0, 6)
src_dest = [env.give_src_dest() for i in range(20001)]
test_src_dest = src_dest[:10000]
# test_src_dest = [env.give_src_dest() for i in range(10000)]

def exp_policy(num_actions, epsilon, Q_values, state):
	val = np.random.random()
	if val<epsilon:
		action = np.random.randint(0, num_actions-1)
	else:
		q_values = np.array([Q_values.get((state, i), 0) for i in range(num_actions)])
		action = np.argmax(q_values)
	return action



def Q_Learning(Q_values = None, Wrsrp = 1, Who=1, k=6):
	# env = Environment(6000, 5000, 50, Wrsrp, Who, k)
	gamma = 0.3
	alpha = 0.5
	epsilon = 0.25
	
	if Q_values is None:
		Q_values = {} 
	
	r = 100

	rewards = []
	last_r = []
	local_src_dest = src_dest[:]
	local_test_src_dest = test_src_dest[:]
	for i in tqdm(range(0,20001)):

		src, dest = local_src_dest.pop()
		route = env.compute_route(src, dest)
		state = route.popleft()
		state.append(env.sector_cells[src][0][0]) #Setting strongest cell as the initial serving cell
		done = 0
		total_reward = 0	
		

		state = tuple(state)

		while done==0:
			action = exp_policy(k, epsilon, Q_values, state)
			next_state, reward, done, change = env.step(state, route, action, dest)
			reward*=Wrsrp
			reward-=Who*change
			if (state, action) in Q_values:
				Q_values[(state, action)] = (1-alpha)*Q_values[(state, action)] + alpha*(reward + gamma*max([Q_values.get((next_state, i), 0) for i in range(k)]))
			else:
				Q_values[(state, action)] = reward + gamma*max([Q_values.get((next_state, i), 0) for i in range(k)])
			
			total_reward+=reward
			state = next_state

		last_r.append(total_reward)
		
		if not i%r:
			rewards.append(np.average(np.array(last_r)))
			last_r = []

		epsilon*=0.99
	hos = []

	for i in tqdm(range(0,10000)):
		src, dest = local_test_src_dest.pop()
		route = env.compute_route(src, dest)
		state = route.popleft()
		state.append(env.sector_cells[src][0][0]) #Setting strongest cell as the initial serving cell
		done = 0
		total_reward = 0	
		

		state = tuple(state)
		num_ho = 0

		while done==0:
			action = exp_policy(k, 0, Q_values, state)
			next_state, reward, done, change = env.step(state, route, action, dest)
			
			if change:
				num_ho+=1
			total_reward+=reward
			state = next_state

		hos.append(num_ho)




	plt.plot(rewards)
	plt.xlabel('Iterations (x10)')
	plt.ylabel('Average Q_Values')
	plt.ylim([0,60])
	plt.title(f'Q Learning for Who/Wrsrp = {Who}/{Wrsrp}')
	plt.show()

	return Q_values, rewards, np.mean(hos)




# def exp_policy_test(num_actions,Q_values, state):
# 	q_values = np.array([Q_values.get((state, i), 0) for i in range(num_actions)])
# 	action = np.argmax(q_values)

# 	return action


def baseline_hos(k=6):
	# env = Environment(6000, 5000, 50, 1, 0, k)
	hos = []
	rewards = []
	local_test_src_dest = test_src_dest[:]
	for i in tqdm(range(0,10000)):
		src, dest = local_test_src_dest.pop()
		route = env.compute_route(src, dest)
		state = route.popleft()
		state.append(env.sector_cells[src][0][0]) #Setting strongest cell as the initial serving cell
		done = 0
		total_reward = 0	
		

		state = tuple(state)
		num_ho = 0
		while done==0:
			next_state, reward, done, change = env.step(state, route, 0, dest) #Greedy action selection
			if change:
				num_ho+=1
			

			total_reward+=reward
			state = next_state
			
		hos.append(num_ho)
		rewards.append(total_reward)
	average_handovers = sum(hos)/len(hos)
	return np.mean(rewards), average_handovers




def test(Q_table):
	average_handovers = []
	for i in range(len(Q_table)):
		Wrsrp = Q_table[i][1]
		Who = Q_table[i][2]
		k = Q_table[i][3]
		Q_values = Q_table[i][0]
		env = Environment(6000, 5000, 50, Wrsrp, Who, k)
		hos = []
		epsilon = 0
		for i in tqdm(range(0,10000)):
			src, dest = env.give_src_dest()
			route = env.compute_route(src, dest)
			state = route.popleft()
			state.append(env.sector_cells[src][0][0]) #Setting strongest cell as the initial serving cell
			done = 0
			total_reward = 0	
			

			state = tuple(state)
			num_ho = 0

			while done==0:
				action = exp_policy(k, epsilon, Q_values, state)
				next_state, reward, done, change = env.step(state, route, action, dest)
				
				if change:
					num_ho+=1
				total_reward+=reward
				state = next_state

			hos.append(num_ho)

		assert(len(hos)==10000)	
		average_handovers.append(np.mean(hos))

	print(average_handovers)	
	name = [f'{Q_table[i][2]}/{Q_table[i][1]}' for i in range(len(Q_table))]
	name.insert(0,'Baseline')
	average_handovers.insert(0,baseline_hos())	
	plt.bar(name, average_handovers)
	plt.ylabel('Average Number of Handovers')
	plt.xlabel('Who/Wrsrp')
	plt.show()
	print(average_handovers)


if __name__=='__main__':
	Q_values_1, rewards_1, ho1 = Q_Learning(Q_values = None, Wrsrp = 1, Who=0, k=6)
	Q_values_2, rewards_2, ho2 = Q_Learning(Q_values = None, Wrsrp = 1, Who=1/9, k=6)
	Q_values_3, rewards_3, ho3 = Q_Learning(Q_values = None, Wrsrp = 1, Who=1, k=6)
	reward, avg_ho = baseline_hos()

	name = ['baseline', '0/1', '1/9', '1/1']
	vals = [reward, rewards_1[-1], rewards_2[-1], rewards_3[-1]]
	plt.bar(name ,vals)
	plt.ylabel('Average Reward')
	plt.xlabel('Who/Wrsrp')
	plt.show()

	hos = [avg_ho, ho1, ho2, ho3]
	print(hos)
	plt.bar(name, hos)
	plt.ylabel('Average Handovers')
	plt.xlabel('Who/Wrsrp')
	plt.show()

#density of base stations look up	



		
