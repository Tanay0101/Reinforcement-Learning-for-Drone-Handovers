import numpy as np
from collections import deque
import heapq
import math
import matplotlib.pyplot as plt
import tensorflow as tf

class Environment:
	'''
	Length and width  = dimensions of the world
	div_size = area of each sector
	Wrsrp = weight for rsrp values
	Who = weight for hand overs
	k = number of cells to choose from in each sector
	7 basestations each with 3 cells are present in the world
	'''
	def __init__(self, length, width, div_size, Wrsrp, Who, k):
		self.Wrsrp = Wrsrp
		self.Who = Who
		self.length = length
		self.width = width
		self.div_size = div_size
		self.k = k
		self.cell_location = {
								1: [0, 0, 0],
								2: [0, 0, 1],
								3: [0, 0, 2],
								4: [length/4, 0, 0],
								5: [length/4, 0, 1],
								6: [length/4, 0, 2],
								7: [length/8, width/4, 0],
								8: [length/8, width/4, 1],
								9: [length/8, width/4, 2],
								10: [-length/8, width/4, 0],
								11: [-length/8, width/4, 1],
								12: [-length/8, width/4, 2],
								13: [-length/4, 0, 0],
								14: [-length/4, 0, 1],
								15: [-length/4, 0, 2],
								16: [-length/8, -width/4, 0],
								17: [-length/8, -width/4, 1],
								18: [-length/8, -width/4, 2],
								19: [length/8, -width/4, 0],
								20: [length/8, -width/4, 1],
								21: [length/8, -width/4, 2]
							}

		print(f'Location and directions of cells are: \n{self.cell_location} \n \n')

		self.sector_cells = self.calculate_strongest_k()
		self.sector_cells = self.normalise()

	def give_src_dest(self):
		'''Randomly generates src and dest coordinates '''
		src_x, dest_x = np.random.choice(list(range(-1*self.length//2, self.length//2 + self.div_size, self.div_size)), size = 2, replace = False)
		src_y, dest_y = np.random.choice(list(range(-1*self.width//2, self.width//2 + self.div_size, self.div_size)), size = 2, replace = False)
		return (src_x, src_y), (dest_x, dest_y)

	def compute_route(self, src, dest):
		'''Gives route of drone in form of a queue with each element of form (x, y, direction). Direction is from 0-3, East- South '''
		route = deque([])
		src_x, src_y = src
		dest_x, dest_y = dest
		x = src_x
		y = src_y

		while x!=dest_x or y!=dest_y:
			if x<dest_x:
				if y < dest_y:   
					direction = 1 #north east
					route.append([x,y,direction])
					x += self.div_size
					y += self.div_size
				elif y > dest_y: 
					direction = 7 #south east
					route.append([x,y,direction])
					x += self.div_size
					y -= self.div_size
				else:
					direction = 0  #east
					route.append([x, y, direction])
					x += self.div_size
					
			elif x>dest_x:
				if y < dest_y:
					direction = 3 #north-west
					route.append([x,y,direction])
					x -= self.div_size
					y += self.div_size
				elif y > dest_y:
					direction = 5 #south-west
					route.append([x,y,direction])
					x -= self.div_size
					y -= self.div_size
				else :
					direction = 4 # west
					route.append([x, y, direction])
					x -= self.div_size

			else :   #when x == dest_x => no chane in x
				if y< dest_y:  
					direction = 2 #north
					route.append([x, y, direction])
					y += self.div_size
				elif y > dest_y:
					direction = 6 #south
					route.append([x, y, direction])
					y -= self.div_size 

				
		route.append([dest_x, dest_y, 0])
			
		return route

	def step(self, state, route, action, dest):
		'''Returns next_state (x, y, direction, serving_cell), reward and done. new_cell is a list which contains [cell_number, RSRP_val]'''
		sector = (state[0], state[1])
		strongest_k_cells = self.sector_cells[sector]
		new_cell = strongest_k_cells[action]
		
		if route:
			next_state = route.popleft()
			index = new_cell[0]
			depth = len(self.cell_location)
			one_hot_cell = make_one_hot(index, depth)
			one_hot_direction = make_one_hot(next_state[-1]+1, 8)
			next_state = next_state[:-1]
			next_state.extend(one_hot_direction)
			next_state.extend(one_hot_cell)

		else:
			print(state)
			print(dest)
			state = list(state)
			print(tuple(state[:2])==dest)

		if (next_state[0], next_state[1])==dest:
			done = 1
		else:
			done = 0


		next_state = tuple(next_state)

		change = 1 if new_cell[0]!=int(np.argmax(state[10:])) + 1 else 0  

		reward = self.Wrsrp*new_cell[1] - self.Who*change

		return next_state, reward, done, change


	def calculate_strongest_k(self):
		'''Returns dictionary containing strongest k cells for each sector and their respective RSRP values.'''
		sector_cells = {}
		for x in range(-self.length//2, self.length//2 + self.div_size, self.div_size):
			for y in range(-self.width//2, self.width//2 + self.div_size, self.div_size):
				rsrp = np.zeros((21,2))
				iterations = 1
				for i in range(iterations):
					x_temp = x #+ np.random.randint(0,50)
					y_temp = y #+ np.random.randint(0,50)
					rsrp += np.array(self.calculate_rsrp(x_temp, y_temp))
				
				rsrp = list(map(list, rsrp/iterations))		
				# if x==0 and y==0:
				# 	print(rsrp)
				# 	print('\n\n\n\n')		
				sector_cells[(x, y)] = heapq.nlargest(self.k, rsrp, key = lambda x:x[1])


				
				if (x == 0 or x == 50) and (y==0 or y == 100):
					print(f'Strongest cells for sector {x, y}')
					print(sector_cells[(x,y)])
					print()
					# print(rsrp)
					print()
		
		return sector_cells 

	def calculate_rsrp(self, x, y):
		'''Calculates path loss of every cell for a sector and returns list containing cell number and rsrp value'''
		tower_height = 35 #meters #From internet
		drone_height = 100 #meters #From research paper
		alpha = 3.04
		theta_0 = -3.61
		A = -23.29
		B = 4.14
		neta = 20.70
		a = -0.41
		pi = 3.1416
		sigma_0 = 5.86
		power = 46

		path_loss = []

		for i in range(21):
			# print(i)
			cell_x, cell_y, cell_dir = self.cell_location[i+1]
			d = math.sqrt((x-cell_x)**2 + (y-cell_y)**2)
			if d!=0:
				if x>cell_x and y>=cell_y:
					angle = math.atan((y-cell_y)/(x-cell_x))
					if 0<=angle<=pi/3:
						sec_dir = 0
					else:
						sec_dir = 1

				elif x<=cell_x and y>=cell_y:
					sec_dir = 1

				elif x>cell_x and y<cell_y:
					angle = abs(math.atan((cell_y-y)/(x-cell_x)))
					if 0<=angle<=pi/3:
						sec_dir = 0
					else:
						sec_dir = 2
				elif x<=cell_x and y<=cell_y:
					sec_dir = 2


				# if x==0 and y==0 and i==3:
				# 	print('\n\n\n')
				# 	print(sec_dir)
				# 	assert(sec_dir==1)
				theta = math.atan((drone_height-tower_height)/d)*180/pi
				if sec_dir!=cell_dir:
					path_loss.append([i+1,float('-inf')])
				
				else:
					euclidean_distance = math.sqrt(d**2+(drone_height-tower_height)**2)
					# loss = max(23.9-1.8*math.log10(drone_height), 20)*math.log10(euclidean_distance) + 20*math.log10(40*pi*1.5*(10**9)/3) + np.random.normal(0, 4.2*math.exp(-0.0046*drone_height))				
					
					loss = 28 + 22*math.log10(euclidean_distance) + 20*math.log10(2.5)+ np.random.normal(0, 4.2*math.exp(-0.0046*drone_height)) # Assuming frequency of 2.5GHz
					antenna_gain = self.calculate_gain(x, y, cell_x, cell_y, tower_height, drone_height)

					path_loss.append([i+1,power + antenna_gain -loss]) #Asssuming 46 dBm tranmitted power
			else:
				path_loss.append([i+1, float('-inf')]) #No signal

		return path_loss

	def calculate_gain(self, x, y, cell_x, cell_y, tower_height, drone_height):
		'''Calculates antenna gain'''
		d = math.sqrt((x-cell_x)**2 + (y-cell_y)**2)
		delta_h = drone_height-tower_height
		pi = math.pi

		theta = math.atan(d/delta_h)*180/pi
		# print(theta)

		if x-cell_x==0:
			phi = 90
		else:
			phi = math.atan(abs(y-cell_y)/abs(x-cell_x))*180/pi
		
		Aev = -min((12*(theta-90)/65)**2, 30)
		Aeh = -min(12*(phi/65)**2, 30)

		Ae = -min(-Aev-Aeh, 30)
		Ge = 8+Ae
		N = 8
		Ga = 10*math.log10(math.sin(N/2*pi*math.cos(theta))**2/(N**2*math.sin(1/2*pi*math.cos(theta))**2))

		return Ge+Ga

	def normalise(self):
		sector_cells = self.sector_cells
		data = np.array(list(sector_cells.values()))[:,:,1]
		data = data.flatten()
		print(data)
		min_val = float('-inf')
		data = list(data)
		data.sort()
		max_val = data[-1]
		i = 0
		# while min_val==float('-inf'):
		# 	min_val = data[i]
		# 	i+=1

		min_val = -100
		print(f'Maximum RSRP value = {max_val}, Minimum RSRP value = {min_val}')

		for key in sector_cells:
			sector_cells[key] = np.array(sector_cells[key])
			sector_cells[key][:, 1] = (sector_cells[key][:, 1] - min_val)/(max_val-min_val)
			# for i in range(self.k):
			# 	if sector_cells[key][i, 1]<0:
			# 		sector_cells[key][i, 1] = 0

		data = np.array(list(sector_cells.values()))[:,:,1]
		max_val = np.max(data)
		min_val = np.min(data)
		print(f'New maximum RSRP value = {max_val}, New minimum RSRP value = {min_val}')

		return sector_cells


def make_one_hot(index, depth):
	one_hot = [0]*depth
	index = int(index)
	one_hot[index-1] = 1
	return one_hot


if __name__ == '__main__':
	env = Environment(6000, 5000, 50, 1, 1, 6)
	print('\n\n')
	sector_cells = env.sector_cells

	x = []
	y = []
	colors = []

	vals = []
	for key,value in sector_cells.items():
		vals.append(value[0][1])
		if value[0][1]>0 and 12<value[0][0]<=15:
			x.append(key[0])
			y.append(key[1])
			colors.append(value[0][0])


	# print(sum(vals)/len(vals))
	# print(min(vals))
			
	plt.scatter(x, y, c = colors)
	plt.title('Strongest Cell Association Map')
	plt.xlabel('X Coordinate')
	plt.ylabel('Y Coordinate')
	plt.show()
	
	# colors = []
	# vals = []
	# x = []
	# y = []
	# for key,value in sector_cells.items():
	# 	vals.append(value[0][1])
	# 	x.append(key[0])
	# 	y.append(key[1])
	# 	colors.append(value[2][0])


	# plt.scatter(x, y, c = colors)
	# plt.title('Strongest Cell Association Map')
	# plt.xlabel('X Coordinate')
	# plt.ylabel('Y Coordinate')
	# plt.show()





