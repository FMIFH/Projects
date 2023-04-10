import random
from itertools import combinations_with_replacement
import numpy as np
from gym import Env, spaces

class Passenger:
    def __init__(self,n_floors):
        self.initialPos = random.randint(0,n_floors-1)
        self.destination = random.choice(list(range(0,self.initialPos)) + list(range(self.initialPos, n_floors)))
        self.total_time = 0
        self.inElevator = False
        
    
    def addToElevator(self) -> None:
        self.inElevator = True
    
    def time(self) -> None:
        self.total_time += 1

    def getPassengerStatus(self):
        return [self.initialPos,self.destination, self.inElevator]

class Elevator:

    def __init__(self,n_floors):
        self.n_floors = n_floors
        self.position = 0
        self.passengers = []
        self.doorsOpen = False
        self.capacity = 4
        self.destinations = np.full(self.capacity,-1)

    def addPassenger(self, passenger:Passenger) -> None:
        if passenger.initialPos == self.position and not passenger.inElevator and len(self.passengers) < self.capacity:
            self.passengers.append(passenger)
            self.destinations[np.where(self.destinations == -1)[0][0]] = passenger.destination
            passenger.addToElevator()

    def dropPassengers(self) -> list:
        dropped = []
        for passenger in self.passengers:
            if self.position == passenger.destination:
                dropped.append(passenger)
                self.passengers.remove(passenger)
        self.destinations[np.where(self.destinations == self.position)] = -1
        return dropped
    
    def moveDoors(self) -> list:
        self.doorsOpen = not self.doorsOpen

    def moveElevator(self,direction:int) -> None:
        if not self.doorsOpen:
            self.position = min(max(self.position + direction, 0), self.n_floors-1)

    
    def labeledElevatorStatus(self):
        return{
            'Current Floor' : self.position,
            'Doors Open' : self.doorsOpen,
            'Max Capacity' : self.capacity,
            'Current Passengers' : len(self.passengers),
            'Destinations' : self.destinations,
        }

class ElevatorSystemEnv(Env):
    
    def __init__(self, n_elevators,n_floors):
        self.n_floors = n_floors
        self.n_elevators = n_elevators
        self.elevators = [Elevator(self.n_floors) for _ in range(self.n_elevators)]
        self.passengers = []
        self.total_time = 0

        self.done = False
        self.observation = np.array([])
        self.reward = 0
        self.info = {}

        #-1 go down 1 go up
        #0 do nothing
        #2 move doors
        self.actions =  np.array(list(combinations_with_replacement([-1,0,1,2],self.n_elevators)))

        self.action_space = spaces.Discrete(self.actions.shape[0])
        #elevator_pos, elevator_doors, elevator_pass, elevator_destinations  * n_elevators
        #floors_with_pass
        self.observation_space = spaces.Box(low=np.array([0,0,0,-1,-1,-1,-1]*n_elevators + [0]*n_floors),
                                            high=np.array([n_floors-1,1,4,n_floors-1,n_floors-1,n_floors-1,n_floors-1]*n_elevators + [1]*n_floors),
                                            shape=(7*n_elevators+n_floors,),dtype=np.int8)


    def elevator_info(self, e:Elevator):
        position = e.position
        destinations = e.destinations
        passengers = len(e.passengers)
        doors = e.doorsOpen

        return np.concatenate((np.array([position,doors,passengers]), destinations), axis=None) #3+4
    

    def floors_with_unpicked_pass(self):
        floors = [0]*self.n_floors
        for p in self.passengers:
            if not p.inElevator:
                floors[p.initialPos] = 1
            if sum(floors) == self.n_floors:
                break
        return np.array(floors)


    def spawnPassenger(self):
        if random.random() <= 0.01:
            self.passengers.append(Passenger(self.n_floors))


    def moveElevator(self, elevator, direction):
        self.elevators[elevator].moveElevator(direction)

    def moveDoors(self, elevator: int):
        self.elevators[elevator].moveDoors()
        if self.elevators[elevator].doorsOpen:
            for p in self.passengers:
                self.elevators[elevator].addPassenger(p)
            return self.elevators[elevator].dropPassengers()
        return []

    def reset(self):
        self.total_time = 0
        self.elevators = [Elevator(self.n_floors) for _ in range(self.n_elevators)]
        self.passengers = []
        self.done = False
        self.observation = np.array([])
        #elevator_pos, elevator_doors, elevator_pass, elevator_destinations  * n_elevators
        #floors_with_pass
        for e in self.elevators:
            self.observation = np.concatenate((self.observation, self.elevator_info(e)), axis=None)

        self.observation = np.concatenate((self.observation, self.floors_with_unpicked_pass()), axis=None)
       
        return self.observation

    def render(self):
        pass

    def step(self, action) -> int:
        self.total_time += 1
        for p in self.passengers:
            p.time()
            
        if (self.total_time % 10 == 0) and (self.total_time < 10**4-2*self.n_floors):
            self.passengers.append(Passenger(self.n_floors))
        
        dropped = []
        for elevator, p in enumerate(self.actions[action]):
            if p == 2:
                dropped += self.moveDoors(elevator)
            else:
                self.moveElevator(elevator,p)
 
        for p in dropped:
            self.passengers.remove(p)

        self.reward = len(dropped)
        self.done = (self.total_time >= 10**4)
        self.observation = np.array([])
        for e in self.elevators:
            self.observation = np.concatenate((self.observation, self.elevator_info(e)), axis=None)

        self.observation = np.concatenate((self.observation, self.floors_with_unpicked_pass()), axis=None)

        return self.observation, self.reward, self.done, self.info
