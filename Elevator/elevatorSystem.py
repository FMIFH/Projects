import random
from itertools import combinations_with_replacement
import numpy as np
import gym

class Passenger:
    def __init__(self,n_floors):
        self.initialPos = random.randint(0,n_floors)
        self.destination = random.choice(list(range(0,self.initialPos)) + list(range(self.initialPos, n_floors)))
        self.total_time = 0
        self.inElevator = False
    
    def addToElevator(self) -> None:
        self.inElevator = True

    def isInElevator(self) -> bool:
        return self.inElevator

    def getInitialPos(self) -> int:
        return self.initialPos
    
    def getDestination(self) -> int:
        return self.destination
    
    def getTotalTime(self) -> int:
        return self.total_time
    
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
        if passenger.getInitialPos() == self.position and not passenger.isInElevator() and len(self.passengers) < self.capacity:
            self.passengers.append(passenger)
            self.destinations[np.where(self.destinations == -1)[0][0]] = passenger.getDestination()
            passenger.addToElevator()

    def dropPassengers(self) -> list:
        dropped = []
        for passenger in self.passengers:
            if self.position == passenger.getDestination():
                dropped.append(passenger)
                self.passengers.remove(passenger)
        self.destinations[np.where(self.destinations == self.position)] = -1
        return dropped
    
    def getDoorsStatus(self) -> bool:
        return self.doorsOpen
    
    def moveDoors(self) -> list:
        self.doorsOpen = not self.doorsOpen

    def moveElevator(self,direction:int) -> None:
        if not self.doorsOpen:
            self.position = min(max(self.position + direction, 0), self.n_floors)

    def getElevatorPos(self) -> int:
        return self.position
    
    def labeledElevatorStatus(self):
        return{
            'Current Floor' : self.position,
            'Doors Open' : self.doorsOpen,
            'Max Capacity' : self.capacity,
            'Current Passengers' : len(self.passengers),
            'Destinations' : self.destinations
        }
        
    def getElevatorStatus(self):
        return np.append(np.array([self.position,self.doorsOpen,self.capacity,len(self.passengers)]),self.destinations)
    
    def inElevatorDestinations(self):
        return self.destinations


class ElevatorSystem:
    
    def __init__(self, n_eleveators,n_floors):
        self.n_elevators = n_eleveators
        self.n_floors = n_floors
        self.elevators = [Elevator(self.n_floors) for i in range(n_eleveators)]
        self.passengers = []
        self.total_time = 0
        self.total_delivered = 0

    def reset(self):
        self.elevators = [Elevator(self.n_floors) for i in range(self.n_elevators)]
        self.passengers = []
        self.total_time = 0

    def actions(self):
        return np.array(list(combinations_with_replacement([-1,0,1],self.n_elevators)))

    def moveElevator(self, elevator:int, direction:int) -> None:
        self.elevators[elevator].moveElevator(direction)    

    def moveDoors(self, elevator: int):
        self.elevators[elevator].moveDoors()
        if self.elevators[elevator].getDoorsStatus():
            for p in self.passengers:
                self.elevators[elevator].addPassenger(p)
            return self.elevators[elevator].dropPassengers()
        return []
    
    def labeledEnvironmentStatus(self):
        elevators_status = [e.labeledElevatorStatus() for e in self.elevators]
        floors_waiting = np.zeros(self.n_floors)
        floors_waiting[list(set([p.getInitialPos()-1 for p in self.passengers if not p.isInElevator()]))] = 1

        return {
            'Elevators' : elevators_status,
            'Floors waiting': list(set([p.getInitialPos() for p in self.passengers if not p.isInElevator()])),
        }

    def state(self):
        elevators_status = np.array([e.getElevatorStatus() for e in self.elevators])
        floors_waiting = np.zeros(self.n_floors)
        floors_waiting[list(set([p.getInitialPos()-1 for p in self.passengers if not p.isInElevator()]))] = 1
        return np.append(elevators_status,floors_waiting)

    def step(self, action:list) -> int:
        if self.total_time % 5 == 0:
            self.passengers.append(Passenger(self.n_floors))

        self.total_time += 1
        for p in self.passengers:
            p.time()

        dropped = []
        for elevator, p in enumerate(action):
            if p == 0:
                dropped += self.moveDoors(elevator)
            else:
                self.moveElevator(elevator,p)
        
        reward = -1 + len(dropped)*self.n_floors

        for p in dropped:
            self.passengers.remove(p)
            self.total_delivered += 1

        
        done = (sum([p.getTotalTime() for p in self.passengers]) > 10**6 or self.total_time > 10**6)
        return self.state() , reward , done , self.labeledEnvironmentStatus()
