#Working file

#To-do list
#Soreboard 
#Add walks in the runner equation
#Add more runners

#Later: adapt to a team

import numpy as np

hit = ['called strike', 'swinging strike', 'ball', 'line drive',
       'fly ball', 'groundball']
          
pitch = ['fastball', 'changeup', 'curveball', 'slider']

outcome = ['single', 'double', 'triple', 'out']

outs = 0
runner = 0
runs = 0
runner2 = 0
strikes = 0
balls = 0
move = 0
hit1_counter = 'n'

def pitch_thrown():    
    pitch1 = np.random.choice(pitch, 1, p=[0.4, 0.1, 0.3, 0.2]) 
    print('pitch thrown: %s' % (pitch1))
    return pitch1
  
def hitter_action():
    global hit1
    hit1 = np.random.choice(hit, 1, p=[0.15, 0.15, 0.2, 0.1, 0.2, 0.2]) 
    print('hitter outcome: %s' % (hit1))
    return hit1
    
def play_outcome():
    global outcome1
    
    outcome1 = np.random.choice(outcome, 1, p=[0.30, 0.20, 0.10, 0.4])
    print('play_outcome: %s' % (outcome1))
    
    if outcome1 == 'out':
        global outs
        outs += 1
        print('out recorded; number of outs: %s' % (outs))    
         
def runner_generator():
    if hit1_counter == 'n':
        runner1()
   
    elif hit1_counter == 'a':
        runner2()
        
    elif hit1_counter == 'b':
        runner3()
        
    elif hit1_counter == 'c':
        runner4()
        
    elif hit1_counter == 'd':
        runner5()
        
    else:
        return

def runner1():
    global runnera 
    global hit1_counter
    
    if outcome1 == 'single':
        #runnera += 1
        print('single! first runner goes to first')  
        hit1_counter = 'a'
        
    elif outcome1 == 'double':
        #runnera += 2
        print('double! first runner goes to second')  
        hit1_counter = 'a'
        
    elif outcome1 == 'triple':
        #runnera += 3
        print('triple! first runner goes to third')
        hit1_counter = 'a'
       
def runner2():
    global runnerb
    global hit1_counter
 
    if outcome1 == 'single':
        #runnerb += 1
        print('single! second runner goes to first')  
        hit1_counter = 'b'
        
    elif outcome1 == 'double':
        #runnerb += 2
        print('double! second runner goes to second')  
        hit1_counter = 'b'
        
    elif outcome1 == 'triple':
        #runnerb += 3
        print('triple! second runner goes to third')
        hit1_counter = 'b'
          
def runner3():
    global runnerc
    global hit1_counter
    
    if outcome1 == 'single':
        #runnerc += 1
        print('single! third runner goes to first')  
        hit1_counter = 'c'
        
    elif outcome1 == 'double':
        #runnerc += 2
        print('double! third runner goes to second')  
        hit1_counter = 'c'
        
    elif outcome1 == 'triple':
        #runnerc += 3
        print('triple! third runner goes to third')
        hit1_counter = 'c'
        
def runner4():
    global runnerd
    global hit1_counter
    
    if outcome1 == 'single':
        #runnerd += 1
        print('single! fourth runner goes to first')  
        hit1_counter = 'd'
        
    elif outcome1 == 'double':
        #runnerd += 2
        print('double! fourth runner goes to second')  
        hit1_counter = 'd'
        
    elif outcome1 == 'triple':
        #runnerd += 3
        print('triple! fourth runner goes to third')
        hit1_counter = 'd'    
    
def runner5():
    global runnerd
    global hit1_counter
    
    if outcome1 == 'single':
        #runnere += 1
        print('single! fifth runner goes to first')  
        
    elif outcome1 == 'double':
        #runnere += 2
        print('double! fifth runner goes to second')  
        
    elif outcome1 == 'triple':
        #runnere += 3
        print('triple! fifth runner goes to third')
               
def strike_recorded():
    outcome1 == 'strike recorded'
    global strikes
    strikes +=1
    
    if strikes < 3:
        print('pitch was a strike; number of strikes: %s' % (strikes))
        
    else:
        strikes = 0
        global outs
        outs += 1
        print('strikeout! number of outs: %s' % (outs))
        
def reset_strikes():
    global strikes
    strikes = 0
    
def ball_recorded():
    outcome1 == 'ball recorded'
    global balls
    balls += 1
    
    if balls < 4:
        print('pitch was a ball; number of balls: %s' % (balls))
        
    else:
        balls = 0
        print('walk! runner goes to first')
    
def reset_balls():
    global balls
    balls = 0    
       
def hit_outcomes():
    if hit1 in ('line drive', 'fly ball', 'groundball'):
        play_outcome()
        reset_strikes()
        reset_balls()
        runner_generator() 
        
    elif hit1 in ('called strike', 'swinging strike', 'foul ball'):
        strike_recorded()       
        
    elif hit1 == 'ball':
        ball_recorded()
             
def inning():    
    pitch_thrown()
    hitter_action()
    hit_outcomes()
    
    
while outs < 3:
    inning()   
    print
