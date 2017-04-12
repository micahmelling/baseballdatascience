#Working file

#To-do list
#Runs scored

import numpy as np

hit = ['called strike', 'swinging strike', 'ball', 'line drive',
       'fly ball', 'groundball']
          
pitch = ['fastball', 'changeup', 'curveball', 'slider']

outcome = ['single', 'double', 'triple', 'out']

outs = 0
runner = 0
runs = 0
strikes = 0
balls = 0

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
    outcome1 = np.random.choice(outcome, 1, p=[0.25, 0.15, 0.10, 0.5])
    print('play_outcome: %s' % (outcome1))
    
    if outcome1 == 'out':
        global outs
        outs += 1
        print('out recorded; number of outs: %s' % (outs))
          
    elif outcome1 == 'single':
        print('runner goes to first')
        
    elif outcome1 == 'double':
        print('runner goes to second')
        
    elif outcome1 == 'triple':
        print('runner goes to third')
    
    return
    
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
    balls +=1
    
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

    

        



