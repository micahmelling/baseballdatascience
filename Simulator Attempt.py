#Working file

import random

hit = ('called strike', 'swinging strike', 'ball', 'foul ball', 'line drive',
       'fly ball', 'groundball')
          
pitch = ('fastball', 'changeup', 'curveball', 'slider')

outcome = ('single', 'double', 'triple', 'out')

outs = 0
runner = 0
runs = 0

def pitch_thrown():    
    pitch1 = random.choice(pitch) 
    print('pitch thrown: %s' % (pitch1))
    return pitch1
  
def hitter_action():
    global hit1
    hit1 = random.choice(hit)
    print('hitter outcome: %s' % (hit1))
    return hit1
    
def play_outcome():
    global outcome1
    outcome1 = random.choice(outcome)
    print('play_outcome: %s' % (outcome1))
    
    #Increment outs and runners
    if outcome1 == 'out':
        print('out recorded')
        
    elif outcome1 == 'single':
        print('single')
        
    elif outcome1 == 'double':
        print('double')
        
    elif outcome1 == 'triple':
        print('triple')
    
    return
    
def strike_recorded():
    print('pitch was a strike')
    outcome1 == 'strike recorded'
    
def ball_recorded():
    print('pitch was a ball')
    outcome1 == 'ball recorded'
              
def hit_outcomes():
    if hit1 in ('line drive', 'fly ball', 'groundball'):
        play_outcome()
        
    elif hit1 in ('called strike', 'swinging strike', 'foul ball'):
        strike_recorded()       
        
    elif hit1 == 'ball':
        ball_recorded()
              
def inning():
    outs = 0
    runner = 0
    runs = 0       
    
    pitch_thrown()
    hitter_action()
    hit_outcomes()

inning()   

    

        



