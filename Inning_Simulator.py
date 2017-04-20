#Working file

#To-do list
#Add runners
#Add walks in the runner equation

import numpy as np

hit = ['called strike', 'swinging strike', 'ball', 'line drive',
       'fly ball', 'groundball']
          
pitch = ['fastball', 'changeup', 'curveball', 'slider']

outcome = ['single', 'double', 'triple', 'out']

homerun = ['home run', 'not a home run']

outs = 0
runs = 0
strikes = 0
balls = 0
move = 0
hit1_counter = 'n'
signal = 'n'
signalb = 'n'
signalc = 'n'
signald = 'n'
signale = 'n'
useless_counter = 1
hit1 = 'n'

runnerg = 0
runnerf = 0
runnere = 0
runnerd = 0
runnerc = 0
runnerb = 0
runnera = 0

def pitch_thrown():    
    pitch1 = np.random.choice(pitch, 1, p=[0.4, 0.1, 0.3, 0.2]) 
    print('pitch thrown: %s' % (pitch1))
    return pitch1
  
def home_run_outcome():
    global runs
    global home_runs
    home_runs = np.random.choice(homerun, 1, p=[0.05, 0.95]) 
    
    if home_runs == 'home run':
        runs +=1
        print('batter smacks a home run! team has now scored %s run' % (runs))
        

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
        
    elif hit1_counter == 'e':
        runner6()
        
    elif hit1_counter == 'f':
        runner7()
        
    else:
        return

def runner1():
    global runnera 
    global hit1_counter
    
    if outcome1 == 'single':
        runnera += 1
        print('single! first runner goes to first')  
        hit1_counter = 'a'
        
    elif outcome1 == 'double':
        runnera += 2
        print('double! first runner goes to second')  
        hit1_counter = 'a'
        
    elif outcome1 == 'triple':
        runnera += 3
        print('triple! first runner goes to third')
        hit1_counter = 'a'
       
def runner2():
    global runnerb
    global runnera
    global hit1_counter
 
    if outcome1 == 'single':
        runnerb += 1
        runnera += 1
        print('single! second runner goes to first')  
        hit1_counter = 'b'
        
    elif outcome1 == 'double':
        runnerb += 2
        runnera += 2
        print('double! second runner goes to second')  
        hit1_counter = 'b'
        
    elif outcome1 == 'triple':
        runnerb += 3
        runnera += 3
        print('triple! second runner goes to third')
        hit1_counter = 'b'
          
def runner3():
    global runnerc
    global runnerb
    global runnera
    global hit1_counter
    
    if outcome1 == 'single':
        runnerc += 1
        runnerb += 1
        runnera += 1
        print('single! third runner goes to first')  
        hit1_counter = 'c'
        
    elif outcome1 == 'double':
        runnerc += 2
        runnerb += 2
        runnera += 2
        print('double! third runner goes to second')  
        hit1_counter = 'c'
        
    elif outcome1 == 'triple':
        runnerc += 3
        runnerb += 3
        runnera += 3
        print('triple! third runner goes to third')
        hit1_counter = 'c'
        
def runner4():
    global runnerd
    global runnerc
    global runnerb
    global runnera
    global hit1_counter
    
    if outcome1 == 'single':
        runnerd += 1
        runnerc += 1
        runnerb += 1
        runnera += 1
        print('single! fourth runner goes to first')  
        hit1_counter = 'd'
        
    elif outcome1 == 'double':
        runnerd += 2
        runnerc += 2
        runnerb += 2
        runnera += 2
        print('double! fourth runner goes to second')  
        hit1_counter = 'd'
        
    elif outcome1 == 'triple':
        runnerd += 3
        runnerc += 3
        runnerb += 3
        runnera += 3
        print('triple! fourth runner goes to third')
        hit1_counter = 'd'    
    
def runner5():
    global runnere
    global runnerd
    global runnerc
    global runnerb
    global runnera
    global hit1_counter
    
    if outcome1 == 'single':
        runnere += 1
        runnerd += 1
        runnerc += 1
        runnerb += 1
        runnera += 1
        print('single! fifth runner goes to first')  
        hit1_counter = 'e'
        
    elif outcome1 == 'double':
        runnere += 2
        runnerd += 2
        runnerc += 2
        runnerb += 2
        runnera += 2
        print('double! fifth runner goes to second') 
        hit1_counter = 'e'
        
    elif outcome1 == 'triple':
        runnere += 3
        runnerd += 3
        runnerc += 3
        runnerb += 3
        runnera += 3
        print('triple! fifth runner goes to third')
        hit1_counter = 'e'
        
def runner6():
    global runnerf
    global runnere
    global runnerd
    global runnerc
    global runnerb
    global runnera
    global hit1_counter
    
    if outcome1 == 'single':
        runnerf += 1
        runnere += 1
        runnerd += 1
        runnerc += 1
        runnerb += 1
        runnera += 1
        print('single! sixth runner goes to first')  
        hit1_counter = 'f'
        
    elif outcome1 == 'double':
        runnerf += 2
        runnere += 2
        runnerd += 2
        runnerc += 2
        runnerb += 2
        runnera += 2
        print('double! sixth runner goes to second') 
        hit1_counter = 'f'
        
    elif outcome1 == 'triple':
        runnerf += 3
        runnere += 3
        runnerd += 3
        runnerc += 3
        runnerb += 3
        runnera += 3
        print('triple! sixth runner goes to third')
        hit1_counter = 'f'
        
def runner7():
    global runnerg
    global runnerf
    global runnere
    global runnerd
    global runnerc
    global runnerb
    global runnera
    global hit1_counter
    
    if outcome1 == 'single':
        runnerg += 1
        runnerf += 1
        runnere += 1
        runnerd += 1
        runnerc += 1
        runnerb += 1
        runnera += 1
        print('single! seventh runner goes to first')  
        
    elif outcome1 == 'double':
        runnerg += 2
        runnerf += 2
        runnere += 2
        runnerd += 2
        runnerc += 2
        runnerb += 2
        runnera += 2
        print('double! seventh runner goes to second') 
        
    elif outcome1 == 'triple':
        runnerg += 3
        runnerf += 3
        runnere += 3
        runnerd += 3
        runnerc += 3
        runnerb += 3
        runnera += 3
        print('triple! seventh runner goes to third')
  
def scoreboard():
    global runs
    global signal
    global signalb
    global signalc
    global signald
    global signale
    global useless_counter
    
    if signal == 'x':
        useless_counter +=1
    
    elif runnera >= 4:
        signal = 'a'
    
    if signal == 'a':
        signal = 'x'
        runs += 1
        print('first runner scores! the team has now scored %s runs' % (runs))
        
    else:
        useless_counter +=1
        
    if signalb == 'x':
        useless_counter +=1
    
    elif runnerb >= 4:
        signalb = 'b'
    
    if signalb == 'b':
        signalb = 'x'
        runs += 1
        print('second runner scores! the team has now scored %s runs' % (runs))
        
    else:
        useless_counter +=1
        
    if signalc == 'x':
        useless_counter +=1
    
    elif runnerc >= 4:
        signalc = 'c'
    
    if signalc == 'c':
        signalc = 'x'
        runs += 1
        print('third runner scores! the team has now scored %s runs' % (runs))
        
    else:
        useless_counter +=1
        
    if signald == 'x':
        useless_counter +=1
    
    elif runnerd >= 4:
        signald = 'd'
    
    if signald == 'd':
        signald = 'x'
        runs += 1
        print('fourth runner scores! the team has now scored %s runs' % (runs))
        
    else:
        useless_counter +=1
        
    if signale == 'x':
        useless_counter +=1
    
    elif runnere >= 4:
        signale = 'e'
    
    if signale == 'e':
        signale = 'x'
        runs += 1
        print('fifth runner scores! the team has now scored %s runs' % (runs))
        
    else:
        useless_counter +=1
             
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
        scoreboard()
        
    elif hit1 in ('called strike', 'swinging strike', 'foul ball'):
        strike_recorded()       
        
    elif hit1 == 'ball':
        ball_recorded()
                   
def inning():   
    home_run_outcome()    
    
    if home_runs == 'home run':
        pitch_thrown()
    
    elif home_runs == 'not a home run':
        pitch_thrown()
        hitter_action()
        hit_outcomes()
    
      
while outs < 3:
    inning()   
    print
