# Citation: http://iacs-courses.seas.harvard.edu/courses/am207/blog/lecture-18.html

# Library imports
import pandas as pd
import random
import csv


def run_pitching_markov_chain():

    # Read in data
    df = pd.read_csv('scherzer_pitches.csv')

    # Change pitch names
    df.rename(columns={'15': 'pitch'}, inplace=True)

    pitch_dict = {'CH': 'CH', 'CU': 'CU', 'FA': 'OT',
                  'FC': 'OT', 'FF': 'FF', 'FT': 'OT',
                  'IN': 'OT', 'PO': 'OT', 'SL': 'SL',
                  'UN': 'OT'}

    df['pitch'] = df['pitch'].map(pitch_dict)

    # Transition Matrix
    transitions = {}
    row_sums = {}

    for line in open('scherzer_pitch_sequences.csv'):
        s, e = line.rstrip().split(',')
        transitions[(s, e)] = transitions.get((s, e), 0.) + 1
        row_sums[s] = row_sums.get(s, 0.) + 1

    for k, v in transitions.iteritems():
        s, e = k
        transitions[k] = v / row_sums[s]

    with open('scherzer_transitions.csv', 'wb') as f:
        w = csv.DictWriter(f, transitions.keys())
        w.writeheader()
        w.writerow(transitions)

    # Emission probability calculations
    def calculate_emission_probabilities(df):
        df.rename(columns={'19': 'strikes'}, inplace=True)
        df.rename(columns={'20': 'balls'}, inplace=True)

        df['strikes'] = df['strikes'].astype('str')
        df['balls'] = df['balls'].astype('str')
        df['count'] = df['balls'] + '-' + df['strikes']

        pitch_totals = df['pitch'].groupby(df['pitch']).count()
        pitch_totals = pd.DataFrame(pitch_totals)
        pitch_totals.rename(columns={'pitch': 'pitch_total'}, inplace=True)
        pitch_totals.reset_index(inplace = True)

        pitches_in_counts = df['pitch'].groupby([df['count'], df['pitch']]).count()
        pitches_in_counts = pd.DataFrame(pitches_in_counts)
        pitches_in_counts.rename(columns={'pitch': 'pitch_situations'}, inplace=True)
        pitches_in_counts.reset_index(inplace = True)

        pitches_in_counts = pd.merge(pitches_in_counts, pitch_totals,
                                     how = 'inner', on = 'pitch')

        pitches_in_counts['pitch_percentage'] = pitches_in_counts['pitch_situations'] /\
        pitches_in_counts['pitch_total']

        return pitches_in_counts

    pitches_in_counts = calculate_emission_probabilities(df)

    count_dict = {'0.0-0.0': "'0-0'", '0.0-1.0': "'0-1'", '0.0-2.0': "'0-2'", '1.0-0.0': "'1-0'",
                     '1.0-1.0': "'1-1'", '1.0-2.0': "'1-2'", '2.0-0.0': "'2-0'", '2.0-1.0': "'2-1'",
                     '2.0-2.0': "'2-2'", '3.0-0.0': "'3-0'", '3.0-1.0': "'3-1'", '3.0-2.0': "'3-2'"}

    pitches_in_counts['count'] = pitches_in_counts['count'].map(count_dict)
    pitches_in_counts.to_csv('pitches_in_counts.csv', index = False)

    # Set up states and probabilities
    states = ('Fourseam', 'Change', 'Slider', 'Curve', 'Other')

    observations = ('0-0', '0-1', '0-2', '1-0', '1-1', '1-2', '2-0', '2-1',
                    '2-2', '3-0', '3-1', '3-2')

    start_probability = {'Fourseam': 0.50, 'Change': 0.20, 'Slider': 0.20,
                         'Curve': 0.05, 'Other': 0.05}

    transition_probability = {
       'Fourseam' : {'Fourseam': 0.57, 'Change': 0.19, 'Slider': 0.17, 
                     'Curve': 0.05, 'Other': 0.02},

       'Change' : {'Fourseam': 0.61, 'Change': 0.21, 'Slider': 0.12, 
                   'Curve': 0.04, 'Other': 0.02},

       'Slider' : {'Fourseam': 0.58, 'Change': 0.09, 'Slider': 0.27,
                   'Curve': 0.03, 'Other': 0.03},

       'Curve' : {'Fourseam': 0.61, 'Change': 0.21, 'Slider': 0.09,
                  'Curve': 0.08, 'Other': 0.01},

       'Other' : {'Fourseam': 0.34, 'Change': 0.18, 'Slider': 0.15, 
                  'Curve': 0.27, 'Other': 0.06},
       }

    emission_probability = {
       'Fourseam' : {'0-0': 0.27, '0-1': 0.13, '0-2': 0.07, '1-0': 0.09,
                     '1-1': 0.10, '1-2': 0.10, '2-0': 0.04, '2-1': 0.05,
                     '2-2': 0.08, '3-0': 0.01, '3-1': 0.02, '3-2': 0.05},

       'Change' : {'0-0': 0.17, '0-1': 0.12, '0-2': 0.07, '1-0': 0.11,
                     '1-1': 0.12, '1-2': 0.15, '2-0': 0.02, '2-1': 0.05,
                     '2-2': 0.13, '3-0': 0.0, '3-1': 0.0, '3-2': 0.05}, 

       'Slider' : {'0-0': 0.25, '0-1': 0.16, '0-2': 0.10, '1-0': 0.08,
                     '1-1': 0.10, '1-2': 0.14, '2-0': 0.0, '2-1': 0.03,
                     '2-2': 0.10, '3-0': 0.0, '3-1': 0.0, '3-2': 0.03},

       'Curve' : {'0-0': 0.34, '0-1': 0.18, '0-2': 0.10, '1-0': 0.05,
                     '1-1': 0.10, '1-2': 0.12, '2-0': 0.0, '2-1': 0.01,
                     '2-2': 0.09, '3-0': 0.0, '3-1': 0.0, '3-2': 0.02},

       'Other' : {'0-0': 0.25, '0-1': 0.14, '0-2': 0.07, '1-0': 0.10,
                     '1-1': 0.10, '1-2': 0.09, '2-0': 0.03, '2-1': 0.05,
                     '2-2': 0.08, '3-0': 0.02, '3-1': 0.02, '3-2': 0.04}
       }

    # A HMM is created from the above matices for 100 of Scherzer's pitches
    # Both hidden and visible states are generated
    N = 100
    hidden = []
    visible = []
    
    if random.random() < start_probability[states[0]]:
        hidden.append(states[0])
    else:
        hidden.append(states[1])

    for i in xrange(N):
        current_state = hidden[i]
        if random.random() < transition_probability[current_state][states[0]]:
            hidden.append(states[0])
        else:
            hidden.append(states[1])
        r = random.random()
        prev = 0
        for observation in observations:
            prev += emission_probability[current_state][observation]
            if r < prev:
                visible.append(observation)
                break
                
    hidden.pop()

    # Run the Viterbi algorithm
    def viterbi(obs, states, start_p, trans_p, emit_p):
        V = [{}]
        path = {}

        for y in states:
            V[0][y] = start_p[y] * emit_p[y][obs[0]]
            path[y] = [y]

        for t in range(1, len(obs)):
            V.append({})
            newpath = {}

            for y in states:
                (prob, state) = max((V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states)
                V[t][y] = prob
                newpath[y] = path[state] + [y]

            path = newpath

        (prob, state) = max((V[t][y], y) for y in states)
        return (prob, path[state])

    # Input the generated markov model
    def example_model():
        return viterbi(visible,
                       states,
                       start_probability,
                       transition_probability,
                       emission_probability)

    (prob, p_hidden) = example_model()

    # Assess accuracy of the model
    wrong= 0
    for i in range(len(hidden)):
        if hidden[i] != p_hidden[i]:
            wrong = wrong + 1
    print "accuracy: " + str(1-float(wrong)/N)
    return
  
  
 if __name__ == "__main__":
    run_pitching_markov_chain()
