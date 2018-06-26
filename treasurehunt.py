import numpy as np
import pandas as pd
import time

np.random.seed(2)  # reproducible


N_STATES = 25   # the length of the 1 dimensional world
ACTIONS = ['left', 'right','up', 'down']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 13   # maximum episodes
FRESH_TIME = 0.3    # fresh time for one move


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    # print(table)    # show table
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()    
    return action_name


def get_env_feedback(S, A):
    if A == 'right':  # move right
        if S + 1 == 24:  # terminate
            S_ = 'terminal'
            R = 10
        elif ((S + 1) % 5 == 0) and (S != 4 or S != 9 ):  # hit wall
            S_ = S
            R = -1
        elif S == 2 or S == 10 or S == 15 or S == 22:
            S_ = S + 1
            R = -10
        elif S == 4 or S == 9:
            S_ = S
            R=-10
        else:
            S_ = S + 1
            R = -1
    elif A == 'left':  # move left
        if S % 5 == 0 and ( S != 10 or S != 15 ):  # hit wall
            S_ = S  # reach the wall
            R = -1
        elif S == 2 or S == 4 or S == 9 or S == 22:
            S_ = S - 1
            R = -10
        elif S == 10 or S ==15:
            S_=S
            R=-10
        else:
            S_ = S - 1
            R = -1
    elif A == 'up':
        if S > 4 and (S!=9 or S!=10 or S !=15 or S !=22):
            S_ = S - 5
            R = -1
        elif S == 9 or S == 10 or S == 15 or S == 22:
            S_ = S - 5
            R = -10
        elif S==2 or S==4:
            S_=S
            R=-10
        else:
            S_ = S  # when S lesser than 5, it cannot go up (invalid move)
            R = -1
    else:  # A==down
        if S + 5 == 24:
            S_ = 'terminal'
            R = 10
        elif S == 2 or S == 4 or S == 9 or S == 10 or S == 15:
            S_ = S + 5
            R = -10
        elif S==22:
            S_=S
            R=-10
        elif S >= 20 and (S!=22):
            S_ = S
            R = -1
        else:
            S_ = S + 5
            R = -1
    return S_, R


def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['S', '-', 'W', '-', 'W',
                '-', '-', '-', '-', 'W',
                'W', '-', '-', '-', '-',
                'W', '-', '-', '-', '-',
                '-', '-', 'W', '-', 'G']  # '---------T' our environment

    if S == 'terminal':
        env_list[24] = '0'
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = '0'
        for i in range(5):
            for j in range(5):
                print(env_list[j+5*i],end='')
            print()
        time.sleep(FRESH_TIME)


def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:

            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A) # take action & get next state and reward
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next state

            update_env(S, episode, step_counter+1)
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
