import sys
import json
import numpy as np
from task_models.lib.pomdp import POMDP, GraphPolicyBeliefRunner
from pomdp_setup_reward_transition_matrices import *
from pomdp_setup_observation_matrices import *

def test_command_line_sequence(param_file):
    #read in params
    with open(param_file) as data_file:
       params = json.load(data_file)

    # discount factor
    discount = params["discount"]

    # state variables
    knowledge_states = params["knowledge_states"]
    engagement_states = params["engagement_states"]
    attempt_states = params["attempt_states"]
    num_knowledge_levels = len(knowledge_states)
    num_engagement_levels = len(engagement_states)
    num_attempts = len(attempt_states)
    all_states = combine_states_to_one_list(knowledge_states, engagement_states, attempt_states)
    num_states = len(all_states)

    # starting distribution 
    start = np.zeros(num_states)
    # start[4] = 1.0
    num_start_states = num_knowledge_levels * num_engagement_levels
    for i in range(num_states):
        if i%num_attempts==0:
            start[i] = 1.0 / float(num_start_states)
        else:
            start[i] = 0.0

    #for i in range(len(all_states)):
    #    print all_states[i]
    #    print start[i]
    #    print

    # probabilities associated with the transition matrix
    prob_knowledge_gain = params["prob_knowledge_gain"]
    prob_engagement_gain = params["prob_engagement_gain"]
    prob_engagement_loss = params["prob_engagement_loss"]
    prob_correct_answer = params["prob_correct_answer"]
    prob_correct_answer_after_1_attempt = params["prob_correct_answer_after_1_attempt"]
    prob_drop_for_low_engagement = params["prob_drop_for_low_engagement"]

    # actions
    actions = params["actions"]
    num_actions = len(actions)

    # action-related reward variables
    action_rewards = params["action_rewards"]
    engagement_reward = params["engagement_reward"]
    knowledge_reward = params["knowledge_reward"]
    end_state_remain_reward = params["end_state_remain_reward"]
    reward_for_first_attempt_actions = params["reward_for_first_attempt_actions"]
    action_prob_knowledge_gain_mult = params["action_prob_knowledge_gain_mult"]
    action_prob_engagement_gain_mult = params["action_prob_engagement_gain_mult"]

    # observations
    correctness_obs = params["correctness_obs"]
    speed_obs = params["speed_obs"]
    all_obs = combine_obs_types_to_one_list(correctness_obs, speed_obs)
    num_observations = len(all_obs)

    # observation related variables
    prob_speeds_for_low_engagement = params["prob_speeds_for_low_engagement"]
    prob_speeds_for_high_engagement = params["prob_speeds_for_high_engagement"]
    action_speed_multipliers = np.array(params["action_speed_multipliers"])


    R = generate_reward_matrix(actions=actions,
                               action_rewards=action_rewards, 
                               engagement_reward=engagement_reward, 
                               knowledge_reward=knowledge_reward, 
                               end_state_remain_reward=end_state_remain_reward,
                               num_knowledge_levels=num_knowledge_levels, 
                               num_engagement_levels=num_engagement_levels,
                               num_attempts=num_attempts, 
                               num_observations=num_observations, 
                               reward_for_first_attempt_actions=reward_for_first_attempt_actions)

    T = generate_transition_matrix(num_knowledge_levels=num_knowledge_levels, 
                                   num_engagement_levels=num_engagement_levels,
                                   num_attempts=num_attempts,
                                   prob_knowledge_gain=prob_knowledge_gain,
                                   prob_engagement_gain=prob_engagement_gain,
                                   prob_engagement_loss=prob_engagement_loss,
                                   action_prob_knowledge_gain_mult=action_prob_knowledge_gain_mult,
                                   action_prob_engagement_gain_mult=action_prob_engagement_gain_mult,
                                   prob_correct_answer=prob_correct_answer,
                                   prob_correct_answer_after_1_attempt=prob_correct_answer_after_1_attempt, 
                                   prob_drop_for_low_engagement=prob_drop_for_low_engagement)

    O = generate_observation_matrix(knowledge_states=knowledge_states, 
                                    engagement_states=engagement_states,
                                    attempt_states=attempt_states,
                                    correctness_obs=correctness_obs,
                                    speed_obs=speed_obs,
                                    num_actions=num_actions,
                                    prob_speeds_for_low_engagement=prob_speeds_for_low_engagement,
                                    prob_speeds_for_high_engagement=prob_speeds_for_high_engagement,
                                    action_speed_multipliers=action_speed_multipliers)


    # print transition matrix for no action
    #for i in range(len(T[1])):
    #    print all_states[i]
    #    print T[1][i]
    #    print

    # print observation matrix
    #for i in range(len(O[0])):
    #    print all_states[i]
    #    print O[0][i]
    #    print 

    #create POMDP model
    simple_pomdp = POMDP(T, O, R, np.array(start), discount, states=all_states, actions=actions,
                 observations=all_obs, values='reward')

    simple_pomdp_graph_policy = simple_pomdp.solve(method='grid', verbose=False, n_iterations=500)

    simple_pomdp_graph_policy_belief_runner = GraphPolicyBeliefRunner(simple_pomdp_graph_policy,
                                                                  simple_pomdp)



    knowledge_level_belief = np.zeros(num_knowledge_levels)
    num_states_per_knowledge_level = num_engagement_levels * num_attempts
    problem_num = 1
    attempt_num = 1
    receiving_obs = True
    while receiving_obs is True:
        obs = raw_input("Enter observation: ")
        if obs == "done":
            receiving_obs = False
            break
        if obs not in all_obs:
            print "Invalid observation provided\n"
            continue
        knowledge_level_index = 0
        action = simple_pomdp_graph_policy_belief_runner.get_action()
        current_belief = simple_pomdp_graph_policy_belief_runner.step(obs, action)
        print "\nProblem %i, Attempt %i: (%s, %s)" % (problem_num, attempt_num, action, obs)
        
        belief_str = ""
        sum_across_states = 0.0
        for k in range(num_states):
            sum_across_states += current_belief[k]
            if k % num_attempts == num_attempts - 1:
                belief_str += "%s: %.3f\t\t" % (all_states[k][:-3], sum_across_states)
                #knowledge_level_belief[knowledge_level_index] = sum_across_states
                knowledge_level_index += 1
                sum_across_states = 0.0
            if k % num_states_per_knowledge_level == num_states_per_knowledge_level-1:
                belief_str += "\n"

        print belief_str

        if "R" in obs or attempt_num==3:
            problem_num += 1
            attempt_num = 1
        else:
            attempt_num += 1


if __name__ == "__main__":
    if len(sys.argv)>1:
        test_command_line_sequence(sys.argv[1])
    else:
        print "please provide the name of the input parameter file as a command line argument"
