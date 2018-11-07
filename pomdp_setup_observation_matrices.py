import json
import math
import numpy as np
from pomdp_setup_reward_transition_matrices import combine_states_to_one_list


def combine_obs_types_to_one_list (correctness_obs, speed_obs):
    combined_obs_list = []

    for correctness_observation in correctness_obs:
    	for speed_observation in speed_obs:
    		combined_obs_list.append(correctness_observation + "-" + speed_observation)

    return combined_obs_list

def generate_observation_matrix(knowledge_states, engagement_states, attempt_states, 
                                correctness_obs, speed_obs, num_actions, prob_speeds_for_low_engagement, 
                                prob_speeds_for_high_engagement, action_speed_multipliers):

    all_states = combine_states_to_one_list(knowledge_states, engagement_states, attempt_states)
    all_obs = combine_obs_types_to_one_list(correctness_obs, speed_obs)
    #num_actions = len(actions)
    num_states = len(all_states)
    num_engagement_levels = len(engagement_states)
    num_attempts = len(attempt_states)
    num_observations = len(correctness_obs) * len(speed_obs)
    num_states_per_knowledge_level = len(engagement_states) * len(attempt_states)

    output_observation_matrix = np.zeros((num_actions, num_states, num_observations))
    
    state_to_obs_matrix = np.zeros((num_states, num_observations))


    #RE-DO initial state_to_obs_matrix with new attempt state added in
    for i in range(num_states):
        end_state_knowledge_index = i / (num_engagement_levels * num_attempts)
        end_state_engagement_index = (i - (end_state_knowledge_index * num_engagement_levels * num_attempts)) / num_attempts
        end_state_attempt_index = i % num_attempts

        if end_state_attempt_index == 0: #in A0r
            prob_right = 1.0
            prob_wrong = 0.0
        elif end_state_attempt_index > 0: #in A0w, A1, or A2
            prob_right = 0.0
            prob_wrong = 1.0

        for j in range(num_observations):
            obs_index = j / len(speed_obs)
            if obs_index < 1:
                if j%len(speed_obs)==1:
                    state_to_obs_matrix[i][j] = prob_right #* .8 #med speed should always be the most likely
                else:
                    state_to_obs_matrix[i][j] = prob_right #* .1
            else:
                if j%len(speed_obs)==1:
                    state_to_obs_matrix[i][j] = prob_wrong #* .8
                else:
                    state_to_obs_matrix[i][j] = prob_wrong #* .1

    #print state_to_obs_matrix


    #amend the likelihood of different speeds based on engagement
    if len(engagement_states) > 1:
        #case where we have likely have 2 engagement states: low, high
        lower_engagement_multipliers_for_speed = prob_speeds_for_low_engagement #[1.2, 1.6, 1.2]
        higher_engagement_multipliers_for_speed = prob_speeds_for_high_engagement #[1.1, 1.8, 1.1]

        for i in range(num_states):
            knowledge_index = i / (num_engagement_levels * num_attempts)
            engagement_index = (i - knowledge_index * num_engagement_levels * num_attempts) / num_attempts
            #for j in range(len(engagement_states)):
                #if engagement_states[j] in all_states[i]:
            if engagement_index < len(engagement_states)/2:
                for k in range(num_observations):
                    state_to_obs_matrix[i][k] = state_to_obs_matrix[i][k] * lower_engagement_multipliers_for_speed[k%len(speed_obs)]
            else:
                for k in range(num_observations):
                    state_to_obs_matrix[i][k] = state_to_obs_matrix[i][k] * higher_engagement_multipliers_for_speed[k%len(speed_obs)]
            #break
            row_sum = np.sum(state_to_obs_matrix[i])
            state_to_obs_matrix[i] /= row_sum 
    #print state_to_obs_matrix

    #amend the likelihood of different speed observations based on the action
    for i in range(num_actions):
        action_specific_obs_matrix = np.copy(state_to_obs_matrix)
        for j in range(action_specific_obs_matrix.shape[0]):
            for k in range(num_observations):
                action_specific_obs_matrix[j][k] = action_specific_obs_matrix[j][k] * action_speed_multipliers[i][k%len(speed_obs)]
            row_sum = np.sum(action_specific_obs_matrix[j])
            action_specific_obs_matrix[j] /= row_sum

        output_observation_matrix[i] = action_specific_obs_matrix

    #print output_observation_matrix
    return output_observation_matrix
                 

def generate_sample_student_obs(all_states, observations, observation_matrix, state, current_action_index):
    #choose the matrix within the full observation_matrix that corresponds to the current action index
    obs_probs = observation_matrix[current_action_index]
    
    
    state_index = all_states.index(state)

    state_distribution = obs_probs[state_index]
    #print start_state_distribution

    num_observations = len(observations)
    obs_index = np.random.choice(np.arange(0, num_observations), p=state_distribution)
    obs = observations[obs_index]

    return obs



if __name__ == "__main__": 

    param_file = "control_student_logfiles/03_13_B.json"
    with open(param_file) as data_file:
            params = json.load(data_file)

    # state variables
    knowledge_states = params["knowledge_states"]
    engagement_states = params["engagement_states"]
    attempt_states = params["attempt_states"]
    all_states = combine_states_to_one_list(knowledge_states, engagement_states, attempt_states)

    correctness_obs = params["correctness_obs"]
    speed_obs = params["speed_obs"]
    all_obs = combine_obs_types_to_one_list(correctness_obs, speed_obs)

    # action variables
    actions = params["actions"]

    #probabilities associated with observation matrix
    prob_speeds_for_low_engagement = params["prob_speeds_for_low_engagement"]
    prob_speeds_for_high_engagement = params["prob_speeds_for_high_engagement"]
    action_speed_multipliers = np.array(params["action_speed_multipliers"])
    
    #for each action, use this multiplier for slow, med, fast
    #action_speed_multipliers = np.array(
    #    [[1.1, 1.8, 1.1], #no-action
    #    [1.2, 1.7, 1.1],  #interactive-tutorial
    #    [1.3, 1.6, 1.1],  #worked-example 
    #    [1.1, 1.6, 1.3],  #hint
    #    [1.1, 1.8, 1.1],  #think-aloud
    #    [1.1, 1.7, 1.2]]  #break
    #)

    O = generate_observation_matrix(knowledge_states=knowledge_states, 
                                    engagement_states=engagement_states,
                                    attempt_states=attempt_states,
                                    correctness_obs=correctness_obs,
                                    speed_obs=speed_obs,
                                    num_actions=len(actions),
                                    prob_speeds_for_low_engagement=prob_speeds_for_low_engagement,
                                    prob_speeds_for_high_engagement=prob_speeds_for_high_engagement,
                                    action_speed_multipliers=action_speed_multipliers)
    
    print O
    
    # sample_observation = generate_sample_student_obs(all_states=all_states,
    #                                                 observations=all_obs,
    #                                                 observation_matrix=O,
    #                                                 state="K1-E0-A1",
    #                                                 current_action_index=5)
    #print sample_observation



