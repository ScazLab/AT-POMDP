import numpy as np
from pomdp_setup_reward_transition_matrices import combine_states_to_one_list


def combine_obs_types_to_one_list(correctness_obs, speed_obs):
    combined_obs_list = []

    for correctness_observation in correctness_obs:
        for speed_observation in speed_obs:
            combined_obs_list.append(correctness_observation + "-" + speed_observation)

    return combined_obs_list

def generate_observation_matrix(knowledge_states, engagement_states, attempt_states,
                                correctness_obs, speed_obs, num_actions,
                                prob_speeds_for_low_engagement, prob_speeds_for_high_engagement,
                                action_speed_multipliers):

    all_states = combine_states_to_one_list(knowledge_states, engagement_states, attempt_states)
    num_states = len(all_states)
    num_engagement_levels = len(engagement_states)
    num_attempts = len(attempt_states)
    num_observations = len(correctness_obs) * len(speed_obs)

    output_observation_matrix = np.zeros((num_actions, num_states, num_observations))

    state_to_obs_matrix = np.zeros((num_states, num_observations))

    for i in range(num_states):
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
                if j%len(speed_obs) == 1:
                    state_to_obs_matrix[i][j] = prob_right #* .8 #med speed is the most likely
                else:
                    state_to_obs_matrix[i][j] = prob_right #* .1
            else:
                if j%len(speed_obs) == 1:
                    state_to_obs_matrix[i][j] = prob_wrong #* .8
                else:
                    state_to_obs_matrix[i][j] = prob_wrong #* .1


    #amend the likelihood of different speeds based on engagement
    if len(engagement_states) > 1:
        #case where we have 2 engagement states: low, high
        lower_engagement_multipliers_for_speed = prob_speeds_for_low_engagement
        higher_engagement_multipliers_for_speed = prob_speeds_for_high_engagement

        for i in range(num_states):
            knowledge_index = i / (num_engagement_levels * num_attempts)
            engagement_index = (i - knowledge_index * num_engagement_levels * num_attempts) / num_attempts

            if engagement_index < len(engagement_states)/2:
                for k in range(num_observations):
                    state_to_obs_matrix[i][k] = state_to_obs_matrix[i][k] * lower_engagement_multipliers_for_speed[k%len(speed_obs)]
            else:
                for k in range(num_observations):
                    state_to_obs_matrix[i][k] = state_to_obs_matrix[i][k] * higher_engagement_multipliers_for_speed[k%len(speed_obs)]

            row_sum = np.sum(state_to_obs_matrix[i])
            state_to_obs_matrix[i] /= row_sum

    #amend the likelihood of different speed observations based on the action
    for i in range(num_actions):
        action_specific_obs_matrix = np.copy(state_to_obs_matrix)
        for j in range(action_specific_obs_matrix.shape[0]):
            for k in range(num_observations):
                action_specific_obs_matrix[j][k] = action_specific_obs_matrix[j][k] * action_speed_multipliers[i][k%len(speed_obs)]
            row_sum = np.sum(action_specific_obs_matrix[j])
            action_specific_obs_matrix[j] /= row_sum

        output_observation_matrix[i] = action_specific_obs_matrix

    return output_observation_matrix
