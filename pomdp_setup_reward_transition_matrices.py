import json
import math
import numpy as np
np.set_printoptions(threshold=np.nan, precision=4, suppress=True)


def combine_states_to_one_list (knowledge_states, engagement_states, attempt_states):

    combined_state_list = []

    for knowledge_state in knowledge_states:
        num_states_for_each_knowledge_state = len(engagement_states) * len(attempt_states)
        for i in range(num_states_for_each_knowledge_state):
            combined_state_list.append(knowledge_state)

    for i in range(len(engagement_states)):
        for j in range(len(knowledge_states)):
            for k in range(len(attempt_states)):
                idx_engagement = (j * len(engagement_states) * len(attempt_states) + 
                                  i * len(attempt_states) + k)
                combined_state_list[idx_engagement] += "-" + engagement_states[i]
                combined_state_list[idx_engagement] += "-" + attempt_states[k]

    return combined_state_list


def generate_reward_matrix (actions, action_rewards, engagement_reward, knowledge_reward, 
                            end_state_remain_reward, num_knowledge_levels, num_engagement_levels, 
                            num_attempts, num_observations, reward_for_first_attempt_actions):

    num_states = num_knowledge_levels * num_engagement_levels * num_attempts
    num_actions = len(action_rewards)

    output_reward_matrix = np.zeros((num_actions, num_states, num_states, num_observations))
    reward_independent_of_action = np.zeros((num_states, num_states, num_observations))

    for i in range(num_states):
        start_knowledge = i / (num_engagement_levels * num_attempts)
        start_engagement = (i - start_knowledge * num_engagement_levels * num_attempts) / num_attempts
        start_attempt = i % num_attempts

        for j in range(num_states):
            end_knowledge = j / (num_engagement_levels * num_attempts)
            end_engagement = (j - end_knowledge * num_engagement_levels * num_attempts) / num_attempts
            end_attempt = j % num_attempts

            # reward knowledge gains
            if (end_knowledge > start_knowledge):
                reward_independent_of_action[i][j] += knowledge_reward * (end_knowledge - start_knowledge)
                # print "K%i-E%i-A%i to K%i-E%i-A%i K+ reward: %i" % (start_knowledge, start_engagement, start_attempt, end_knowledge, end_engagement, end_attempt, knowledge_reward * (end_knowledge - start_knowledge))

            # reward engagement gains / losses
            if (end_engagement > start_engagement):
                reward_independent_of_action[i][j] += engagement_reward * (end_engagement - start_engagement)
            elif (end_engagement < start_engagement):
                reward_independent_of_action[i][j] -= engagement_reward * (start_engagement - end_engagement)
                # print "K%i-E%i-A%i to K%i-E%i-A%i E+ reward: %i" % (start_knowledge, start_engagement, start_attempt, end_knowledge, end_engagement, end_attempt, engagement_reward * (end_engagement - start_engagement))

    
            # reward the participant in the end state (highest knowledge, highest engagement)
            if (start_knowledge == end_knowledge and start_knowledge == (num_knowledge_levels - 1) 
                and start_engagement == end_engagement and start_engagement == (num_engagement_levels - 1)):

                reward_independent_of_action[i][j] += end_state_remain_reward

    # find the action index of the "no-action"
    no_action_idx = actions.index("no-action")

    # rewards of the specific actions
    for i in range(len(action_rewards)):
        output_reward_matrix[i] = reward_independent_of_action + action_rewards[i]

        # if the action is not "no-action", then heavily penalize 
        if (i != no_action_idx):
            for j in range(num_states):
                attempt_num = j % num_attempts

                # only for the first attempt
                if (attempt_num == 0 or attempt_num == 1):
                    output_reward_matrix[i][j] += reward_for_first_attempt_actions


    return output_reward_matrix


def generate_transition_matrix (num_knowledge_levels, num_engagement_levels, num_attempts,
                                prob_knowledge_gain, prob_engagement_gain, 
                                prob_engagement_loss, action_prob_knowledge_gain_mult, 
                                action_prob_engagement_gain_mult, prob_correct_answer,
                                prob_correct_answer_after_1_attempt, prob_drop_for_low_engagement):

    # set up the output transition matrix
    num_actions = len(action_prob_knowledge_gain_mult)
    num_states = num_knowledge_levels * num_engagement_levels * num_attempts
    output_transition_matrix = np.zeros((num_actions, num_states, num_states))

    for i in range(num_actions):
        # go through each possible transition from state to state
        for j in range(num_states):
            # start state parameters
            start_knowledge = j / (num_engagement_levels * num_attempts)
            start_engagement = (j - start_knowledge * num_engagement_levels * num_attempts) / num_attempts
            start_attempt = j % num_attempts

            for k in range(num_states):
                # end state parameters
                end_knowledge = k / (num_engagement_levels * num_attempts)
                end_engagement = (k - end_knowledge * num_engagement_levels * num_attempts) / num_attempts
                end_attempt = k % num_attempts    

                # start - end state change variables
                change_in_knowledge_level = end_knowledge - start_knowledge
                change_in_engagement_level = end_engagement - start_engagement

                # define, given the start state, the probability of changing knowledge state, or 
                # not changing knowledge state
                prob_knowledge_gain_for_start_egmt = (prob_knowledge_gain[start_engagement] * action_prob_knowledge_gain_mult[i])
                if (start_knowledge == (num_knowledge_levels - 1)):
                    prob_knowledge_no_change = 1.0
                else: 
                    prob_knowledge_no_change = 1.0 - prob_knowledge_gain_for_start_egmt

                # given that the inputs to the fuction give us the probabilities of engagement gain and
                # loss, we set the probability that there will be no change in engagement 
                prob_engagement_gain_for_start_k = (prob_engagement_gain[start_knowledge] * action_prob_engagement_gain_mult[i])
                prob_engagement_loss_for_start_k = prob_engagement_loss[start_knowledge]
                if (start_engagement == 0):
                    prob_engagement_no_change = 1.0 - prob_engagement_gain_for_start_k
                elif (start_engagement == 1):
                    prob_engagement_no_change = 1.0 - prob_engagement_loss_for_start_k
                else:
                    print "Transition matrix function error: engagement should be either 1 or 0"


                if (start_engagement == 0):
                    correct_prob_mult_for_engagement = prob_drop_for_low_engagement
                else: 
                    correct_prob_mult_for_engagement = 1.0

                # if the engagement and knowledge don't change, the proability of start_attempt going
                # to end_attempt (encoding logic of transitions between attempts)
                #
                #        A0r     A0w     A1      A2
                #   A0r  0.80    0       0.20    0
                #   A0w  0.80    0       0.20    0
                #   A1   0.70    0       0       0.30
                #   A2   0.60    0.40    0       0
                #       

                prob_attempt_change_indep = 0.0

                # A0r --> A0r
                if (start_attempt == 0 and end_attempt == 0):
                    prob_attempt_change_indep = prob_correct_answer[start_knowledge] * correct_prob_mult_for_engagement
                # A0r --> A1
                elif (start_attempt == 0 and end_attempt == 2): 
                    prob_attempt_change_indep = 1.0 - prob_correct_answer[start_knowledge] * correct_prob_mult_for_engagement
                # A0w --> A0r
                elif (start_attempt == 1 and end_attempt == 0):
                    prob_attempt_change_indep = prob_correct_answer[start_knowledge] * correct_prob_mult_for_engagement
                # A0w --> A1
                elif (start_attempt == 1 and end_attempt == 2): 
                    prob_attempt_change_indep = 1.0 - prob_correct_answer[start_knowledge] * correct_prob_mult_for_engagement
                # A1 --> A0r
                elif (start_attempt == 2 and end_attempt == 0):
                    prob_attempt_change_indep = prob_correct_answer_after_1_attempt[start_knowledge] * correct_prob_mult_for_engagement
                # A1 --> A2
                elif (start_attempt == 2 and end_attempt == 3):
                    prob_attempt_change_indep = 1.0 - prob_correct_answer_after_1_attempt[start_knowledge] * correct_prob_mult_for_engagement
                # A2 --> A0r
                elif (start_attempt == 3 and end_attempt == 0):
                    prob_attempt_change_indep = prob_correct_answer_after_1_attempt[start_knowledge] * correct_prob_mult_for_engagement
                # A2 --> A0w
                elif (start_attempt == 3 and end_attempt == 1):
                    prob_attempt_change_indep = 1 - (prob_correct_answer_after_1_attempt[start_knowledge] * correct_prob_mult_for_engagement)

                start_to_end_state_trans_prob = 0.0

                # if the knowledge level change is 0 or positive (we don't allow losses in knowledge)
                # and if our attempt transition is viable
                if (change_in_knowledge_level >= 0 and prob_attempt_change_indep > 0.0):
                    # if there's a positive change in engagement and knowledge
                    if (change_in_engagement_level > 0 and change_in_knowledge_level > 0):
                        start_to_end_state_trans_prob = (
                            math.pow(prob_engagement_gain_for_start_k, change_in_engagement_level) * 
                            math.pow(prob_knowledge_gain_for_start_egmt, change_in_knowledge_level) * 
                            prob_attempt_change_indep
                        )
                    # if there's a positive change in engagement and no change in knowledge
                    elif (change_in_engagement_level > 0 and change_in_knowledge_level == 0):
                        start_to_end_state_trans_prob = (
                            math.pow(prob_engagement_gain_for_start_k, change_in_engagement_level) * 
                            prob_knowledge_no_change * 
                            prob_attempt_change_indep
                        )
                    # if there's a negative change in engagement and a positive change in knowledge
                    elif (change_in_engagement_level < 0 and change_in_knowledge_level > 0):
                        start_to_end_state_trans_prob = (
                            math.pow(prob_engagement_loss_for_start_k, -change_in_engagement_level) * 
                            math.pow(prob_knowledge_gain_for_start_egmt, change_in_knowledge_level) * 
                            prob_attempt_change_indep
                        )
                    # if there's a negative change in engagement and no change in knowledge
                    elif (change_in_engagement_level < 0 and change_in_knowledge_level == 0):
                        start_to_end_state_trans_prob = (
                            math.pow(prob_engagement_loss_for_start_k, -change_in_engagement_level) * 
                            prob_knowledge_no_change * 
                            prob_attempt_change_indep
                        )
                    # if there's no change in engagement and a positive cahnge in knowledge
                    elif (change_in_engagement_level == 0 and change_in_knowledge_level > 0):
                        start_to_end_state_trans_prob = (
                            prob_engagement_no_change * 
                            math.pow(prob_knowledge_gain_for_start_egmt, change_in_knowledge_level) * 
                            prob_attempt_change_indep
                        )
                    # if there's no change in either engagement or knowledge
                    elif (change_in_engagement_level == 0 and change_in_knowledge_level == 0):
                        start_to_end_state_trans_prob = (
                            prob_engagement_no_change *
                            prob_knowledge_no_change *
                            prob_attempt_change_indep
                        )

                output_transition_matrix[i][j][k] = start_to_end_state_trans_prob

            row_sum = np.sum(output_transition_matrix[i][j])
            output_transition_matrix[i][j] /= row_sum

        # # now for each action, factor in the increased likelihood of improving knowledge and 
        # # engagement
        # for j in range(num_states):
        #     start_knowledge = j / (num_engagement_levels * num_attempts)
        #     start_engagement = (j - start_knowledge * num_engagement_levels * num_attempts) / num_attempts
        #     for k in range(num_states):
        #         end_knowledge = k / (num_engagement_levels * num_attempts)
        #         end_engagement = (k - end_knowledge * num_engagement_levels * num_attempts) / num_attempts

        #         # start - end state change variables
        #         change_in_knowledge_level = end_knowledge - start_knowledge
        #         change_in_engagement_level = end_engagement - start_engagement

        #         # based on the specific action, increase the likelihood of gains of engagement and
        #         # knowledge
        #         if (change_in_knowledge_level > 0):
        #             output_transition_matrix[i][j][k] *= math.pow(action_prob_knowledge_gain_mult[i], 
        #                 change_in_knowledge_level)

        #         if (change_in_engagement_level > 0):
        #             output_transition_matrix[i][j][k] *= math.pow(action_prob_engagement_gain_mult[i], 
        #                 change_in_engagement_level)

        #     # Re-normalize the rows of the matrices
        #     row_sum = np.sum(output_transition_matrix[i][j])
        #     output_transition_matrix[i][j] /= row_sum


    return output_transition_matrix











