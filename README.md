# AT-POMDP

The Assistive Tutor POMDP (AT-POMDP) developed by Aditi Ramachandran and Sarah Strohkorb Sebo, was designed to select appropriate tutoring actions to support 5th grade students in a one-on-one tutoring interaction with a Nao robot on the topic of long-division mathemtaics. 

If you are using this software or one of its components, we recommend that you cite the paper in which we present the AT-POMDP and its evaluation with 5th grade students:

> Aditi Ramachandran, Sarah Strohkorb Sebo, Brian Scassellati (2019). Personalized Robot Tutoring using the Assistive Tutor POMDP (AT-POMDP). In *Proceedings of The Thirty-Third AAAI Conference on Artificial Intelligence (AAAI 2019)*. Honolulu, HI, USA. [PDF](https://scazlab.yale.edu/sites/default/files/files/Ramachandran_Sebo_AAAI_2018.pdf)

## Prerequisites

## Usage

## File-by-File Details

## Computing Model Parameters from the JSON File

This subsection describes all of the variables in the parameter file ([atpomdp_params.json](https://github.com/ScazLab/AT-POMDP/blob/master/atpomdp_params.json)) and how they combine to compute the POMDP model parameters. 

### State Space, Action Space, and Observation Space

The state space has 3 dimensions: knowledge level (`"knowledge_states"`), engagement level (`"engagement_states"`), and attempt number (`"attempt_states"`). Since there are 4 knowledge levels, 2 engagement states (low and high), and 4 attempt states (attempt 1 after getting a problem right, attempt 1 after getting a problem right, attempt 2, attempt 3) there are a total of 4 x 2 x 4 = 32 possible states.

The action space consists of 6 possible tutoring actions `"actions": ["no-action", "interactive-tutorial", "worked-example", "hint", "think-aloud", "break"]`.

The observation space has 2 dimensions: attempt correctness (`"correctness_obs"`) - right or wrong, and attempt speed (`"speed_obs"`) - slow, medium, or fast. The observation space has a size of 3 x 2 = 6. 

### Computing the Reward Matrix

The reward function is represnted by a matrix with 4 dimensions: start state (*s*), end state (*s'*), action, and observation. With this matrix representation we can reward transitions between states. We compute the reward matrix in the `generate_reward_matrix()` function in [pomdp_setup_reward_transition_matrices.py](https://github.com/ScazLab/AT-POMDP/blob/master/pomdp_setup_reward_transition_matrices.py). 

There are 5 parameters in our JSON file that influence the computation of this reward matrix: 
- `"reward_for_first_attempt_actions"`: If the start state is one of the attempt 1 states, we set the reward of taking any action other than "no-action" as -1000 to ensure that our POMDP never chooses any action other than "no-action" for the first attempt. This allows the 5th grade student to make a first attempt on a math problem before receiving help from the system. 
- `"engagement_reward"`: Any time engagement state changes from the start state to the end state, the engagement reward is applied to that transition, this applies for both gains and losses in engagement. 
- `"knowledge_reward"`: Any time engagement state changes from the start state to the end state, the engagement reward is applied to that transition, this applies only to gains in knowledge, since our model doesn't allow loss of knowledge (transitions from higher to lower knowledge states). 
- `"action_rewards"`: Each action has a certain cost associated with it. This cost is derived from how long each help action takes for the student and the robot to complete. 
- `"end_state_remain_reward"`: This reward value is the one given for states with the highest possible engagement and knowledge. Once students have reached the highest possible engagement and knowledge, we do not reward them further. 

### Computing the Transition Matrix

The transition function is represnted by a matrix with 3 dimensions: start state (*t*), end state (*t+1*), and action. We compute the transition matrix in the `generate_transition_matrix()` function in [pomdp_setup_reward_transition_matrices.py](https://github.com/ScazLab/AT-POMDP/blob/master/pomdp_setup_reward_transition_matrices.py). 

Each transition from the start state to end state taking a specified action is calculated according to the following formula: Pr(*s'*|*s*,*a*) = Pr(*s<sub>k</sub>*'|*s*,*a*)) x Pr(*s<sub>e</sub>*'|*s*,*a*)) x Pr(*s<sub>a</sub>*'|*s*,*a*)), where *s<sub>k</sub>* represents the knowledge dimension of the state, *s<sub>e</sub>* represents the engagement dimension of the state, and *s<sub>a</sub>* represents the attempt dimension of the state. 

In order to compute Pr(*s<sub>k</sub>*'|*s*,*a*)), the probability of the transition from state *s* with action *a* to the knowledge level of state *s'* (*s<sub>k</sub>*'), we multiply the `"prob_knowledge_gain"` (the probablility the student will transition from a lower to higher knowledge state, given their engagement level in *s*) by the `"action_prob_knowledge_gain_mult"` of action *a* (a multiplier that makes transitions to higher knowledge levels more likely if the help action is more helpful with the information it gives). For each knowledge level separating *s* from *s'*, this quantity is exponentiated. For example, if *s* is at knowledge level 1 and *s'* is at knowledge level 3, the quantity would be squared. Finally, as mentioned before, transitions to lower knolwedge states in *s'* from *s* are not allowed, since we assume that knowledge cannot be lost. 

In order to compute Pr(*s<sub>e</sub>*'|*s*,*a*)), the probability of the transition from state *s* with action *a* to the engagement level of state *s'* (*s<sub>e</sub>*'):
- If *s<sub>e</sub>* is the low engagement state, we multiply the `"prob_engagement_gain"` (the probability the student will transition from a low to high engagement state, dependent on *s<sub>k</sub>*) by `"action_prob_engagement_gain_mult"` of action *a* (a multiplier that makes transitions from low to high engagement more likely if the action is aimed at increasing engagement).
- If *s<sub>e</sub>* is the high engagement state, we take the `"prob_engagement_loss"` (the probability the student will transition from a high to low engagement state, dependent on *s<sub>k</sub>*).

In order to compute Pr(*s<sub>a</sub>*'|*s*,*a*)), the probability of state *s* with action *a* to the attempt level of state *s'* (*s<sub>a</sub>*'). This quantity is also equivalent to asking what is the probability of the student getting the correct answer, since we're assessing the likelihood of moving from one attempt to the next and after getting a question wrong, the student moves to the next attempt, as where if they get the question correct they move to the first attempt of the next question. The student has a maximum of 3 attempts, after which they are moved to the first attempt of the next question. This quantity is computed by multiplying `"prob_correct_answer"` (the probability of answering the attempt correctly based on *s<sub>k</sub>* if *s<sub>a</sub>* represents the first attempt) or `"prob_correct_answer_after_1_attempt"` (the same as `"prob_correct_answer"`, but for *s<sub>a</sub>* greater than the first attempt) by `"correct_prob_mult_for_engagement"` (a float value represnting the decrease in expected accuracy if the student is disengaged in *s*). 

### Computing the Observation Matrix















