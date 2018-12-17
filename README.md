# AT-POMDP

The Assistive Tutor POMDP (AT-POMDP) developed by Aditi Ramachandran and Sarah Strohkorb Sebo, was designed to select appropriate tutoring actions to support 5th grade students in a one-on-one tutoring interaction with a Nao robot on the topic of long-division mathematics. 

If you are using this software or one of its components, we recommend that you cite the paper in which we present the AT-POMDP and its evaluation with 5th grade students:

> Aditi Ramachandran, Sarah Strohkorb Sebo, Brian Scassellati (2019). Personalized Robot Tutoring using the Assistive Tutor POMDP (AT-POMDP). In *Proceedings of The Thirty-Third AAAI Conference on Artificial Intelligence (AAAI 2019)*. Honolulu, HI, USA. [PDF](https://scazlab.yale.edu/sites/default/files/files/Ramachandran_Sebo_AAAI_2018.pdf)

## Prerequisites and Dependencies
This subsection describes what needs to be set up prior to being able to run the code used to generate AT-POMDP. The requirements are listed below:
- Install `numpy`. 
- The python code in this repository will look for the `pomdp-solve` executable in your `$PATH`. The POMDP solver used to create AT-POMDP was originally implemented by Kaelbling, Littman, and Cassandra (1998) and was modified by Roncone, Mangin, and Scassellati (2017). Please refer to their [instructions](https://github.com/scazlab/task-models#prerequisites-for-using-the-pomdp-solvers) on installing this executable by reading the section titled "Prerequisites for using the POMDP solvers" and utilizing the code they reference at (https://github.com/scazlab/pomdp-solve).

## File-by-File Details
A description of each of the files in the top-level directory is listed below: 
1. `atpomdp_params.json` -  This file contains all of the parameter choices used in creating AT-POMDP. This file is provided as a command line argument when running `command_line_demo.py`.
2.  `command_line_demo.py` - This python script contains code to: 
    * Parse parameters from the parameter input file 
    * Call functions to create the individual matrices (reward, transition, and observation) required to build AT-POMDP. The functions called are defined in `pomdp_setup_reward_transition_matrices.py` and `pomdp_setup_observation_matrices.py` to generate these matrices based on the input parameters.
    * Create the AT-POMDP model and solve for AT-POMDP's policy.
    * Execute AT-POMDP's policy in an interactive command line demo in which a user can enter student observations and see which action AT-POMDP selects.
3. `pomdp_setup_observation_matrices.py` - This python file contains the code to build the observation matrix used in the creation of AT-POMDP.
4. `python_setup_reward_transition_matrices.py` - This python file contains the code to build the reward and transition matrices used in the creation of AT-POMDP.

## Usage
For an interactive demo on how AT-POMDP can be used, run `python command_line_demo.py atpomdp_params.json`. This script will execute the code to generate AT-POMDP's policy and allows the user to input valid observations to see the actions that AT-POMDP selects along with the changing belief state. Valid observations are formatted as follows: `{"R-slow", "R-med", "R-fast", "W-slow", "W-med", "W-fast"}`. These observations encapsulate whether the student answered the attempt right ("R") or wrong ("W") and whether their speed on the attempt was slow, medium, or fast ("slow", "med", "fast"). When the user is done entering observations, they can enter "done" to exit the command line demo. The demo is designed to show how AT-POMDP was used in the scenario developed for 5th grade students and shows the action selected by AT-POMDP for problems that each have 3 attempts.

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

The transition function is represnted by a matrix with 3 dimensions: start state (*s*), end state (*s'*), and action. We compute the transition matrix in the `generate_transition_matrix()` function in [pomdp_setup_reward_transition_matrices.py](https://github.com/ScazLab/AT-POMDP/blob/master/pomdp_setup_reward_transition_matrices.py). 

Each transition from the start state to end state taking a specified action is calculated according to the following formula: Pr(*s'*|*s*,*a*) = Pr(*s<sub>k</sub>'*|*s*,*a*)) x Pr(*s<sub>e</sub>'*|*s*,*a*)) x Pr(*s<sub>a</sub>'*|*s*,*a*)), where *s<sub>k</sub>* represents the knowledge dimension of the state, *s<sub>e</sub>* represents the engagement dimension of the state, and *s<sub>a</sub>* represents the attempt dimension of the state. 

In order to compute Pr(*s<sub>k</sub>'*|*s*,*a*)), the probability of the transition from state *s* with action *a* to the knowledge level of state *s'* (*s<sub>k</sub>*'), we multiply the `"prob_knowledge_gain"` (the probablility the student will transition from a lower to higher knowledge state, given their engagement level in *s*) by the `"action_prob_knowledge_gain_mult"` of action *a* (a multiplier that makes transitions to higher knowledge levels more likely if the help action is more helpful with the information it gives). For each knowledge level separating *s* from *s'*, this quantity is exponentiated. For example, if *s* is at knowledge level 1 and *s'* is at knowledge level 3, the quantity would be squared. Finally, as mentioned before, transitions to lower knolwedge states in *s'* from *s* are not allowed, since we assume that knowledge cannot be lost. 

In order to compute Pr(*s<sub>e</sub>'*|*s*,*a*)), the probability of the transition from state *s* with action *a* to the engagement level of state *s'* (*s<sub>e</sub>'*):
- If *s<sub>e</sub>* is the low engagement state, we multiply the `"prob_engagement_gain"` (the probability the student will transition from a low to high engagement state, dependent on *s<sub>k</sub>*) by `"action_prob_engagement_gain_mult"` of action *a* (a multiplier that makes transitions from low to high engagement more likely if the action is aimed at increasing engagement).
- If *s<sub>e</sub>* is the high engagement state, we take the `"prob_engagement_loss"` (the probability the student will transition from a high to low engagement state, dependent on *s<sub>k</sub>*).

In order to compute Pr(*s<sub>a</sub>'*|*s*,*a*)), the probability of state *s* with action *a* to the attempt level of state *s'* (*s<sub>a</sub>'*). This quantity is also equivalent to asking what is the probability of the student getting the correct answer, since we're assessing the likelihood of moving from one attempt to the next and after getting a question wrong, the student moves to the next attempt, as where if they get the question correct they move to the first attempt of the next question. The student has a maximum of 3 attempts, after which they are moved to the first attempt of the next question. This quantity is computed by multiplying `"prob_correct_answer"` (the probability of answering the attempt correctly based on *s<sub>k</sub>* if *s<sub>a</sub>* represents the first attempt) or `"prob_correct_answer_after_1_attempt"` (the same as `"prob_correct_answer"`, but for *s<sub>a</sub>* greater than the first attempt) by `"correct_prob_mult_for_engagement"` (a float value represnting the decrease in expected accuracy if the student is disengaged in *s*). 

### Computing the Observation Matrix

The observation function is represnted by a matrix with 3 dimensions: end state (*s'*), action, and observation. We compute the transition matrix in the `generate_observation_matrix()` function in [pomdp_setup_observation_matrices.py](https://github.com/ScazLab/AT-POMDP/blob/master/pomdp_setup_observation_matrices.py). 

The observation function computes the likelihood of seeing a particular observation, given *s'* and *a*. Since the *s'* encodes which attempt we are on, we have 100% certainty about whether the student answered the attempt correctly or incorrectly. Thus, we just need to compute the likelihood of the observation's speed component. This is computed by multiplying either the `"prob_speeds_for_high_engagement"` or `"prob_speeds_for_low_engagement"` (depending on *s<sub>e</sub>'* - low or high engagement in the end state) by `"action_speed_multipliers"` (a feature currently unused since all of its values are 1.0). 













