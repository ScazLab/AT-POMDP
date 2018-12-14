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

The reward function is represnted by a matrix with 4 dimensions: start state (*t*), end state (*t+1*), action, and observation. With this matrix representation we can reward transitions between states. We compute the reward matrix in the `generate_reward_matrix()` function in [pomdp_setup_reward_transition_matrices.py](https://github.com/ScazLab/AT-POMDP/blob/master/pomdp_setup_reward_transition_matrices.py). 

There are 5 parameters in our JSON file that influence the computation of this reward matrix: 
- `"reward_for_first_attempt_actions"`: If the start state is one of the attempt 1 states, we set the reward of taking any action other than "no-action" as -1000 to ensure that our POMDP never chooses any action other than "no-action" for the first attempt. This allows the 5th grade student to make a first attempt on a math problem before receiving help from the system. 
- `"engagement_reward"`: Any time engagement state changes from the start state to the end state, the engagement reward is applied to that transition, this applies for both gains and losses in engagement. 
- `"knowledge_reward"`: Any time engagement state changes from the start state to the end state, the engagement reward is applied to that transition, this applies only to gains in knowledge, since our model doesn't allow loss of knowledge (transitions from higher to lower knowledge states). 
- `"action_rewards"`: Each action has a certain cost associated with it. This cost is derived from how long each help action takes for the student and the robot to complete. 
- `"end_state_remain_reward"`: This reward value is the one given for states with the highest possible engagement and knowledge. Once students have reached the highest possible engagement and knowledge, we do not reward them further. 

### Computing the Transition Matrix

### Computing the Observation Matrix