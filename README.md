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

The state space has 3 dimensions: knowledge level (`"knowledge_states`), engagement level (`"engagement_states"`), and attempt number (`"attempt_states"`). Since there are 4 knowledge levels, 2 engagement states (low and high), and 4 attempt states (attempt 1 after getting a problem right, attempt 1 after getting a problem right, attempt 2, attempt 3) there are a total of 4 x 2 x 4 = 32 possible states. 