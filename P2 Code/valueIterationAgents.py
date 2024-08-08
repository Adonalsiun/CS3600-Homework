# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        # Implementation found in Textbook Figure 17.2.6, PG. 1057
        "*** YOUR CODE HERE ***"

        # Get all states in MDP
        states = self.mdp.getStates()

        # Iterate for a specific number of iterations
        for i in range(0,self.iterations):
            newValues = util.Counter()

            # Loop through each state in MDP
            for state in states:
                maxValue = float("-inf")
                actions = self.mdp.getPossibleActions(state)

                # Evaluate actions to find the one with the max value
                for action in actions:
                    transition = self.mdp.getTransitionStatesAndProbs(state, action)
                    sum = 0.0
                    for nextState, prob in transition:
                        reward = self.mdp.getReward(state, action, nextState)
                        sum += prob * (reward + self.discount * self.values[nextState])
                    maxValue = max(maxValue, sum)
                if maxValue != float("-inf"):
                    newValues[state] = maxValue
            self.values = newValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # Implementation found in Textbook Equation 17.8, PG 1050
        "*** YOUR CODE HERE ***"
        # Get the transition states and probabilities for states and actions
        transition = self.mdp.getTransitionStatesAndProbs(state, action)

        # Set initial Q-Value
        qValue = 0.0

        # Create loop to get parts of Bellman (Reward, Discount, Transition)
        # Add to Q-Value
        for nextState, prob in transition:
            reward = self.mdp.getReward(state, action, nextState)
            discountValue = self.discount * self.values[nextState]
            qValue += prob * (reward + discountValue)
        return qValue
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # Implementation found in Textbook Figure 17.9, PG. 1063
        "*** YOUR CODE HERE ***"
        # Check for current state (terminal, then end)
        if self.mdp.isTerminal(state):
            return None

        # Create values to track best action and maximum q value
        bestAction = None
        maxQValue = float("-inf")

        # Get possible actions and loop to find maximum q value
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            qValue = self.computeQValueFromValues(state, action)
            # Update values if the inequality is true
            if qValue > maxQValue:
                maxQValue = qValue
                bestAction = action
        return bestAction
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
