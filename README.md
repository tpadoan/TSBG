# La fuga di Marco

Presented at the Sharper Night 2023 at Immaginario Scientifico and Trieste Next 2024 in Piazza Unità d'Italia, Trieste (Italy), the project is a digital tabletop game, spiritually similar to Scotland Yard.
The player, moving through a graph, has to avoid getting caught by the opponents, controlled by AI (either a pure-probabilistic method -- see below -- or based on reinforcement learning).
The AI only knows the player's initial position and the transports taken to move. 

The project includes a solver to compute optimal policies for all players, based on min-maxing the rewards associated with the possible moves in a fully observable setting.
The resulting policy is then applied stochastically based on the probability distribution of the possible player's current locations, inferred from the sequence of transports taken.

A neural network was trained using PPO, against a player who would move to the farthest position most of the time, otherwise randomly.
The corresponding model is also included, and the game can be played against the previous and the latter policy.
