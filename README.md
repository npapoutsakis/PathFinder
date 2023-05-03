## Optimal Path Finding
The project was implemented in the context of the course "Artificial Intelligence (AI)" at the Technical University of Crete

This program solves the problem of finding an optimal path for a vehicle to travel from its initial position to a final destination while avoiding obstacles and staying within the road boundaries.

Setup
To use this program, you will need to have Python 3.7 installed. You can download it from the official Python website.

Next, download the platform provided in the "Documents" folder of the eclass. The platform automates several operations such as opening and visualizing scenarios, and displaying the branching and paths taken by the algorithms.

Usage
To run the program, navigate to the root folder of the exercise and run main.py. From there, you can select the scenario you want to run and enable/disable the algorithms you have developed within the Algorithms folder.

Algorithms
This program uses two types of search algorithms: Weighted A* and IDA*. For each variant of Weighted A*, a weight w is used with the estimation function f(n) = g(n) + w * h(n). The cost function g(n) is given as a ready-made function within the files you are asked to fill in.

Two heuristic implementations are requested: one based on the Euclidean distance and one that provides better performance for at least one algorithm in one of the scenarios. The chosen heuristic should have a logical basis and be documented in the report.

Different values of w > 1 are experimented with in each scenario, and a value of w is chosen based on the observations made. The results are summarized in tables, graphs, or screenshots in the report, and the final value of w is justified based on the experimental results.

Output
The program displays on the screen and produces an output file listing for each algorithm and scenario:

The number of nodes expanded (Visited nodes number)
The exact route proposed by the algorithm (Path)
The value of the selected heuristic from the original node
Conclusion
This program provides a solution for finding an optimal path for a vehicle to travel while avoiding obstacles and staying within the road boundaries. It uses two types of search algorithms and different heuristic implementations to achieve this goal. The program displays the output on the screen and produces an output file for each algorithm and scenario.
