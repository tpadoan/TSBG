# Adjacency matrix:
* file name: matrix.txt
* format: space separated values, one matrix row per text line
* values are in \[0,7\] s.t. the i-th bit in the binary representation means that transport is available, 0-th bit for taxi, 1-st bit for bus, 2-nd bit for underground (ug).

# Edges lists:
* one file for each transport: taxi.txt, bus.txt, ug.txt
* format: one line for each node with some outgoing edge of the corresponding transport, containing: id of the source location followed by ids of destinations, space separated.

# Locations coordinates list:
* file name: coords.txt
* format: one line for each location, containing: id of the location followed by the its X Y coordinates (central pixel) in the map image, space separated.
