# Experiment Description: starlink_550_120_satfill_60s

This file provides concise exogenous context for multimodal satellite
path-traffic tensor completion. It should describe static experiment knowledge
only. It must not include train/validation/test split statistics, masked
traffic observations, prediction results, or time-slice target values.

## 1. Simulation Setup

- Simulation window: 60 seconds.
- Time resolution: 1000 ms per tensor time slice.
- Number of time slices: 60.
- Satellite count: 120.
- Ground station count: 120.
- Ground-station assignment mode: satellite-locked, one ground station is
  associated with each satellite.
- Transport protocol: TCP NewReno with loss-based AIMD congestion control.
- Random seed: 123456789.

## 2. Constellation And Temporal Context

- Constellation name: Starlink-550-120.
- Orbital altitude: 550 km.
- Orbital planes: 10.
- Satellites per plane: 12.
- Inclination: 53.0 degrees.
- Orbit period: about 94.8 minutes.
- The 60-second simulation window covers only about 1.05 percent of one orbit.
- The normalized time phase time_index / 59 is used as a local phase inside
  this short simulation window. It helps distinguish time slices even when the
  topology changes sparsely.

## 3. ISL Topology And Routing Context

- ISL topology mode: isls_plus_grid.
- Each satellite can connect to neighboring satellites inside the same orbital
  plane and to neighboring satellites in adjacent planes.
- Intra-plane ISLs are links between satellites in the same orbital plane.
- Inter-plane ISLs are cross-plane links between adjacent orbital planes.
- Cross-plane ISLs can be removed when distance or geometry constraints make
  them infeasible.
- Routing identifier: algorithm_free_one_only_over_isls.
- End-to-end satellite traffic is routed over inter-satellite links, not through
  ground-station relay.
- Shortest-path hop count describes the topological distance between an ordered
  source-destination satellite pair.

## 4. Capacity And Bottleneck Context

- ISL link rate: 10000 Mbit/s.
- GSL link rate: 100 Mbit/s.
- GSL capacity is much lower than ISL capacity and can be a major access-side
  bottleneck.
- Inside the satellite mesh, a link with high edge betweenness lies on many
  shortest paths and can represent a structural bottleneck.
- If a few inter-satellite links carry a large share of shortest-path usage,
  path traffic prediction may depend strongly on those links.

## 5. Path-Traffic Tensor Semantics

- Target tensor: X[source_satellite, destination_satellite, time].
- Tensor shape: 120 x 120 x 60.
- Target unit: MB.
- Each non-zero entry represents the path traffic volume between one source
  satellite and one destination satellite at one time slice.
- The tensor completion task predicts unobserved non-zero path-traffic entries
  from a random transductive split.
- Zero target entries are excluded by the existing CoSTCo experiment pipeline.

## 6. Topology Metrics Used In Endogenous Text

- Algebraic connectivity lambda2 is the second-smallest eigenvalue of the graph
  Laplacian. Larger lambda2 usually indicates stronger connectivity redundancy.
- Mean shortest-path hop count summarizes typical path length.
- Network diameter is the largest finite shortest-path hop count.
- The ratio of OD pairs with more than 8 hops describes long-path pressure.
- Rolling edge-change average summarizes recent topology-change intensity.
- Top bottleneck links are the links with the highest edge betweenness in the
  current topology.
