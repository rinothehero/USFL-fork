"""
Unified Simulation Framework for Split Federated Learning.

Lightweight, synchronous SFL simulation supporting 7 methods:
SFL, USFL, SCAFFOLD, Mix2SFL, GAS, MultiSFL, FedCBS.

Eliminates async/queue/polling overhead from the original SFL framework
while reusing all existing components (models, splitters, selectors, etc.).
"""
