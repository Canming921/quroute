import numpy as np
from qiskit import QuantumCircuit

from quroute import (
    GreedyDistancePolicy,
    PolicyRouter,
    RandomMaskedPolicy,
    RoutingEnv,
    grid_topology,
    linear_topology,
    run_episode,
)


def _hw_valid(circuit, cm):
    edges = {tuple(e) for e in cm.get_edges()}
    for inst in circuit.data:
        qs = [circuit.find_bit(q).index for q in inst.qubits]
        if len(qs) == 2 and (qs[0], qs[1]) not in edges and (qs[1], qs[0]) not in edges:
            return False
    return True


def _long_range_circuit():
    qc = QuantumCircuit(6)
    qc.h(range(6))
    qc.cx(0, 5)
    qc.cx(1, 4)
    qc.cx(0, 3)
    return qc


def test_env_reset_shapes():
    cm = grid_topology(2, 3)
    env = RoutingEnv(cm, circuit=_long_range_circuit())
    obs, info = env.reset(seed=0)
    assert obs.shape == (env.observation_size,)
    assert info["action_mask"].shape == (env.num_actions,)
    assert info["action_mask"].dtype == bool


def test_env_mask_never_all_false():
    cm = grid_topology(2, 3)
    env = RoutingEnv(cm, circuit=_long_range_circuit())
    env.reset(seed=0)
    assert env.valid_action_mask().any()


def test_random_policy_completes_and_is_valid():
    cm = grid_topology(2, 3)
    env = RoutingEnv(cm, circuit=_long_range_circuit(), max_steps=2000)
    res = run_episode(env, RandomMaskedPolicy(seed=1), seed=1)
    assert _hw_valid(res.circuit, cm)
    assert env.front == []


def test_greedy_policy_completes_and_is_valid():
    cm = grid_topology(2, 3)
    env = RoutingEnv(cm, circuit=_long_range_circuit())
    pol = GreedyDistancePolicy(env.n_phys, env.num_actions, seed=2)
    res = run_episode(env, pol, seed=2)
    assert _hw_valid(res.circuit, cm)


def test_no_routing_needed_terminates_immediately():
    cm = linear_topology(3)
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(1, 2)
    env = RoutingEnv(cm, circuit=qc)
    env.reset(seed=0)
    assert env.front == []  # all gates already executable -> no SWAPs to decide


def test_policy_router_integrates_with_pass():
    from qiskit.transpiler import PassManager

    from quroute import QurouteRouter

    cm = grid_topology(2, 3)
    env_for_dims = RoutingEnv(cm, circuit=_long_range_circuit())
    pol = GreedyDistancePolicy(env_for_dims.n_phys, env_for_dims.num_actions, seed=3)
    pm = PassManager([QurouteRouter(cm, PolicyRouter(pol))])
    routed = pm.run(_long_range_circuit())
    assert _hw_valid(routed, cm)


def test_reward_sign_swap_costs():
    cm = grid_topology(2, 3)
    env = RoutingEnv(cm, circuit=_long_range_circuit(), shaping=0.0, swap_cost=-1.0)
    env.reset(seed=0)
    mask = env.valid_action_mask()
    action = int(np.flatnonzero(mask)[0])
    _, reward, _, _, _ = env.step(action)
    assert reward == -1.0  # pure swap cost when shaping is off
