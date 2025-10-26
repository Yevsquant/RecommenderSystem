from recommender.abtest import ABTestSimulator

def test_abtest_basic():
    sim = ABTestSimulator(n_users=2000)
    sim.assign_buckets("testexp")
    _, metrics = sim.run_ab_test("testexp", [0], [1])
    assert "ctr_diff" in metrics and isinstance(metrics["ctr_diff"], float)

def test_holdout_and_reverse():
    sim = ABTestSimulator(n_users=2000)
    h = sim.simulate_holdout_diff()
    r = sim.simulate_reverse_experiment()

    assert abs(h["holdout_ctr_diff"]) < 1
    assert abs(r["diff"]) < 1