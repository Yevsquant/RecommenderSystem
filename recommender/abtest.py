# A/B Testing + holdout simulation
import numpy as np
import pandas as pd
from typing import Dict, Tuple

class ABTestSimulator:
    """
    Simulate an A/B tseting, holdout mechanism, and reverse experiment for recommender experiments.
    Initialize a pool of users ('experiment': 90%, 'holdout': 10%)
    """

    def __init__(self, n_users=10000, seed=42):
        """
        Initialize the user pool
        Split uesrs into experiment and holdout
        """
        self.rng = np.random.default_rng(seed)
        self.n_users = n_users
        self.users = pd.DataFrame({
            "user_id": np.arange(n_users),
            "group": np.where(self.rng.random(n_users) < 0.9, "experiment", "holdout")
        })

    def assign_buckets(self, experiemnt_name: str, n_buckets=10) -> pd.DataFrame:
        """
        Create n equally sized buckets for experiment-layer randomization
        """
        self.users[f"{experiemnt_name}_bucket"] = self.rng.integers(0, n_buckets, size=self.n_users)
        return self.users
    
    def run_ab_test(self, experiemnt_name: str, control_buckets: list, experiment_buckets: list,
                    base_ctr=0.05, experiment_lift=0.02, days=7) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Simulate CTR improvement over a number of days between control and experiment groups.
        Log CTR
        """
        df = self.users[self.users.group == "experiment"].copy() # Filter out experiment users
        mask_control = df[f"{experiemnt_name}_bucket"].isin(control_buckets)
        mask_experiment = df[f"{experiemnt_name}_bucket"].isin(experiment_buckets)

        df["group_type"] = np.where(mask_control, "control", np.where(mask_experiment, "experiment", "unused"))

        # Simulate daily activity
        logs = []
        for day in range(days):
            # Loop through all users in the A/B test (except 'unused')
            for _, r in df[df.group_type != "unused"].iterrows():
                ctr = base_ctr + (experiment_lift if r.group_type == "experiment" else 0)
                active = self.rng.random() < 0.6
                clicks = np.sum(self.rng.random(10) < ctr) if active else 0
                logs.append({
                    "day": day,
                    "user_id": r.user_id,
                    "group_type": r.group_type,
                    "active": active,
                    "clicks": clicks
                })
        log_df = pd.DataFrame(logs)
        results = log_df.groupby("group_type")["clicks"].mean().to_dict()
        diff = results["experiment"] - results["control"]

        return log_df, {"ctr_diff": diff,
                        "control_ctr": results["control"],
                        "experiment_ctr": results["experiment"]}
    
    def simulate_holdout_diff(self, holdout_ctr=0.05, live_ctr=0.06):
        """
        Simulate metric difference between 90% live traffic (experiment) and 10% holdout bucket.
        """
        holdout_users = self.users[self.users.group == "holdout"]
        experiment_users = self.users[self.users.group == "experiment"]

        holdout_clicks = np.sum(self.rng.random(len(holdout_users)) < holdout_ctr)
        exp_clicks = np.sum(self.rng.random(len(experiment_users)) < live_ctr)
        diff = (exp_clicks / len(experiment_users)) - (holdout_clicks / len(holdout_users))

        return {"holdout_ctr_diff": diff, "holdout_ctr": holdout_ctr, "live_ctr": live_ctr}
    
    def simulate_reverse_experiment(self, new_strategy_ctr=0.065, old_strategy_ctr=0.05, reverse_frac=0.05):
        """
        Simulate reverse experiment where 5% users continue to see old model after rollout for long-term monitoring
        """
        n_reverse = int(self.n_users * reverse_frac)
        reverse_users = self.rng.choice(self.users.user_id, size=n_reverse, replace=False)
        new_users = np.setdiff1d(self.users.user_id, reverse_users)
        new_avg = np.mean(self.rng.random(len(new_users)) < new_strategy_ctr)
        old_avg = np.mean(self.rng.random(len(reverse_users)) < old_strategy_ctr)

        return {
            "new_ctr": new_avg,
            "reverse_ctr": old_avg,
            "diff": new_avg - old_avg,
            "reverse_frac": reverse_frac
        }