from src.bandit import GaussianThompsonSampler


def test_bandit_updates_mean():
    assets = ["AAPL", "GOOGL"]
    bandit = GaussianThompsonSampler(assets)

    # initial means should be zero
    assert bandit.means["AAPL"] == 0.0

    # apply a positive reward
    bandit.update("AAPL", 0.02)

    # mean should move in direction of reward
    assert bandit.means["AAPL"] > 0.0