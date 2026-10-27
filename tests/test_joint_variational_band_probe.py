import torch

from scripts.probe_joint_variational_band import gaussian_kl


def test_gaussian_kl_is_zero_for_identical_distribution_with_zero_gradient():
    mu = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    value = gaussian_kl(mu, sigma, target_y=4.0, prior_sigma=1.5)
    grad_mu, grad_sigma = torch.autograd.grad(value, (mu, sigma))

    assert torch.allclose(value, torch.tensor(0.0, dtype=torch.float64), atol=1e-15)
    assert torch.allclose(grad_mu, torch.tensor(0.0, dtype=torch.float64), atol=1e-15)
    assert torch.allclose(grad_sigma, torch.tensor(0.0, dtype=torch.float64), atol=1e-15)


def test_gaussian_kl_penalizes_mean_and_scale_mismatch():
    prior_mean, prior_sigma = 4.0, 1.5
    matched = gaussian_kl(torch.tensor(prior_mean), torch.tensor(prior_sigma), prior_mean, prior_sigma)
    shifted = gaussian_kl(torch.tensor(prior_mean + 1), torch.tensor(prior_sigma), prior_mean, prior_sigma)
    diffuse = gaussian_kl(torch.tensor(prior_mean), torch.tensor(prior_sigma * 2), prior_mean, prior_sigma)

    assert matched.item() == 0.0
    assert shifted.item() > matched.item()
    assert diffuse.item() > matched.item()
