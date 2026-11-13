"""Model mechanics on NON-MEDICAL fixtures (see fixtures_nonmedical.py). 64x64 inputs for speed;
one 240x240 test pins the production geometry."""
import numpy as np
import pytest
import torch
from sklearn.decomposition import PCA

from brainsentinel.models import build_model, load_model, save_model
from brainsentinel.models.patchcore_model import greedy_coreset
from brainsentinel.models.reducers import PCAReducer, RandomSelectReducer
from tests.fixtures_nonmedical import batches, field

S = 64
SQ = (20, 24, 12)


@pytest.fixture(scope="module")
def normals():
    return field(24, S, seed=0)


@pytest.fixture(scope="module")
def abnormal():
    return field(6, S, seed=1, square=SQ)


def _fit(kind, normals, **hp):
    torch.manual_seed(0)
    m = build_model(kind, {"pretrained": False, "input_size": S, **hp}) if kind != "autoencoder" else \
        build_model(kind, {"input_size": S, **hp})
    info = m.fit(batches(normals), device="cpu") if kind != "autoencoder" else \
        m.fit(batches(normals), val_loader=batches(normals[:8]), epochs=3, device="cpu", log=lambda *_: None)
    return m, info


@pytest.mark.parametrize("kind,hp", [("padim", {"d_out": 40}), ("patchcore", {"coreset_ratio": 0.2}),
                                     ("autoencoder", {"width": 8})])
def test_output_shapes_finite_and_deterministic(kind, hp, normals, abnormal):
    m, _ = _fit(kind, normals, **hp)
    maps, scores = m.predict(abnormal)
    assert maps.shape == (6, S, S) and scores.shape == (6,)
    assert torch.isfinite(maps).all() and torch.isfinite(scores).all()
    maps2, scores2 = m.predict(abnormal)
    assert torch.equal(maps, maps2) and torch.equal(scores, scores2)


@pytest.mark.parametrize("kind,hp", [("padim", {"d_out": 40}), ("patchcore", {"coreset_ratio": 0.2})])
def test_feature_models_flag_and_localise_injected_region(kind, hp, normals, abnormal):
    m, _ = _fit(kind, normals, **hp)
    _, s_norm = m.predict(field(6, S, seed=7))
    maps, s_abn = m.predict(abnormal)
    assert s_abn.min() > s_norm.max()                           # separable on this trivial fixture
    y, x, s = SQ
    inside = maps[:, y:y + s, x:x + s].mean()
    outside = torch.cat([maps[:, :y].flatten(), maps[:, y + s + 6:].flatten()]).mean()
    assert inside > 1.5 * outside


def test_autoencoder_loss_decreases_and_early_stop_uses_val_normals(normals):
    torch.manual_seed(0)
    m = build_model("autoencoder", {"width": 8, "input_size": S})
    info = m.fit(batches(normals), val_loader=batches(normals[:8]), epochs=6, device="cpu", log=lambda *_: None)
    h = info["history"]
    assert h[-1]["train_mse"] < h[0]["train_mse"]
    assert all(np.isfinite(r["val_normal_mse"]) for r in h)


@pytest.mark.parametrize("kind,hp", [("padim", {"d_out": 40}), ("patchcore", {"coreset_ratio": 0.2}),
                                     ("autoencoder", {"width": 8})])
def test_save_load_roundtrip_is_identical(kind, hp, normals, abnormal, tmp_path):
    m, _ = _fit(kind, normals, **hp)
    save_model(m, tmp_path)
    m2 = load_model(tmp_path)
    a, sa = m.predict(abnormal)
    b, sb = m2.predict(abnormal)
    assert torch.allclose(a, b, atol=1e-5) and torch.allclose(sa, sb, atol=1e-5)


def test_production_geometry_240():
    """layer2 30x30 + layer3 15x15 -> 384-d patches; map aligns with the 240x240 mask grid."""
    x = field(2, 240, seed=3)
    pc = build_model("patchcore", {"pretrained": False, "coreset_ratio": 0.5})
    info = pc.fit(batches(x, 2), device="cpu")
    maps, scores = pc.predict(x)
    assert pc.grid == (30, 30) and info["d_used"] == 384 and info["n_patches"] == 2 * 900
    assert maps.shape == (2, 240, 240) and scores.shape == (2,)
    pd = build_model("padim", {"pretrained": False, "d_out": 100})
    pd.fit(batches(x, 2), device="cpu")
    assert pd.predict(x)[0].shape == (2, 240, 240) and tuple(pd.mean.shape) == (900, 100)


def test_unfitted_models_refuse_to_predict():
    for kind in ("padim", "patchcore"):
        m = build_model(kind, {"pretrained": False, "input_size": S})
        with pytest.raises(RuntimeError):
            m.predict(field(1, S, 0))


def test_padim_covariance_matches_numpy(normals):
    m, _ = _fit("padim", normals, d_out=12, reg_eps=0.01)
    z = torch.cat([m.reducer.transform(m.embed(b["image"])) for b in batches(normals)]).double().numpy()
    p = 7
    cov = np.cov(z[:, p, :].T) + 0.01 * np.eye(12)      # reg_eps keeps the matrix invertible
    ref = np.linalg.inv(cov)
    got = m.inv_cov[p].double().numpy()
    assert np.allclose(m.mean[p].numpy(), z[:, p, :].mean(0), atol=1e-4)
    assert np.allclose(got, ref, rtol=2e-3, atol=2e-3 * np.abs(ref).max())


def test_padim_needs_two_images():
    m = build_model("padim", {"pretrained": False, "input_size": S, "d_out": 10})
    with pytest.raises(ValueError):
        m.fit(batches(field(1, S, 0), 1), device="cpu")


# ------------------------------------------------------------------------------ PCA / coreset
def test_pca_matches_sklearn_and_is_train_only():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 20)) @ rng.normal(size=(20, 20)) + 5.0
    r = PCAReducer(20, variance=0.95).fit([torch.from_numpy(X[:250]).float(), torch.from_numpy(X[250:]).float()])
    sk = PCA(n_components=20).fit(X)
    k = int(np.searchsorted(np.cumsum(sk.explained_variance_ratio_), 0.95 - 1e-12)) + 1
    assert r.d_out == k and r.n_fit_vectors == 500
    assert np.allclose(r.explained_variance_ratio.numpy(), sk.explained_variance_ratio_, atol=1e-4)
    z = r.transform(torch.from_numpy(X).float()).numpy()
    assert np.allclose(z.var(0, ddof=1), sk.explained_variance_[:k], rtol=1e-2)
    held = rng.normal(size=(50, 20)) @ rng.normal(size=(20, 20)) + 5.0         # unseen data: no refit
    r.transform(torch.from_numpy(held).float())
    assert r.n_fit_vectors == 500


def test_pca_argument_validation():
    with pytest.raises(ValueError):
        PCAReducer(8)
    with pytest.raises(ValueError):
        PCAReducer(8, n_components=3, variance=0.9)
    with pytest.raises(ValueError):
        PCAReducer(8, variance=1.5)
    with pytest.raises(RuntimeError):
        PCAReducer(8, n_components=2).transform(torch.zeros(3, 8))


def test_random_selection_is_seeded_and_bounded():
    a, b = RandomSelectReducer(384, 100, 0), RandomSelectReducer(384, 100, 0)
    assert torch.equal(a.index, b.index) and len(set(a.index.tolist())) == 100
    with pytest.raises(ValueError):
        RandomSelectReducer(10, 11)


def test_greedy_coreset_covers_better_than_random_and_is_deterministic():
    g = torch.Generator().manual_seed(0)
    X = torch.cat([torch.randn(3000, 16, generator=g), 6 + 0.3 * torch.randn(40, 16, generator=g)])  # rare cluster
    cover = lambda idx: torch.cdist(X, X[idx]).min(1).values.max()  # noqa: E731
    gi = greedy_coreset(X, 60, proj_dim=16, seed=0)
    ri = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))[:60]
    assert cover(gi) < cover(ri)
    assert torch.equal(gi, greedy_coreset(X, 60, proj_dim=16, seed=0))
    assert len(set(gi.tolist())) == 60


def test_patchcore_pca_variant_shrinks_bank_dimension(normals):
    full, i0 = _fit("patchcore", normals, coreset_ratio=0.1)
    red, i1 = _fit("patchcore", normals, coreset_ratio=0.1, reduction="pca", pca_variance=0.9)
    assert i1["d_used"] < i0["d_used"] == 384
    assert red.reducer.n_fit_vectors <= 500 * 64                  # PCA saw only the (train) fit images
    assert red.predict(field(2, S, 9))[0].shape == (2, S, S)


def test_pretrained_flag_false_has_no_download(normals):
    m = build_model("patchcore", {"pretrained": False, "input_size": S})
    assert m.backbone.pretrained is False
