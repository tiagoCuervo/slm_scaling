from slm.vocoder import ASSETS


def test_official_vocoder_assets_are_pinned():
    assert set(ASSETS) == {"checkpoint", "config"}
    for asset in ASSETS.values():
        assert asset["url"].startswith("https://dl.fbaipublicfiles.com/")
        assert len(asset["sha256"]) == 64
