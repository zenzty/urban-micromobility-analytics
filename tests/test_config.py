from visual_analytics.config import DATA_DIR, PROJECT_ROOT


def test_project_paths_exist() -> None:
    assert PROJECT_ROOT.exists()
    assert DATA_DIR.exists()
