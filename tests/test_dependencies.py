import importlib.util

def test_datasets_available():
    """Verify that the 'datasets' library is installed and available."""
    loader = importlib.util.find_spec("datasets")
    assert loader is not None, "datasets library not found"

def test_datasets_vision_extras():
    """Verify that image processing support is available in datasets."""
    import datasets
    from datasets import Features, Image
    
    # Check if Image feature exists
    features = Features({"image": Image()})
    assert "image" in features
