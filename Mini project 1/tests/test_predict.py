

def test_predict_one_returns_positive_float(trained_predictor, sample_laptop):
    price = trained_predictor.predict_one(sample_laptop)
    assert isinstance(price, float)
    assert price > 0


def test_predict_one_is_deterministic(trained_predictor, sample_laptop):
    """the same input should always give the same output"""
    price1 = trained_predictor.predict_one(sample_laptop)
    price2 = trained_predictor.predict_one(sample_laptop)
    assert price1 == price2


def test_predict_one_reasonable_range(trained_predictor, sample_laptop):
    """price must be in a reasonable range for laptops"""
    price = trained_predictor.predict_one(sample_laptop)
    assert 5000 < price < 400000


def test_predict_batch_returns_correct_count(trained_predictor, sample_laptop):
    rows = [sample_laptop, sample_laptop, sample_laptop]
    prices = trained_predictor.predict_batch(rows)
    assert len(prices) == 3
    assert all(p > 0 for p in prices)