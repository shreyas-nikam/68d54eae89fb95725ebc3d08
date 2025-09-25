import pytest
from definition_ede02ffed5da4a9d82f3e2489608bfa3 import interactive_analysis

@pytest.mark.parametrize("bias_factor, reweighting_factor", [
    (0.3, 0.2),  # Typical values
    (0.0, 0.0),  # No bias, no reweighting
    (0.5, 1.0),  # High bias, full reweighting
    (0.2, 0.0),   # Some bias, no reweighting
    (0.0, 0.5),  # No bias, some reweighting
])
def test_interactive_analysis(bias_factor, reweighting_factor, capsys):
    # Call the function and capture stdout
    interactive_analysis(bias_factor, reweighting_factor)
    captured = capsys.readouterr()

    # Assert that something was printed (basic check, replace with more specific assertions if needed)
    assert len(captured.out) > 0