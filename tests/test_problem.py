from mathagentbench.core.problem import Problem, load_benchmark


def test_problem_dataclass_creation():
    """Test Problem Class accurately represents a problem from the datasets"""
    p = Problem(id="test", question="What is 1+1?", answer="2", answer_type="integer")
    assert p.id == "test"
    assert p.tags == []


def test_load_benchmark_valid_json(sample_benchmark):
    """Test that Problems loaded through JSON are valid"""
    """using mock subset for now"""
    problems = load_benchmark(sample_benchmark)
    assert len(problems) == 1
    assert problems[0].id == "test_001"


def test_load_benchmark_missing_fields():
    """Test error handling for loading from JSON"""
    pass


def test_problem_validation():
    """not entirely sure what this is supposed to be"""
    pass
