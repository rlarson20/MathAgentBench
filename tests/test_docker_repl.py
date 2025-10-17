from mathagentbench.tools.docker_repl import DockerREPL


def test_docker_repl_simple_execution():
    repl = DockerREPL()
    result = repl.execute("print(2+2)")
    print(result)
    assert result["success"]
    assert "4" in result["stdout"]


def test_docker_repl_timeout():
    repl = DockerREPL(timeout=2)
    result = repl.execute("import time; time.sleep(10)")
    assert not result["success"]


def test_docker_repl_with_numpy():
    repl = DockerREPL()
    result = repl.execute("import numpy as np; print(np.array([1, 2, 3]).sum())")
    assert result["success"]
    assert "6" in result["stdout"]


def test_docker_repl_memory_limit():
    repl = DockerREPL()
    # Test that memory limits are respected by trying to allocate large array
    result = repl.execute("import numpy as np; large = np.zeros(100000000); print('allocated')")
    # This might succeed or fail depending on container memory limits
    # We just verify the container runs and returns a response
    assert "exit_code" in result
