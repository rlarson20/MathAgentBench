# TO IMPLEMENT
# test_docker_repl_with_numpy()
# test_docker_repl_memory_limit()


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
