"""Gradio UI for running evaluations and viewing results."""

import gradio as gr


def run_eval(benchmark_file, model: str, agent_type: str):
    """Run evaluation (callback for Gradio)."""
    # TODO: Implement
    return {"status": "Not implemented"}


def view_traces(results_file):
    """Load and format traces for display."""
    # TODO: Implement
    return "Not implemented"


def compare_models(results_files):
    """Generate comparison charts."""
    # TODO: Implement
    return "Not implemented"


def create_ui():
    """Build Gradio interface."""
    with gr.Blocks(title="MathAgent Bench") as demo:
        gr.Markdown("# MathAgent Bench")
        gr.Markdown("Evaluate LLM agents on math problems with tool use")

        with gr.Tab("Run Evaluation"):
            benchmark = gr.File(label="Benchmark JSON")
            model = gr.Dropdown(
                choices=[
                    "openai/gpt-4o-mini",
                    "anthropic/claude-3.5-sonnet",
                    "meta-llama/llama-3.1-70b-instruct",
                ],
                label="Model",
                value="openai/gpt-4o-mini",
            )
            agent = gr.Radio(
                choices=["react", "reflexion"], label="Agent Type", value="react"
            )
            run_btn = gr.Button("Run Evaluation")
            output = gr.JSON(label="Results")

            run_btn.click(run_eval, inputs=[benchmark, model, agent], outputs=output)

        with gr.Tab("View Traces"):
            gr.Markdown("TODO: Trace viewer")

        with gr.Tab("Compare Models"):
            gr.Markdown("TODO: Model comparison")

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch()
