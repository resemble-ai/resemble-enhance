import argparse
from functools import partial

import gradio as gr
import torch
import torchaudio

from resemble_enhance.enhancer.inference import denoise, enhance

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def _fn(path, solver, nfe, tau, denoising, unlimited):
    if path is None:
        gr.Warning("Please upload an audio file.")
        return None, None

    info = torchaudio.info(path)
    if not unlimited and (info.num_frames / info.sample_rate > 60):
        gr.Warning("Only audio files shorter than 60 seconds are supported.")
        return None, None

    solver = solver.lower()
    nfe = int(nfe)
    lambd = 0.9 if denoising else 0.1

    dwav, sr = torchaudio.load(path)
    dwav = dwav.mean(dim=0)

    wav1, new_sr = denoise(dwav, sr, device)
    wav2, new_sr = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=lambd, tau=tau)

    wav1 = wav1.cpu().numpy()
    wav2 = wav2.cpu().numpy()

    return (new_sr, wav1), (new_sr, wav2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlimited", action="store_true")
    args = parser.parse_args()

    inputs: list = [
        gr.Audio(type="filepath", label="Input Audio"),
        gr.Dropdown(
            choices=["Midpoint", "RK4", "Euler"],
            value="Midpoint",
            label="CFM ODE Solver (Midpoint is recommended)",
        ),
        gr.Slider(
            minimum=1,
            maximum=128,
            value=64,
            step=1,
            label="CFM Number of Function Evaluations (higher values in general yield better quality but may be slower)",
        ),
        gr.Slider(
            minimum=0,
            maximum=1,
            value=0.5,
            step=0.01,
            label="CFM Prior Temperature (higher values can improve quality but can reduce stability)",
        ),
        gr.Checkbox(
            value=False,
            label="Denoise Before Enhancement (tick if your audio contains heavy background noise)",
        ),
    ]

    outputs: list = [
        gr.Audio(label="Output Denoised Audio"),
        gr.Audio(label="Output Enhanced Audio"),
    ]

    interface = gr.Interface(
        fn=partial(_fn, unlimited=args.unlimited),
        title="Resemble Enhance",
        description="AI-driven audio enhancement for your audio files, powered by Resemble AI.",
        inputs=inputs,
        outputs=outputs,
    )

    interface.launch()


if __name__ == "__main__":
    main()
