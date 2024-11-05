# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import torch
import torchaudio
import zipfile
import tempfile
import io
from typing import Union

from resemble_enhance.inference import inference
from resemble_enhance.enhancer.train import Enhancer, HParams
from resemble_enhance.enhancer.download import download


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        run_dir = download(None) # download the latest model
        hp = HParams.load(run_dir)
        enhancer = Enhancer(hp)
        path = run_dir / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
        state_dict = torch.load(path, map_location="cpu")["module"]
        enhancer.load_state_dict(state_dict)
        enhancer.eval()
        enhancer.to("cuda")
        self.enhancer = enhancer

    def denoise(self, dwav, sr):
        return inference(model=self.enhancer.denoiser, dwav=dwav, sr=sr, device="cuda")
    
    def enhance(self, dwav, sr, nfe=32, solver="midpoint", lambd=0.5, tau=0.5, denoise=True):
        self.enhancer.configurate_(nfe=nfe, solver=solver, lambd=lambd, tau=tau, denoise=denoise)
        return inference(model=self.enhancer, dwav=dwav, sr=sr, device="cuda")

    def predict(
        self,
        audio_file: Path = Input(description="Input audio file or ZIP containing multiple WAV files"),
        denoise_only: bool = Input(
            description="Only apply denoising without enhancement", default=False
        ),
        enhance_only: bool = Input(
            description="Only apply enhancement without denoising (will be ignored if denoise_only is True)", default=False
        ),
        lambd: float = Input(
            description="Denoise strength for enhancement (0.0 to 1.0)", 
            ge=0, le=1.0, 
            default=1.0
        ),
        tau: float = Input(
            description="CFM prior temperature (0.0 to 1.0)", 
            ge=0, le=1.0, 
            default=0.5
        ),
        solver: str = Input(
            description="Solver for CFM prior (midpoint, rk4, euler)",
            choices=["midpoint", "rk4", "euler"],
            default="midpoint"
        ),
        nfe: int = Input(
            description="Number of function evaluations",
            ge=1, le=128,
            default=64
        ),
    ) -> Path:
        """Run audio enhancement on the input file(s)"""
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        temp_dir = Path(tempfile.mkdtemp())
        
        # Check if input is a zip file
        if str(audio_file).lower().endswith('.zip'):
            # Process zip file
            output_dir = temp_dir / "enhanced"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(audio_file, 'r') as zip_ref:
                # Filter only wav files
                wav_files = [f for f in zip_ref.namelist() if f.lower().endswith('.wav')]
                
                for wav_file in wav_files:
                    # Read wav file from zip
                    with zip_ref.open(wav_file) as file:
                        # Convert to bytes IO for torchaudio
                        wav_bytes = io.BytesIO(file.read())
                        dwav, sr = torchaudio.load(wav_bytes)
                        dwav = dwav.mean(0)  # Convert to mono
                        
                        # Process audio
                        if denoise_only:
                            hwav, sr = self.denoise(
                                dwav=dwav,
                                sr=sr,
                            )
                        elif enhance_only:
                            hwav, sr = self.enhance(
                                dwav=dwav,
                                sr=sr,
                                nfe=nfe,
                                solver=solver,
                                lambd=lambd,
                                tau=tau,
                                denoise=False
                            )
                        else:
                            hwav, sr = self.enhance(
                                dwav=dwav,
                                sr=sr,
                                nfe=nfe,
                                solver=solver,
                                lambd=lambd,
                                tau=tau,
                                denoise=True
                            )
                        
                        # Save enhanced audio
                        out_path = output_dir / Path(wav_file).name
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        torchaudio.save(out_path, hwav[None], sr)
            
            # Create output zip file
            output_zip = temp_dir / "enhanced.zip"
            with zipfile.ZipFile(output_zip, 'w') as zip_out:
                for wav_path in output_dir.rglob('*.wav'):
                    zip_out.write(wav_path, wav_path.relative_to(output_dir))
            
            return Path(output_zip)
        
        else:
            # Process single audio file
            dwav, sr = torchaudio.load(str(audio_file))
            dwav = dwav.mean(0)  # Convert to mono

            if denoise_only:
                hwav, sr = self.denoise(
                    dwav=dwav,
                    sr=sr,
                )
            elif enhance_only:
                hwav, sr = self.enhance(
                    dwav=dwav,
                    sr=sr,
                    nfe=nfe,
                    solver=solver,
                    lambd=lambd,
                    tau=tau,
                    denoise=False
                )
            else:
                hwav, sr = self.enhance(
                    dwav=dwav,
                    sr=sr,
                    nfe=nfe,
                    solver=solver,
                    lambd=lambd,
                    tau=tau,
                    denoise=True
                )
    
            output_path = temp_dir / "enhanced.wav"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(output_path, hwav[None], sr)
            
            return Path(output_path)
